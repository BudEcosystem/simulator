"""Bridge between OpenEvolve's evaluator interface and BudSim.

R2-BE1: the evolved candidate is a pure function (e.g. ``schedule_batch`` /
``schedule_batch_cpu`` / ``evict_entries``). Previously the bridge scored a candidate by
its AST *shape* only and NEVER executed it, so fitness was candidate-independent -- two
different schedulers produced identical ``throughput_rps``/``combined_score``. We now
EXECUTE the candidate in a sandboxed thread over a single-source synthetic workload
(``WorkloadConfig`` defaults, seed=42) and fold an executed POLICY-EFFICIENCY factor
``E in [eps, 1]`` into the score::

    perf = genz_throughput_ceiling * slo * E

``E`` is the realized quality of the policy (batch token-fill efficiency for schedulers,
retained-hot-entry fraction for cache eviction). This is NOT a full-lifecycle simulation:
the candidate has no injection point into ``BatchScheduler`` / ``ServingSimulator``. The
result carries ``fitness_basis`` and ``policy_efficiency`` so the API never implies more
than a per-policy-call efficiency measurement.
"""
import ast
import builtins as _builtins
import multiprocessing as mp
import warnings
from typing import Dict, List, Optional, Any, Callable

from ..types import ServingConfig, EvalResult
from ..evaluator import BudSimEvaluator
from ..roofline_analyzer import RooflineAnalyzer

# Imports the sandboxed candidate is allowed to use (reused AST allowlist).
_ALLOWED_IMPORTS = ("math", "collections", "itertools", "functools", "typing")

# Minimal builtins whitelist for executing an untrusted candidate. The previous sandbox copied the
# ENTIRE real builtins dict (only overriding __import__), leaving open/eval/exec/compile and the
# object.__subclasses__() gadget live -> an adversarial candidate could read/write files or reach
# os.system. We now expose ONLY pure, side-effect-free names a scheduling/eviction policy legitimately
# needs. open/eval/exec/compile/__import__/globals/vars/locals/input/getattr/setattr/delattr/memoryview
# are deliberately ABSENT, and the AST validator additionally rejects dunder-attribute gadgets.
_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "format",
    "frozenset", "int", "len", "list", "map", "max", "min", "pow", "range", "reversed", "round",
    "set", "slice", "sorted", "str", "sum", "tuple", "zip", "bytes", "complex", "hash", "repr",
    "True", "False", "None", "isinstance", "issubclass", "ord", "chr", "bin", "hex", "oct",
    "hasattr",  # safe (returns bool only); the base scheduler/eviction policies use attribute access
    "ValueError", "TypeError", "KeyError", "IndexError", "StopIteration", "Exception",
)
_SAFE_BUILTINS = {n: getattr(_builtins, n) for n in _SAFE_BUILTIN_NAMES if hasattr(_builtins, n)}
# Class definition (`class X: ...`) compiles to a __build_class__ call. It is benign — creating a
# class is not an escape (dunder access / getattr-gadget / open-eval-exec all remain blocked) — and a
# legitimate policy may define a small helper class, so it is permitted.
if hasattr(_builtins, "__build_class__"):
    _SAFE_BUILTINS["__build_class__"] = _builtins.__build_class__

# Builtin names a candidate must never call (the dangerous ones the whitelist already omits, but we
# also reject them at AST-validation time so a malicious candidate is caught BEFORE execution).
# NOTE: getattr is NOT forbidden — the base policies legitimately use getattr(req, 'input_tokens', d);
# instead the sandbox injects a GUARDED getattr that blocks dunder names (the gadget path).
_FORBIDDEN_CALL_NAMES = frozenset({
    "open", "eval", "exec", "compile", "__import__", "globals", "vars", "locals", "input",
    "setattr", "delattr", "memoryview", "breakpoint", "exit", "quit", "help",
})
# Dunder attribute access used by sandbox-escape gadgets (object.__subclasses__(), f.__globals__, ...).
_FORBIDDEN_ATTRS = frozenset({
    "__subclasses__", "__class__", "__bases__", "__mro__", "__globals__", "__builtins__",
    "__import__", "__dict__", "__getattribute__", "__base__", "__subclasshook__", "__reduce__",
    "__reduce_ex__", "__code__", "__closure__", "__func__", "__self__", "__module__",
})

# Floor on the executed policy-efficiency factor. A syntactically valid, executable
# candidate keeps combined_score > 0 even if its realized efficiency is ~0, which preserves
# the mocked bridge tests (they assert combined_score > 0 for a working candidate).
_EFFICIENCY_EPS = 1e-3

# Sandbox execution budget. A single policy call over a bounded queue is trivial; 2.0s
# generously bounds runaway candidates (infinite loops) without flaking on a loaded host.
_SANDBOX_TIMEOUT_S = 2.0

# Bound on the synthetic queue handed to a candidate (defensive cap on memory / runtime).
_MAX_QUEUE = 2048

_FITNESS_BASIS = "executed_policy_efficiency x genz_throughput_ceiling"


class _SynthRequest:
    """Lightweight request exposing the attributes scheduler candidates read.

    Mirrors the contract of the base algorithms (``.input_tokens``, ``.output_tokens``,
    ``.priority``, ``.arrival_time``) without dragging in the full serving ``Request``
    (which uses ``max_output_tokens`` and many lifecycle fields a scheduler must not need).
    """

    __slots__ = ("input_tokens", "output_tokens", "priority", "arrival_time", "request_id")

    def __init__(self, request_id, input_tokens, output_tokens, arrival_time):
        self.request_id = request_id
        self.input_tokens = int(input_tokens)
        self.output_tokens = int(output_tokens)
        self.priority = 0
        self.arrival_time = arrival_time


class _SynthCacheEntry:
    """Lightweight cache entry exposing the attributes eviction candidates read."""

    __slots__ = ("entry_id", "last_access_time", "frequency", "size_bytes", "prefix_length")

    def __init__(self, entry_id, last_access_time, frequency, size_bytes, prefix_length):
        self.entry_id = entry_id
        self.last_access_time = last_access_time
        self.frequency = frequency
        self.size_bytes = size_bytes
        self.prefix_length = prefix_length


def _build_synthetic_queue(max_tokens_budget: int) -> List[_SynthRequest]:
    """Build a deterministic synthetic request queue (seed=42, single source).

    Uses the GenZ ``WorkloadConfig`` defaults so the workload matches the rest of the
    serving stack. Bounded at ``_MAX_QUEUE`` requests.
    """
    from llm_memory_calculator.genz.serving.workload import (
        WorkloadConfig, WorkloadGenerator,
    )

    cfg = WorkloadConfig()  # defaults: seed=42, lognormal input ~512, exponential output ~128
    requests = WorkloadGenerator(cfg).generate()
    requests = requests[:_MAX_QUEUE]
    return [
        _SynthRequest(
            request_id=r.request_id,
            input_tokens=r.input_tokens,
            output_tokens=r.max_output_tokens,
            arrival_time=r.arrival_time_ns,
        )
        for r in requests
    ]


def _build_synthetic_cache(n: int = 256) -> List[_SynthCacheEntry]:
    """Build a deterministic synthetic cache (seed=42) with a hot/cold structure.

    A well-defined "hot" subset (recently accessed + high frequency) lets us measure how
    much of the hot working set an eviction policy *retains*.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    n = min(n, _MAX_QUEUE)
    entries = []
    now = 1_000_000.0
    for i in range(n):
        # Half the entries are hot (recent + frequent), half cold.
        hot = i % 2 == 0
        last_access = now - (rng.uniform(0, 100) if hot else rng.uniform(1000, 5000))
        frequency = int(rng.uniform(50, 100) if hot else rng.uniform(1, 10))
        entries.append(_SynthCacheEntry(
            entry_id=i,
            last_access_time=float(last_access),
            frequency=frequency,
            size_bytes=int(rng.uniform(1024, 65536)),
            prefix_length=int(rng.uniform(16, 512)),
        ))
    return entries


def _hot_entry_ids(entries: List[_SynthCacheEntry]) -> set:
    """Identify the "hot" entries: top half by a recency+frequency score."""
    if not entries:
        return set()
    now = max(e.last_access_time for e in entries)
    scored = sorted(
        entries,
        key=lambda e: e.frequency - (now - e.last_access_time) * 1e-3,
        reverse=True,
    )
    k = max(1, len(scored) // 2)
    return {id(e) for e in scored[:k]}


def _load_candidate_fn(code: str):
    """Compile the candidate in the minimal-builtins sandbox; return (name, fn) or None.

    Uses _SAFE_BUILTINS (no open/eval/exec/import) so even if AST validation were bypassed the
    runtime environment cannot perform I/O. Called inside the worker SUBPROCESS."""
    def _guarded_import(name, *args, **kwargs):
        root = name.split(".")[0]
        if root not in _ALLOWED_IMPORTS:
            raise ImportError(f"import of {name!r} is not allowed in sandbox")
        return __import__(name, *args, **kwargs)

    def _guarded_getattr(obj, name, *default):
        # allow normal attribute reads (req.input_tokens) but block the dunder-gadget escape path
        # (getattr(x, '__class__').__bases__... -> object.__subclasses__() -> os).
        if isinstance(name, str) and name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"access to dunder attribute {name!r} is not allowed in sandbox")
        return getattr(obj, name, *default)

    safe_builtins = dict(_SAFE_BUILTINS)
    safe_builtins["__import__"] = _guarded_import
    safe_builtins["getattr"] = _guarded_getattr
    # __name__ is required for `class X:` to resolve its __module__; it is an inert string label.
    ns: Dict[str, Any] = {"__builtins__": safe_builtins, "__name__": "candidate"}
    try:
        exec(compile(code, "<candidate>", "exec"), ns)
    except Exception:
        return None
    for name in ("schedule_batch", "schedule_batch_cpu", "evict_entries"):
        fn = ns.get(name)
        if callable(fn):
            return name, fn
    return None


def _score_policy_inproc(code: str, slo: float):
    """Load + execute + score a candidate, returning (E, error). Runs INSIDE the worker subprocess
    so the subprocess itself (not an un-killable thread) is the timeout/isolation boundary."""
    loaded = _load_candidate_fn(code)
    if loaded is None:
        return _EFFICIENCY_EPS, "candidate did not define a known policy function"
    fn_name, fn = loaded
    try:
        if fn_name in ("schedule_batch", "schedule_batch_cpu"):
            return _score_scheduler_result(fn, fn_name, slo)
        if fn_name == "evict_entries":
            return _score_cache_result(fn)
    except BaseException as e:  # candidate code is untrusted
        return _EFFICIENCY_EPS, f"{type(e).__name__}: {e}"
    return _EFFICIENCY_EPS, f"unsupported policy: {fn_name}"


def _score_scheduler_result(fn: Callable, fn_name: str, slo: float):
    """E for scheduler families = batch token-fill efficiency, SLO-down-weighted.

    Gaming-hardened: the returned batch is DEDUPED by identity (a candidate returning
    [queue[0]]*100000 cannot inflate the fill past the true unique tokens) and capped at the token
    budget."""
    max_tokens = 8192 if fn_name == "schedule_batch" else 16384
    queue = _build_synthetic_queue(max_tokens)
    batch = fn(queue)

    if not isinstance(batch, list):
        return _EFFICIENCY_EPS, "schedule contract violation: did not return a list"
    queue_ids = {id(r) for r in queue}
    if any(id(r) not in queue_ids for r in batch):
        return _EFFICIENCY_EPS, "schedule contract violation: returned foreign requests"

    # Dedup by identity so duplicating a request cannot game the fill (gaming fix).
    seen, unique = set(), []
    for r in batch:
        if id(r) not in seen:
            seen.add(id(r))
            unique.append(r)
    batch_tokens = sum(
        getattr(r, "input_tokens", 0) + getattr(r, "output_tokens", 0) for r in unique
    )
    fill = max(0.0, min(batch_tokens / float(max_tokens), 1.0))
    slo = slo if slo and slo > 0 else 1.0
    return _floor_eff(fill * slo), None


def _score_cache_result(fn: Callable):
    """E for the cache family = retained-hot-entry fraction, scaled by whether the policy actually
    freed the requested space (a candidate that evicts NOTHING retains all hot entries but did not do
    its job -> gets ~0, not 1)."""
    entries = _build_synthetic_cache()
    hot_ids = _hot_entry_ids(entries)
    num_to_evict = len(entries) // 2
    evicted = fn(entries, num_to_evict)

    if not isinstance(evicted, list):
        return _EFFICIENCY_EPS, "evict contract violation: did not return a list"
    entry_ids = {id(e) for e in entries}
    if any(id(e) not in entry_ids for e in evicted):
        return _EFFICIENCY_EPS, "evict contract violation: returned foreign entries"
    if not hot_ids:
        return _EFFICIENCY_EPS, "no hot entries in synthetic cache"

    evicted_ids = {id(e) for e in evicted}  # set => duplicate evictions don't inflate the count
    retained_ids = entry_ids - evicted_ids
    retained_hot = len(hot_ids & retained_ids)
    retain_frac = retained_hot / float(len(hot_ids))
    # Must actually free ~num_to_evict slots: a policy that evicts far fewer didn't do its job.
    freed_frac = min(1.0, len(evicted_ids) / float(num_to_evict)) if num_to_evict else 1.0
    return _floor_eff(retain_frac * freed_frac), None


def _floor_eff(efficiency: float) -> float:
    """Clamp efficiency into [eps, 1] so a valid candidate keeps combined_score > 0."""
    if efficiency != efficiency:  # NaN
        return _EFFICIENCY_EPS
    return max(_EFFICIENCY_EPS, min(float(efficiency), 1.0))


def _policy_worker(code: str, slo: float, q) -> None:
    """Subprocess entry point: score the candidate and put (E, error) on the queue."""
    try:
        result = _score_policy_inproc(code, slo)
    except BaseException as e:
        result = (_EFFICIENCY_EPS, f"worker error: {type(e).__name__}: {e}")
    try:
        q.put(result)
    except Exception:
        pass


def _run_candidate_sandboxed(code: str, slo: float):
    """Execute the candidate in a KILLABLE subprocess with a hard timeout, returning (E, error).

    Replaces the prior daemon-thread runner: a Python thread cannot be killed, so an infinite-loop
    candidate leaked a GIL-contending thread forever. A subprocess can be terminated, and also gives
    real (separate-address-space) isolation on top of the minimal-builtins sandbox. Falls back to a
    direct in-process call only if multiprocessing is unavailable (the builtins whitelist still
    prevents I/O in that case)."""
    try:
        ctx = mp.get_context("fork")
        q = ctx.Queue()
        p = ctx.Process(target=_policy_worker, args=(code, slo, q), daemon=True)
        p.start()
        p.join(_SANDBOX_TIMEOUT_S)
        if p.is_alive():
            p.terminate()
            p.join(1.0)
            if p.is_alive():
                p.kill()
                p.join(1.0)
            return _EFFICIENCY_EPS, f"timeout after {_SANDBOX_TIMEOUT_S}s (terminated)"
        try:
            return q.get(timeout=1.0)
        except Exception:
            return _EFFICIENCY_EPS, "candidate produced no result (crashed)"
    except Exception:
        # multiprocessing unavailable (e.g. restricted env): direct call, no kill-on-timeout, but
        # the minimal-builtins sandbox still prevents I/O / escapes.
        return _score_policy_inproc(code, slo)


class BudSimEvalBridge:
    """OpenEvolve-compatible evaluator that uses BudSim as fitness function.

    OpenEvolve expects an evaluate(program_path) -> dict function.
    This bridge wraps BudSimEvaluator to provide that interface.

    The evolved candidate is loaded, validated (syntax + safety), EXECUTED in a sandbox
    over the synthetic workload, and its realized policy efficiency is folded into the
    GenZ-derived throughput ceiling to produce a candidate-dependent fitness score.
    """

    def __init__(
        self,
        model: str,
        hardware: str,
        input_tokens: int = 512,
        output_tokens: int = 128,
        batch_size: int = 32,
        precision: str = "bf16",
    ):
        self._model = model
        self._hardware = hardware
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._batch_size = batch_size
        self._precision = precision
        self._evaluator = BudSimEvaluator()
        self._roofline = RooflineAnalyzer()

    def evaluate(self, program_path: str) -> Dict:
        """Evaluate an evolved algorithm using BudSim.

        Loads the evolved code, validates its syntax and safety, EXECUTES the candidate
        policy over the synthetic workload to measure a policy-efficiency factor E, then
        folds E into the GenZ throughput ceiling: combined ~= genz_ceiling * slo * E.

        Args:
            program_path: Path to evolved Python file from OpenEvolve.

        Returns:
            Dict with 'combined_score' and individual metrics, including
            'policy_efficiency' (E) and 'fitness_basis'.
        """
        # Stage 1: Load and validate the evolved code
        code_quality = self._validate_evolved_code(program_path)
        if not code_quality["valid"]:
            return {
                "combined_score": 0.0,
                "throughput_rps": 0.0,
                "ttft_ms": float("inf"),
                "tpot_ms": float("inf"),
                "slo_compliance": 0.0,
                "feasible": 0.0,
                "code_quality": 0.0,
                "policy_efficiency": 0.0,
                "fitness_basis": _FITNESS_BASIS,
                "error": code_quality.get("error", "invalid code"),
            }

        # Stage 2: BudSim performance evaluation (GenZ throughput ceiling, candidate-independent).
        cfg = ServingConfig(
            model=self._model, hardware=self._hardware,
            batch_size=self._batch_size, precision=self._precision,
        )
        result = self._evaluator.evaluate_config(
            cfg, input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )

        genz_ceiling = result.throughput_rps
        slo = result.slo_compliance if result.slo_compliance > 0 else 1.0

        # Stage 3: EXECUTE the candidate to obtain a candidate-dependent efficiency E.
        efficiency, exec_error = self._measure_policy_efficiency(
            code_quality["code"], result,
        )

        quality_bonus = code_quality["score"]  # 0.0 to 1.0 (kept as a minor signal)

        # R2-BE1: fitness is now candidate-dependent via the executed efficiency factor E.
        perf_score = genz_ceiling * slo * efficiency
        combined = 0.8 * perf_score + 0.2 * perf_score * quality_bonus

        out = {
            "combined_score": combined,
            "throughput_rps": genz_ceiling,
            "ttft_ms": result.ttft_ms,
            "tpot_ms": result.tpot_ms,
            "slo_compliance": slo,
            "feasible": 1.0 if result.feasible else 0.0,
            "code_quality": quality_bonus,
            "policy_efficiency": efficiency,
            "fitness_basis": _FITNESS_BASIS,
        }
        if exec_error:
            out["error"] = exec_error
        return out

    # ------------------------------------------------------------------ #
    # Sandboxed candidate execution                                       #
    # ------------------------------------------------------------------ #

    def _measure_policy_efficiency(self, code: str, result: EvalResult):
        """Execute the candidate (in a killable sandbox subprocess) and return (E, error).

        E is the realized policy efficiency in [eps, 1]; on timeout / exception / contract-violation,
        E = eps-floored 0 and an error string is returned. Delegates to the module-level
        ``_run_candidate_sandboxed`` so an infinite-loop candidate is TERMINATED (not leaked as a
        GIL-spinning thread) and runs with the minimal-builtins sandbox in a separate address space.
        """
        slo = result.slo_compliance if getattr(result, "slo_compliance", 0) else 0.0
        return _run_candidate_sandboxed(code, slo)

    # ------------------------------------------------------------------ #
    # Validation (syntax + safety + structural quality)                   #
    # ------------------------------------------------------------------ #

    def _validate_evolved_code(self, program_path: str) -> Dict:
        """Validate evolved code: syntax check, safety check, quality score.

        Args:
            program_path: Path to the evolved Python file.

        Returns:
            Dict with 'valid' (bool), 'score' (float 0-1), 'code' (str on success),
            'error' (str if invalid).
        """
        # Read the file
        try:
            with open(program_path, "r") as f:
                code = f.read()
        except (FileNotFoundError, PermissionError, OSError) as e:
            return {"valid": False, "score": 0.0, "error": f"Cannot read file: {e}"}

        if not code.strip():
            return {"valid": False, "score": 0.0, "error": "Empty file"}

        # Syntax validation via AST parse
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"valid": False, "score": 0.0, "error": f"Syntax error: {e}"}

        # Safety check: reject dangerous constructs BEFORE the code is ever executed. The previous
        # version only inspected imports, which let a candidate call open()/eval()/exec() or reach
        # os via the object.__subclasses__() gadget (the runtime _SAFE_BUILTINS whitelist is the
        # second line of defense; this is the first). We reject: non-allowlisted imports, calls to
        # forbidden builtins by bare name, and access to escape-gadget dunder attributes.
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] not in _ALLOWED_IMPORTS:
                        return {"valid": False, "score": 0.0,
                                "error": f"Unsafe import: {alias.name}"}
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] not in _ALLOWED_IMPORTS:
                    return {"valid": False, "score": 0.0,
                            "error": f"Unsafe import from: {node.module}"}
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in _FORBIDDEN_CALL_NAMES:
                    return {"valid": False, "score": 0.0,
                            "error": f"Unsafe builtin call: {node.func.id}()"}
            elif isinstance(node, ast.Attribute):
                if node.attr in _FORBIDDEN_ATTRS:
                    return {"valid": False, "score": 0.0,
                            "error": f"Unsafe attribute access: .{node.attr}"}
            elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_CALL_NAMES:
                # e.g. assigning `f = open` then calling f(...) — reject the bare reference too.
                return {"valid": False, "score": 0.0,
                        "error": f"Unsafe builtin reference: {node.id}"}

        # Structural quality scoring
        score = self._score_code_structure(tree, code)
        return {"valid": True, "score": score, "code": code}

    def _score_code_structure(self, tree: ast.AST, code: str) -> float:
        """Score the structural quality of evolved code.

        Rewards: function definitions, conditional logic, docstrings.
        Penalizes: overly long functions, no functions, trivial code.

        Args:
            tree: Parsed AST of the code.
            code: Raw source code string.

        Returns:
            Quality score between 0.0 and 1.0.
        """
        score = 0.0

        # Count function definitions
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        if functions:
            score += 0.3  # Has at least one function
        else:
            return 0.1  # Trivial code without functions

        # Check for conditional logic (if/else) — indicates decision-making
        conditionals = [n for n in ast.walk(tree) if isinstance(n, (ast.If, ast.IfExp))]
        if conditionals:
            score += 0.2

        # Check for loops — indicates iteration over data
        loops = [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]
        if loops:
            score += 0.2

        # Check for return statements
        returns = [n for n in ast.walk(tree) if isinstance(n, ast.Return)]
        if returns:
            score += 0.15

        # Penalize overly short code (likely trivial)
        lines = [l for l in code.strip().split("\n") if l.strip() and not l.strip().startswith("#")]
        if len(lines) >= 5:
            score += 0.15

        return min(score, 1.0)
