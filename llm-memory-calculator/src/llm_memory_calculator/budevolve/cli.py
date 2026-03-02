"""BudEvolve CLI — reverse-optimization for LLM serving."""
import argparse
import json
import sys
from typing import List


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="budsim-evolve",
        description="BudEvolve: Reverse-optimization for LLM serving",
    )
    subparsers = parser.add_subparsers(dest="command", help="Optimization mode")

    # optimize
    opt = subparsers.add_parser("optimize", help="Multi-objective config optimization")
    opt.add_argument("--model", required=True, help="Model name (HuggingFace ID)")
    opt.add_argument("--hardware", required=True, help="Hardware name (e.g. H100_GPU)")
    opt.add_argument("--objectives", default="throughput,latency", help="Comma-separated objectives")
    opt.add_argument("--constraints", default="", help="Constraints e.g. 'ttft<500,tpot<50'")
    opt.add_argument("--generations", type=int, default=100)
    opt.add_argument("--pop-size", type=int, default=50)
    opt.add_argument("--input-tokens", type=int, default=512)
    opt.add_argument("--output-tokens", type=int, default=128)
    opt.add_argument("--output", default=None, help="Output JSON file")

    # explore-hardware
    hw = subparsers.add_parser("explore-hardware", help="Hardware design space exploration")
    hw.add_argument("--model", required=True)
    hw.add_argument("--objectives", default="throughput,cost")
    hw.add_argument("--constraints", default="")
    hw.add_argument("--generations", type=int, default=100)
    hw.add_argument("--output", default=None)

    # what-if
    wi = subparsers.add_parser("what-if", help="What-if hardware parameter sweep")
    wi.add_argument("--base-hardware", required=True)
    wi.add_argument("--param", required=True, help="Parameter to sweep")
    wi.add_argument("--range", required=True, help="lo,hi range")
    wi.add_argument("--steps", type=int, default=20)
    wi.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    wi.add_argument("--output", default=None)

    # sensitivity
    sa = subparsers.add_parser("sensitivity", help="Parameter sensitivity analysis")
    sa.add_argument("--model", required=True)
    sa.add_argument("--hardware", required=True)
    sa.add_argument("--target", default="throughput")
    sa.add_argument("--mode", choices=["config", "hardware", "both"], default="config")
    sa.add_argument("--output", default=None)

    # evolve-scheduler
    es = subparsers.add_parser("evolve-scheduler", help="Evolve scheduling algorithm")
    es.add_argument("--model", required=True)
    es.add_argument("--hardware", required=True)
    es.add_argument("--iterations", type=int, default=100)
    es.add_argument("--llm-endpoint", default="https://api.together.xyz/v1")
    es.add_argument("--llm-model", default="openai/gpt-oss-120b")
    es.add_argument("--output", default="evolved_scheduler")

    # evolve-cache-policy
    ec = subparsers.add_parser("evolve-cache-policy", help="Evolve KV cache eviction policy")
    ec.add_argument("--model", required=True)
    ec.add_argument("--hardware", required=True)
    ec.add_argument("--iterations", type=int, default=100)
    ec.add_argument("--llm-endpoint", default="https://api.together.xyz/v1")
    ec.add_argument("--llm-model", default="openai/gpt-oss-120b")
    ec.add_argument("--output", default="evolved_cache_policy")

    # evolve-cpu-scheduler
    cpu = subparsers.add_parser("evolve-cpu-scheduler", help="Evolve CPU-optimized scheduling algorithm")
    cpu.add_argument("--model", required=True)
    cpu.add_argument("--hardware", required=True, help="CPU hardware (e.g. GraniteRapids_CPU, Turin_CPU)")
    cpu.add_argument("--iterations", type=int, default=100)
    cpu.add_argument("--llm-endpoint", default="https://api.together.xyz/v1")
    cpu.add_argument("--llm-model", default="openai/gpt-oss-120b")
    cpu.add_argument("--output", default="evolved_cpu_scheduler")

    return parser


def _parse_constraints(s: str) -> dict:
    """Parse constraint string like 'ttft<500,tpot<50' into dict."""
    if not s:
        return {}
    constraints = {}
    for part in s.split(","):
        part = part.strip()
        if "<" in part:
            key, val = part.split("<")
            constraints[f"max_{key.strip()}_ms"] = float(val.strip())
    return constraints


def main(argv: List[str] = None):
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    result = None

    if args.command == "optimize":
        from .numeric.config_optimizer import NumericOptimizer
        optimizer = NumericOptimizer(model=args.model, hardware=args.hardware)
        result = optimizer.optimize(
            objectives=args.objectives.split(","),
            constraints=_parse_constraints(args.constraints),
            n_generations=args.generations,
            pop_size=args.pop_size,
            input_tokens=args.input_tokens,
            output_tokens=args.output_tokens,
        )
        from dataclasses import asdict
        result = asdict(result)

    elif args.command == "explore-hardware":
        from .numeric.hardware_explorer import HardwareExplorer
        explorer = HardwareExplorer(model=args.model)
        result = explorer.explore(
            objectives=args.objectives.split(","),
            constraints=_parse_constraints(args.constraints),
            n_generations=args.generations,
        )
        from dataclasses import asdict
        result = asdict(result)

    elif args.command == "what-if":
        from .numeric.hardware_explorer import HardwareExplorer
        explorer = HardwareExplorer(model=args.model)
        lo, hi = [float(x) for x in args.range.split(",")]
        result = explorer.what_if(
            base_hardware=args.base_hardware,
            param=args.param,
            param_range=(lo, hi),
            steps=args.steps,
        )

    elif args.command == "sensitivity":
        from .numeric.sensitivity import SensitivityAnalyzer
        analyzer = SensitivityAnalyzer(model=args.model, hardware=args.hardware)
        if args.mode in ("config", "both"):
            result = analyzer.analyze(target=args.target)
        if args.mode in ("hardware", "both"):
            hw_result = analyzer.analyze_hardware(target=args.target)
            if result:
                result["hardware"] = hw_result
            else:
                result = hw_result

    elif args.command == "evolve-scheduler":
        from .evolve.algorithm_evolver import AlgorithmEvolver
        evolver = AlgorithmEvolver(
            model=args.model, hardware=args.hardware,
            llm_endpoint=args.llm_endpoint, llm_model=args.llm_model,
        )
        result = evolver.evolve_scheduler(
            iterations=args.iterations,
            output_dir=args.output,
        )

    elif args.command == "evolve-cache-policy":
        from .evolve.algorithm_evolver import AlgorithmEvolver
        evolver = AlgorithmEvolver(
            model=args.model, hardware=args.hardware,
            llm_endpoint=args.llm_endpoint, llm_model=args.llm_model,
        )
        result = evolver.evolve_cache_policy(
            iterations=args.iterations,
            output_dir=args.output,
        )

    elif args.command == "evolve-cpu-scheduler":
        from .evolve.algorithm_evolver import AlgorithmEvolver
        evolver = AlgorithmEvolver(
            model=args.model, hardware=args.hardware,
            llm_endpoint=args.llm_endpoint, llm_model=args.llm_model,
        )
        result = evolver.evolve_cpu_scheduler(
            iterations=args.iterations,
            output_dir=args.output,
        )

    if result is not None:
        output_str = json.dumps(result, indent=2, default=str)
        if args.output and args.command not in ("evolve-scheduler", "evolve-cache-policy", "evolve-cpu-scheduler"):
            with open(args.output, "w") as f:
                f.write(output_str)
            print(f"Results written to {args.output}")
        else:
            print(output_str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
