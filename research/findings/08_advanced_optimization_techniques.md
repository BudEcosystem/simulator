# Advanced Optimization Techniques for vllm-tuner

Deep research into optimization frameworks, constraint handling, transfer learning, multi-fidelity methods, cost-aware optimization, simulation-in-the-loop approaches, parameter interactions, and framework comparisons.

---

## Table of Contents

1. [HEBO Framework Deep Dive](#1-hebo-framework-deep-dive)
2. [Hidden Constraint Learning](#2-hidden-constraint-learning)
3. [Transfer Learning for Bayesian Optimization](#3-transfer-learning-for-bayesian-optimization)
4. [Multi-Fidelity Optimization](#4-multi-fidelity-optimization)
5. [Cost-Aware Bayesian Optimization](#5-cost-aware-bayesian-optimization)
6. [Simulation-in-the-Loop Optimization](#6-simulation-in-the-loop-optimization)
7. [vLLM Parameter Interactions](#7-vllm-parameter-interactions)
8. [Framework Comparison: Optuna vs SMAC vs HEBO vs BoTorch](#8-framework-comparison)
9. [Consolidated Recommendations](#9-consolidated-recommendations)

---

## 1. HEBO Framework Deep Dive

### What It Is

HEBO (Heteroscedastic Evolutionary Bayesian Optimisation) is a Bayesian optimization library developed by Huawei Noah's Ark Lab. It won the NeurIPS 2020 Black-Box Optimization Challenge and has been shown to significantly outperform existing black-box optimizers on 108 machine learning hyperparameter tuning tasks (Bayesmark benchmark). SCOOT (the primary paper analyzed in Task #3) is built directly on HEBO.

**Source**: https://github.com/huawei-noah/HEBO, https://arxiv.org/abs/2012.03826

### How It Works

HEBO adds four key enhancements to classical Bayesian Optimization:

1. **Output Warping**: Applies non-linear transformations to the objective function values to handle heteroscedasticity (non-constant variance). Real-world tuning problems often exhibit this -- e.g., throughput variance is much higher in some regions of the parameter space than others. HEBO uses a power transform to stabilize variance.

2. **Input Warping**: Applies non-linear transformations to the input space to handle non-stationarity. The relationship between parameters and performance may vary across regions. Input warping via a Beta CDF allows the GP to adapt its length scales locally.

3. **Multi-Objective Acquisition Ensembles**: Instead of using a single acquisition function (like EI or UCB), HEBO uses a multi-objective formulation where different acquisition functions are treated as competing objectives. It finds the Pareto frontier of acquisition values and randomly selects from it. This prevents the failure mode where a single acquisition function is poorly suited to the current optimization landscape.

4. **Evolutionary Acquisition Maximization**: Uses a genetic algorithm (instead of gradient-based methods) to maximize the acquisition function, making it robust to non-smooth or multi-modal acquisition landscapes.

### How It Compares to Optuna TPE

| Aspect | HEBO | Optuna TPE |
|--------|------|------------|
| **Surrogate model** | Gaussian Process (Matern 5/2 kernel) | Tree-structured Parzen Estimator |
| **Sample efficiency** | Higher -- GP models the full posterior | Lower -- TPE approximates the posterior |
| **Low-trial regime (< 30)** | Significantly better | Competitive but less reliable |
| **High-dimensional (> 20 params)** | Degrades (GP scales O(n^3)) | Better -- TPE is O(n log n) |
| **Heteroscedastic problems** | Excellent (output warping) | Poor (assumes homoscedastic) |
| **Non-stationary problems** | Good (input warping) | Moderate |
| **Multi-objective** | Native support via EHVI + ensemble | NSGA-II or MOTPE |
| **Constraint handling** | Basic (penalty-based) | Pruning + constraint sampling |
| **Speed per trial** | Slower (GP fitting is O(n^3)) | Faster |
| **Parallelism** | Constant liar strategy | Built-in distributed |
| **Ecosystem** | Smaller; available as OptunaHub sampler | Large; extensive integrations |

**Key insight from benchmarks**: HEBO is one of three engines (with Ax and SMAC) that perform significantly better than the rest, especially with moderate trial counts. Optuna TPE is faster but less sample-efficient, which matters when each trial costs 5-10 minutes of stress testing.

### Concrete Implementation Plan for vllm-tuner

**Option A: Use HEBO as Optuna Sampler (Recommended -- Lowest Risk)**

```python
# Installation
# pip install optunahub hebo

import optuna
import optunahub

module = optunahub.load_module("samplers/hebo")
sampler = module.HEBOSampler(seed=42)

study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
)
```

This approach:
- Keeps the existing Optuna-based infrastructure intact.
- Swaps only the sampling algorithm to HEBO's GP-based approach.
- Retains Optuna's pruning, logging, and visualization.
- Available via OptunaHub (`pip install optunahub hebo`).

**Option B: Use HEBO Directly (Higher Effort, Full Control)**

```python
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

space = DesignSpace().parse([
    {'name': 'max_num_seqs', 'type': 'int', 'lb': 64, 'ub': 8192},
    {'name': 'max_num_batched_tokens', 'type': 'int', 'lb': 64, 'ub': 8192},
    {'name': 'block_size', 'type': 'cat', 'categories': [8, 16, 32]},
    {'name': 'scheduler_delay_factor', 'type': 'num', 'lb': 0.0, 'ub': 2.0},
    {'name': 'enable_chunked_prefill', 'type': 'bool'},
    {'name': 'enable_prefix_caching', 'type': 'bool'},
    {'name': 'num_scheduler_steps', 'type': 'int', 'lb': 1, 'ub': 20},
    {'name': 'gpu_memory_utilization', 'type': 'num', 'lb': 0.7, 'ub': 0.95},
])

optimizer = HEBO(space, model_name='gpy')
for i in range(30):
    suggestion = optimizer.suggest(n_suggestions=1)
    # Apply known constraints to filter
    # Evaluate on real vLLM instance
    observation = evaluate(suggestion)
    optimizer.observe(suggestion, observation)
```

**For multi-objective (TTFT + TPOT):**

```python
from hebo.optimizers.general import GeneralBO

optimizer = GeneralBO(space, num_obj=2)
# Returns Pareto-optimal suggestions
```

### Expected Improvement

- **Sample efficiency**: 20-40% fewer trials needed to find near-optimal configurations (based on Bayesmark benchmarks where HEBO outperforms TPE by this margin in the 10-50 trial range).
- **Robustness**: Multi-objective acquisition ensemble eliminates failure modes where a single acquisition function is ill-suited.
- **Quality**: Output warping handles the heteroscedastic nature of LLM serving metrics (throughput variance varies greatly with configuration).

### Complexity Assessment

- **Option A**: Low. 10-20 lines of code change. Drop-in replacement.
- **Option B**: Medium. Requires rewriting the optimization loop but gains full control over constraint handling and parallel suggestion.

---

## 2. Hidden Constraint Learning

### What It Is

In LLM inference engine tuning, many parameter combinations cause the engine to crash (OOM, assertion failures, timeout errors, incompatible feature combinations). These are "hidden constraints" -- they are not documented, not known a priori, and can only be discovered by trying the configuration and observing a crash. Each crash wastes an evaluation (5-10 minutes of stress testing) and provides no useful performance information.

Hidden constraint learning uses a classifier (typically a random forest) trained on observed (config, feasible/infeasible) pairs to predict the probability that a new configuration will be feasible before evaluating it.

### How It Works

The research identifies three high-level strategies for handling hidden constraints:

**Strategy 1: Rejection** -- Simply exclude crashed configurations from the GP training set. The GP never learns about infeasible regions, so it may repeatedly suggest configurations near known crashes.

**Strategy 2: Penalty Substitution** -- Assign a large penalty value to crashed configurations. This corrupts the surrogate model because the penalty value is arbitrary and can distort the GP's predictions near constraint boundaries.

**Strategy 3: Feasibility Prediction (Recommended)** -- Train a separate classifier to predict P(feasible | config). Use this as a filter or multiplier on the acquisition function. This is what SCOOT does.

**SCOOT's Specific Implementation:**
1. Train a random forest on all observed (config, success/crash) pairs.
2. For each candidate configuration, compute POF(x) = P(feasible | x) using the random forest.
3. Only suggest configurations where POF(x) >= Delta (dynamic threshold).
4. Dynamic Delta adjustment:
   - If the suggested config crashes: Delta = min(0.75, max(0.5, Delta + 0.05))
   - After 5 consecutive feasible suggestions: Delta = max(0.25, Delta - 0.05)
   - This balances safety (avoiding crashes) with exploration (testing near boundaries).

**Boundary Exploration (BE-CBO)**: An advanced approach from recent research notes that optimal configurations often lie on the boundary between feasible and infeasible regions. BE-CBO explicitly explores this boundary, which could find configurations that are "just barely feasible" but high-performing.

**Source**: SCOOT (WWW 2025), BE-CBO (https://arxiv.org/abs/2402.07692), Local BO with Crash Constraints (https://arxiv.org/abs/2411.16267)

### Concrete Implementation Plan for vllm-tuner

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class HiddenConstraintLearner:
    def __init__(self, initial_delta=0.5, delta_increment=0.05):
        self.rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.delta = initial_delta
        self.delta_increment = delta_increment
        self.consecutive_feasible = 0
        self.observations_X = []
        self.observations_feasible = []

    def update(self, config_vector, is_feasible):
        """Add observation and update model."""
        self.observations_X.append(config_vector)
        self.observations_feasible.append(int(is_feasible))

        # Retrain RF with all observations
        if len(self.observations_X) >= 5:  # minimum samples
            self.rf.fit(
                np.array(self.observations_X),
                np.array(self.observations_feasible)
            )

        # Dynamic delta adjustment (SCOOT algorithm)
        if not is_feasible:
            self.delta = min(0.75, max(0.5, self.delta + self.delta_increment))
            self.consecutive_feasible = 0
        else:
            self.consecutive_feasible += 1
            if self.consecutive_feasible >= 5:
                self.delta = max(0.25, self.delta - self.delta_increment)
                self.consecutive_feasible = 0

    def predict_feasibility(self, config_vector):
        """Predict P(feasible | config)."""
        if len(self.observations_X) < 5:
            return 1.0  # Optimistic before enough data
        proba = self.rf.predict_proba(
            np.array(config_vector).reshape(1, -1)
        )
        # Return probability of class 1 (feasible)
        return proba[0][1] if proba.shape[1] > 1 else proba[0][0]

    def is_likely_feasible(self, config_vector):
        """Check if config meets feasibility threshold."""
        return self.predict_feasibility(config_vector) >= self.delta
```

**Integration with Optuna:**

```python
class ConstraintAwareSampler(optuna.samplers.BaseSampler):
    def __init__(self, base_sampler, constraint_learner, known_constraints):
        self.base_sampler = base_sampler
        self.constraint_learner = constraint_learner
        self.known_constraints = known_constraints

    def sample_relative(self, study, trial, search_space):
        max_retries = 50
        for _ in range(max_retries):
            params = self.base_sampler.sample_relative(study, trial, search_space)
            config_vector = self._params_to_vector(params)

            # Check known constraints first (free)
            if not self._satisfies_known_constraints(params):
                continue

            # Check learned hidden constraints
            if self.constraint_learner.is_likely_feasible(config_vector):
                return params

        # Fallback: return best-effort suggestion
        return params

    def _satisfies_known_constraints(self, params):
        """Encode vLLM's documented parameter constraints."""
        # max_num_batched_tokens >= max_num_seqs
        if params.get('max_num_batched_tokens', 0) < params.get('max_num_seqs', 0):
            return False
        # chunked_prefill and prefix_caching conflict (older vLLM)
        if params.get('enable_chunked_prefill') and params.get('enable_prefix_caching'):
            return False  # Only for vLLM < 0.6
        return True
```

### Expected Improvement

- **Crash avoidance**: Based on SCOOT's experiments, RF-based POF learning greatly enhanced SLO optimization for the BOT application where crashes were frequent. Without RF, SCOOT could not find feasible configurations for TPOT optimization at all (marked "x" in ablation results).
- **Evaluation savings**: Each avoided crash saves 5-10 minutes of wasted evaluation time. With 30% crash rate (common in vLLM tuning), saving ~9 of 30 evaluations = 45-90 minutes saved.
- **Better exploration**: Dynamic threshold allows exploring near constraint boundaries where high-performing configurations often reside.

### Complexity Assessment

Medium. The random forest classifier is straightforward (sklearn). The main complexity is:
1. Encoding vLLM parameters into a consistent feature vector.
2. Detecting crashes vs. legitimate low-performance results (need to distinguish OOM/timeout from valid but slow configs).
3. Integrating with the Optuna sampling loop.

---

## 3. Transfer Learning for Bayesian Optimization

### What It Is

Transfer learning for BO leverages observations from previous optimization tasks (source tasks) to accelerate optimization on a new task (target task). For vllm-tuner, a "task" is a specific (model, hardware, workload) combination. If we have tuned LLaMA-7B on 2xA100 with a chatbot workload, that knowledge should help tune LLaMA-13B on 2xA100 with a similar workload.

**Source**: "Transfer Learning for Bayesian Optimization: A Survey" (https://arxiv.org/abs/2302.05927)

### How It Works

Transfer learning methods for BO are categorized into four approaches based on which BO component they modify:

**1. Warm-Starting Initial Points**
Use successful configurations from past tasks as starting points for new optimization. Instead of random initialization, begin with configurations that worked well for similar (model, hardware) combos.

- **Implementation**: Store top-K configurations from each completed tuning session. For a new task, rank past tasks by similarity (same GPU type, similar model size, similar workload type) and use their top configs as initial evaluations.
- **Applicable systems**: Restune (SIGMOD 2021) uses meta-features to measure task similarity and select warm-start points.

**2. Surrogate Model Transfer**
Transfer the GP surrogate model (or its learned structure) from source tasks to the target task.

- **Ranking-Weighted GP Ensembles**: Build a separate GP for each source task. Combine them with the target GP using weights proportional to rank correlation between source and target observations. As more target observations arrive, the target GP dominates.
- **Scalable Meta-Learning for BO (Feurer & Letham)**: Uses an ensemble of GPs from past tasks, weighted by observed concordance with the new task.

**3. Acquisition Function Transfer**
Use meta-learned acquisition functions that encode knowledge about good exploration strategies across multiple tasks.

- **MetaBO (ICLR 2020)**: Uses reinforcement learning to meta-train an acquisition function on a set of related tasks. The learned acquisition function implicitly encodes structural knowledge about the task family.

**4. Search Space Transfer**
Narrow the search space based on experience from past tasks.

- **GPTuner approach**: Use LLM + past observations to identify promising subregions of the search space and focus BO there.

### LLAMBO (LLM-Enhanced Warmstarting)

**Paper**: "Large Language Models to Enhance Bayesian Optimization" (ICLR 2024)
**Source**: https://arxiv.org/abs/2402.03921

LLAMBO frames the BO problem in natural language, enabling LLMs to propose promising configurations conditioned on:
1. Parameter descriptions (name, type, range, semantics).
2. Historical observations from the current optimization.
3. Domain knowledge embedded in LLM pretraining.

LLAMBO is especially effective for zero-shot warmstarting (no prior observations) and early-stage search (few observations). Available as an OptunaHub sampler.

### Concrete Implementation Plan for vllm-tuner

**Phase 1: Configuration Database (Low Effort)**

```python
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TuningResult:
    model_name: str
    model_params_billions: float
    gpu_type: str
    num_gpus: int
    workload_type: str  # "chatbot", "classification", "recommendation", etc.
    avg_input_length: float
    avg_output_length: float
    config: dict  # The parameter configuration
    metrics: dict  # throughput, latency, TTFT, TPOT
    is_feasible: bool
    timestamp: str

class TuningKnowledgeBase:
    def __init__(self, db_path: str = "tuning_history.json"):
        self.db_path = Path(db_path)
        self.results: List[TuningResult] = self._load()

    def save_result(self, result: TuningResult):
        self.results.append(result)
        self._persist()

    def get_warmstart_configs(
        self,
        target_model_params: float,
        target_gpu_type: str,
        target_num_gpus: int,
        target_workload_type: str,
        top_k: int = 5
    ) -> List[dict]:
        """Return top-K configs from most similar past tasks."""
        scored = []
        for r in self.results:
            if not r.is_feasible:
                continue
            similarity = self._compute_similarity(
                r, target_model_params, target_gpu_type,
                target_num_gpus, target_workload_type
            )
            scored.append((similarity, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r.config for _, r in scored[:top_k]]

    def _compute_similarity(self, result, model_params, gpu_type, num_gpus, workload_type):
        score = 0.0
        # Same GPU type is most important
        if result.gpu_type == gpu_type:
            score += 3.0
        # Similar GPU count
        score += 1.0 / (1.0 + abs(result.num_gpus - num_gpus))
        # Similar model size (log scale)
        import math
        ratio = max(result.model_params_billions, model_params) / \
                max(min(result.model_params_billions, model_params), 0.1)
        score += 2.0 / ratio
        # Same workload type
        if result.workload_type == workload_type:
            score += 2.0
        return score
```

**Phase 2: LLAMBO Integration (Medium Effort)**

```python
# Use LLAMBO for zero-shot warmstarting via OptunaHub
import optunahub

llambo_module = optunahub.load_module("samplers/llambo")
llambo_sampler = llambo_module.LLAMBOSampler(
    # LLM API configuration
    model="gpt-4",  # or local model
    # Provide vLLM-specific context
    task_description="""
    Tuning vLLM inference engine parameters to maximize throughput
    while maintaining TTFT < 500ms. Model: LLaMA-7B, Hardware: 2x A100 80GB.
    Higher max_num_seqs increases throughput but may cause OOM.
    enable_chunked_prefill helps with large prompts but adds overhead.
    """,
)

# Use LLAMBO for first 5 trials, then switch to HEBO
class HybridSampler(optuna.samplers.BaseSampler):
    def __init__(self, warmstart_sampler, main_sampler, switch_trial=5):
        self.warmstart_sampler = warmstart_sampler
        self.main_sampler = main_sampler
        self.switch_trial = switch_trial

    def sample_relative(self, study, trial, search_space):
        if trial.number < self.switch_trial:
            return self.warmstart_sampler.sample_relative(study, trial, search_space)
        return self.main_sampler.sample_relative(study, trial, search_space)
```

### Expected Improvement

- **Warm-starting**: Reduces initial exploration phase by 5-10 trials (saving 25-100 minutes of evaluation time). Instead of starting from scratch, we begin near configurations known to work for similar setups.
- **LLAMBO zero-shot**: Provides reasonable initial suggestions even with no prior tuning history, leveraging LLM knowledge about vLLM parameter semantics.
- **Cross-task transfer**: As the knowledge base grows, each new tuning session starts closer to the optimum. Expected convergence speedup: 2-4x for similar tasks.

### Complexity Assessment

- **Phase 1 (Knowledge Base)**: Low. JSON storage of past results + similarity function.
- **Phase 2 (LLAMBO)**: Low-Medium. OptunaHub integration exists. Requires LLM API access.
- **Phase 3 (GP Ensemble Transfer)**: High. Requires implementing ranking-weighted GP ensembles, which is research-level code.

---

## 4. Multi-Fidelity Optimization (BOHB, Hyperband)

### What It Is

Multi-fidelity optimization uses cheap, low-fidelity approximations of the objective function to quickly eliminate bad configurations, saving the expensive high-fidelity evaluation for the most promising candidates. For vllm-tuner, the "fidelity" dimension is the benchmark duration -- a 10-second stress test is much cheaper than a 100-second test, and their results are correlated.

**Source**: "BOHB: Robust and Efficient Hyperparameter Optimization at Scale" (ICML 2018), Hyperband (JMLR 2018)

### How It Works

**Hyperband**: Aggressively allocates resources to promising configurations using successive halving:
1. Start with many configurations at lowest fidelity (e.g., 10-second benchmark).
2. Evaluate all; keep the top half.
3. Double the fidelity (20 seconds); evaluate survivors.
4. Repeat until one configuration remains at full fidelity (100+ seconds).

**BOHB (Bayesian Optimization HyperBand)**: Combines Hyperband's multi-fidelity scheduling with BO's intelligent sampling:
- Uses Hyperband to determine how many configs to evaluate at each fidelity level.
- Replaces random configuration selection with TPE-based model-guided sampling.
- Trains the surrogate model on all fidelity levels, weighting higher-fidelity observations more.

**Performance**: BOHB achieves 20x speedup over Random Search initially, improving to 55x with increasing budget.

**Key multi-fidelity concept for vllm-tuner**: The objective function (throughput under load) shows strong correlation between short (10-30s) and long (100-300s) stress tests. Short tests reveal the same parameter trends, just with more noise.

### Fidelity Dimensions for vllm-tuner

| Fidelity Level | Benchmark Duration | Request Count | Reliability | Cost |
|----------------|-------------------|---------------|-------------|------|
| Low (1) | 10 seconds | ~50-150 requests | Noisy but directional | 10s |
| Medium (2) | 30 seconds | ~150-450 requests | Moderate variance | 30s |
| High (3) | 100 seconds | ~500-1500 requests | Low variance | 100s |
| Full (4) | 300 seconds | ~1500-4500 requests | Production-like | 300s |

At low fidelity, we can evaluate ~30 configurations in the time it takes to evaluate ~3 at full fidelity.

### Concrete Implementation Plan for vllm-tuner

**Using Optuna with Hyperband Pruner:**

```python
import optuna

def objective(trial):
    # Suggest parameters
    config = {
        'max_num_seqs': trial.suggest_int('max_num_seqs', 64, 8192, log=True),
        'max_num_batched_tokens': trial.suggest_int('max_num_batched_tokens', 64, 8192, log=True),
        'block_size': trial.suggest_categorical('block_size', [8, 16, 32]),
        'scheduler_delay_factor': trial.suggest_float('scheduler_delay_factor', 0.0, 2.0),
        'enable_chunked_prefill': trial.suggest_categorical('enable_chunked_prefill', [True, False]),
        'num_scheduler_steps': trial.suggest_int('num_scheduler_steps', 1, 20),
        'gpu_memory_utilization': trial.suggest_float('gpu_memory_utilization', 0.7, 0.95),
    }

    # Multi-fidelity: run progressively longer benchmarks
    for step, duration in enumerate([10, 30, 100]):
        metrics = run_benchmark(config, duration_seconds=duration)

        if metrics is None:  # Crash
            return float('inf')  # Report worst value

        throughput = metrics['throughput']
        trial.report(throughput, step)

        # Hyperband pruning: kill bad configs early
        if trial.should_prune():
            raise optuna.TrialPruned()

    return throughput

# Hyperband-style pruner
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=1,     # step 0 = 10s benchmark
        max_resource=3,     # step 2 = 100s benchmark
        reduction_factor=3, # keep top 1/3 at each step
    ),
)
study.optimize(objective, n_trials=60)
```

**Custom BOHB-style with HEBO:**

```python
class MultiFidelityTuner:
    def __init__(self, fidelity_levels=[10, 30, 100]):
        self.fidelity_levels = fidelity_levels
        self.optimizer = HEBO(space)  # HEBO for intelligent sampling

    def run_successive_halving(self, n_configs=27, eta=3):
        """Successive halving with HEBO-guided sampling."""
        # Round 1: evaluate n_configs at lowest fidelity
        candidates = [self.optimizer.suggest(1) for _ in range(n_configs)]

        for fidelity_idx, duration in enumerate(self.fidelity_levels):
            results = []
            for config in candidates:
                metrics = run_benchmark(config, duration_seconds=duration)
                results.append((config, metrics))
                self.optimizer.observe(config, metrics)

            # Keep top 1/eta
            results.sort(key=lambda x: x[1].get('throughput', 0), reverse=True)
            n_survivors = max(1, len(results) // eta)
            candidates = [config for config, _ in results[:n_survivors]]

        return candidates[0]  # Best config after full evaluation
```

### Expected Improvement

- **Time savings**: With successive halving (eta=3) and 3 fidelity levels:
  - 27 configs at 10s + 9 configs at 30s + 3 configs at 100s = 270 + 270 + 300 = 840 seconds total
  - vs. 27 configs at 100s = 2700 seconds (naive approach)
  - **3.2x speedup** while evaluating the same number of initial configurations.
- **Better exploration**: Evaluating 27 configs at low fidelity explores 9x more of the search space than evaluating 3 configs at high fidelity.

### Complexity Assessment

Low-Medium. Optuna's HyperbandPruner provides most of the machinery. The main challenges are:
1. Implementing configurable benchmark duration (must modify the benchmarking harness).
2. Ensuring low-fidelity results correlate well with high-fidelity results (need empirical validation).
3. Handling the case where a config passes low-fidelity but crashes at high-fidelity (the system reaches a different state with longer runs).

---

## 5. Cost-Aware Bayesian Optimization

### What It Is

Standard BO assumes each evaluation has the same cost. In reality, vLLM evaluations have variable costs:
- A configuration that crashes after 10 seconds is "free" compared to one that runs for the full 100 seconds.
- Configurations with more GPUs (`tensor_parallel=4`) cost more compute resources than `tensor_parallel=1`.
- Some configurations cause very slow startup times (model loading, CUDA compilation) that add to evaluation cost.

Cost-aware BO modifies the acquisition function to prefer configurations that offer the best improvement per unit cost.

**Source**: "Cost-aware Bayesian Optimization" (Amazon Science, 2020), EvolCAF (2024)

### How It Works

**Expected Improvement per Unit Cost (EIpu):**

The standard EI acquisition function is divided by the predicted evaluation cost:

```
EIpu(x) = EI(x) / cost(x)
```

This biases the optimizer toward cheaper-to-evaluate configurations, which are more cost-effective for exploration. A separate GP can model the cost function.

**EI-cool (Cost-Optimized Online Learning):**

An alternative that balances the EI/cost tradeoff dynamically, becoming less cost-sensitive as the optimization budget is consumed (early trials prioritize cheap exploration; later trials prioritize quality).

**EvolCAF (2024):**

Uses LLMs + evolutionary computation to automatically design cost-aware acquisition functions. Rather than manually designing EIpu or EI-cool, an LLM proposes acquisition function formulas, and evolutionary search optimizes them. This eliminates the need for domain expertise in acquisition function design.

**Cost-Aware Stopping:**

Recent work (2025) on cost-aware stopping for BO proposes methods to automatically decide when to stop the optimization process based on the expected marginal improvement relative to remaining budget.

### Concrete Implementation Plan for vllm-tuner

```python
class CostAwareOptimizer:
    """Wraps the base optimizer with cost-aware acquisition."""

    def __init__(self, base_optimizer, cost_model=None):
        self.base_optimizer = base_optimizer
        self.cost_model = cost_model or self._default_cost_model
        self.cost_observations = []

    def _default_cost_model(self, config):
        """Estimate evaluation cost based on config parameters."""
        base_cost = 1.0  # normalized

        # More GPUs = more expensive (resource cost)
        tp = config.get('tensor_parallel', 1)
        base_cost *= tp

        # Higher max_num_seqs = longer warmup and evaluation
        seqs = config.get('max_num_seqs', 256)
        base_cost *= (1.0 + seqs / 8192)

        # Chunked prefill configurations may take longer to stabilize
        if config.get('enable_chunked_prefill', False):
            base_cost *= 1.1

        return base_cost

    def suggest_with_cost(self, n_suggestions=1):
        """Generate suggestions weighted by expected cost."""
        # Generate extra candidates
        candidates = []
        for _ in range(n_suggestions * 5):
            candidate = self.base_optimizer.suggest(1)
            cost = self.cost_model(candidate)
            # Get predicted improvement from base optimizer
            ei = self.base_optimizer.predict_improvement(candidate)
            ei_per_cost = ei / max(cost, 0.01)
            candidates.append((candidate, ei_per_cost))

        # Return top-n by EI/cost ratio
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in candidates[:n_suggestions]]
```

**BoTorch native implementation:**

```python
# BoTorch has built-in cost-aware BO support
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.models.cost import AffineFidelityCostModel

cost_model = AffineFidelityCostModel(fidelity_weights={...})
cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
```

### Expected Improvement

- **Resource efficiency**: When evaluation costs vary by 2-4x across configurations, cost-aware BO can find equivalent solutions 30-50% faster in wall-clock time.
- **GPU resource savings**: By preferring cheaper-to-evaluate configurations for exploration, total GPU-hours consumed during tuning can be reduced by 20-40%.
- **Automatic budget allocation**: The optimizer naturally allocates more evaluations to cheaper configs and reserves expensive evaluations for promising candidates.

### Complexity Assessment

Medium. The main challenges are:
1. Building an accurate cost model (requires timing data from past evaluations).
2. Integrating cost into the acquisition function without distorting the optimization.
3. BoTorch provides the cleanest implementation but requires switching from Optuna.

---

## 6. Simulation-in-the-Loop Optimization

### What It Is

Instead of running expensive real stress tests for every candidate configuration, use a simulator to cheaply evaluate most configurations and only run real benchmarks for the most promising ones. This combines the throughput of simulation with the accuracy of real evaluation.

### LLMServingSim

**Paper**: "LLMServingSim: A HW/SW Co-Simulation Infrastructure for LLM Inference Serving at Scale"
**Source**: https://arxiv.org/abs/2408.05499, https://github.com/casys-kaist/LLMServingSim

LLMServingSim is a system-level simulator for LLM inference that jointly simulates serving system software and hardware:
- Models dynamic workload characteristics of autoregressive inference.
- Detailed memory modeling for KV cache behavior.
- Iteration-level simulation granularity with computation reuse across decoder blocks.
- **Error rate**: < 14.7% compared to real GPU-based systems.
- **Speed**: 91.5x faster than accelerator simulators.

### vLLMSim (MIT Thesis 2025)

Provides highly accurate runtime predictions and precomputed performance profiles that can simulate workloads without GPU access.

### Concrete Implementation Plan for vllm-tuner

**Three-Tier Evaluation Architecture:**

```
Tier 1: Simulation (LLMServingSim)     -- 1-5 seconds per config
    |  Screen 100+ candidates
    v
Tier 2: Short Real Benchmark (10-30s)  -- 10-30 seconds per config
    |  Validate top 10-20 candidates
    v
Tier 3: Full Real Benchmark (100-300s) -- 100-300 seconds per config
    |  Finalize top 3-5 candidates
    v
    Final Configuration
```

```python
class SimulationInLoopOptimizer:
    """Use simulator as cheap surrogate, real benchmarks for validation."""

    def __init__(self, simulator, real_evaluator, optimizer):
        self.simulator = simulator  # LLMServingSim or analytical model
        self.real_evaluator = real_evaluator
        self.optimizer = optimizer
        self.sim_correction_model = None  # Learns sim-to-real mapping

    def optimize(self, total_budget=30):
        """Budget = number of real evaluations allowed."""
        # Phase 1: Explore with simulator (free)
        sim_results = []
        for _ in range(200):
            config = self.optimizer.suggest(1)
            sim_metrics = self.simulator.evaluate(config)
            sim_results.append((config, sim_metrics))
            self.optimizer.observe(config, sim_metrics)

        # Phase 2: Validate top candidates with real benchmark
        sim_results.sort(key=lambda x: x[1]['throughput'], reverse=True)
        top_candidates = sim_results[:total_budget]

        real_results = []
        for config, sim_metrics in top_candidates:
            real_metrics = self.real_evaluator.evaluate(config, duration=100)
            real_results.append((config, real_metrics, sim_metrics))

            # Learn sim-to-real correction
            self._update_correction_model(sim_metrics, real_metrics)

        # Phase 3: Use corrected simulator for final refinement
        for _ in range(50):
            config = self.optimizer.suggest(1)
            sim_metrics = self.simulator.evaluate(config)
            corrected = self._apply_correction(sim_metrics)
            self.optimizer.observe(config, corrected)

        return self.optimizer.get_best()

    def _update_correction_model(self, sim_metrics, real_metrics):
        """Learn a linear correction from sim to real metrics."""
        # Simple: learn offset and scale per metric
        # sim_throughput * scale + offset = real_throughput
        pass

    def _apply_correction(self, sim_metrics):
        """Apply learned correction to simulator predictions."""
        if self.sim_correction_model is None:
            return sim_metrics
        return self.sim_correction_model.predict(sim_metrics)
```

**Alternative: Analytical Surrogate Model**

For cases where LLMServingSim is not available, build a simple analytical model:

```python
class AnalyticalSurrogate:
    """Roofline-based analytical model for quick screening."""

    def estimate_throughput(self, config, model_info, hw_info):
        """Estimate throughput from first principles."""
        # Memory per request = KV cache size
        kv_per_token = 2 * model_info['num_layers'] * model_info['hidden_dim'] * \
                       model_info['bytes_per_param']
        kv_per_request = kv_per_token * model_info['avg_seq_len']

        # Available KV cache memory
        total_gpu_mem = hw_info['gpu_memory_bytes'] * config['gpu_memory_utilization']
        model_mem = model_info['model_size_bytes']
        kv_budget = total_gpu_mem - model_mem

        # Max concurrent requests
        max_batch = min(
            config['max_num_seqs'],
            int(kv_budget / kv_per_request)
        )

        # Compute throughput (simplified roofline)
        tokens_per_second = min(
            hw_info['compute_flops'] / model_info['flops_per_token'],
            hw_info['memory_bandwidth'] / model_info['bytes_per_token']
        )

        return tokens_per_second * max_batch / model_info['avg_output_len']
```

### Expected Improvement

- **Evaluation volume**: With simulation, evaluate 100-200 configs vs. 30 configs with real benchmarks only. Explore 3-7x more of the search space.
- **Time savings**: If simulation takes 5 seconds vs. 100 seconds for real evaluation, and we use simulation for 80% of evaluations, total time drops from 50 minutes (30 real evals) to ~25 minutes (6 real + 200 simulated).
- **Better optima**: More exploration = higher probability of finding the global optimum. Simulation-based screening filters out obviously bad regions before expensive evaluation.

### Complexity Assessment

- **Analytical surrogate**: Medium. Requires modeling vLLM's memory management and scheduling.
- **LLMServingSim integration**: High. Requires setting up the simulator, calibrating it for the target hardware, and building the sim-to-real correction model.
- **vLLMSim integration**: Medium. If precomputed profiles are available.

---

## 7. vLLM Parameter Interactions

### Comprehensive Parameter Dependency Map

Based on vLLM documentation, GitHub issues, and community forums, here is the full interaction map:

### 7.1 Critical Known Constraints

| Constraint | Condition | Effect if Violated |
|-----------|-----------|-------------------|
| `max_num_batched_tokens >= max_num_seqs` | Always | Server crash at startup |
| `max_num_batched_tokens >= max_model_len` | When `enable_chunked_prefill=False` | Server crash at startup |
| `enable_chunked_prefill + enable_prefix_caching` | vLLM < 0.6 | Cannot both be True |
| `enable_chunked_prefill + MLA models` | Always for MLA | Assertion error if both enabled |
| `tensor_parallel <= num_available_gpus` | Always | Startup failure |
| `max_num_seqs * max_model_len * kv_per_token` | Must fit in KV cache budget | OOM at runtime |

### 7.2 Parameter Interaction Groups

**Group 1: Batch Size Controls (Strong Interactions)**

```
max_num_seqs <---> max_num_batched_tokens <---> gpu_memory_utilization
     |                    |                          |
     |                    |                          |
     v                    v                          v
Concurrent      Tokens per batch         KV cache budget
requests        (prefill chunks)         (memory allocation)
```

- `max_num_seqs` and `max_num_batched_tokens` jointly determine actual batch size.
- Both are constrained by `gpu_memory_utilization` which controls KV cache budget.
- In vLLM V1: KV cache requirement = `max_num_seqs * max_model_len`, creating a hard memory constraint.
- Increasing `max_num_seqs` beyond what memory supports causes preemption (evicting KV cache).

**Group 2: Scheduling Controls (Moderate Interactions)**

```
scheduler_delay_factor <---> num_scheduler_steps <---> enable_chunked_prefill
         |                          |                          |
         v                          v                          v
   Request batching          CPU-GPU sync           Prefill/decode
   wait time                 frequency              interleaving
```

- `scheduler_delay_factor` controls how long the scheduler waits before batching. Higher values = larger batches (better throughput) but higher latency.
- `num_scheduler_steps` controls multi-step scheduling. Higher values = less CPU overhead but higher TTFT for new requests (they wait for current multi-step to finish).
- `enable_chunked_prefill` changes the scheduling policy to prioritize decode requests and chunk prefills.

**Group 3: Caching Controls (Strong Interactions)**

```
enable_prefix_caching <---> enable_chunked_prefill <---> block_size
         |                          |                         |
         v                          v                         v
   KV cache reuse           Prefill chunking          Memory granularity
   for shared prefixes      strategy                  fragmentation vs overhead
```

- When both `enable_prefix_caching` and `enable_chunked_prefill` are enabled, only the first chunk leverages prefix caching, resulting in suboptimal cache utilization.
- `block_size` affects both prefix caching (larger blocks = coarser matching but less overhead) and memory fragmentation (smaller blocks = finer granularity but more metadata).

**Group 4: Memory Controls (Strong Interactions)**

```
gpu_memory_utilization <---> tensor_parallel <---> max_model_len
         |                          |                    |
         v                          v                    v
   Available memory          Model sharding         Max context length
   for KV cache             across GPUs            (memory per request)
```

- `tensor_parallel` splits the model across GPUs, making each GPU responsible for less model memory but adding communication overhead.
- Higher `gpu_memory_utilization` allocates more memory for KV cache but leaves less for overhead and CUDA kernels.
- `max_model_len` directly controls memory per request. Reducing it from 8192 to 2048 can 4x the number of concurrent requests.

### 7.3 Performance Impact Directions

| Parameter | Increase Effect on Throughput | Increase Effect on TTFT | Increase Effect on TPOT |
|-----------|------------------------------|------------------------|------------------------|
| `max_num_seqs` | Increases (more batching) | Increases (more queuing) | Increases (more interference) |
| `max_num_batched_tokens` | Increases (more tokens/batch) | Decreases (more prefill/batch) | May increase (prefill interference) |
| `scheduler_delay_factor` | Increases (better batching) | Increases (wait for batch) | Neutral |
| `num_scheduler_steps` | Increases (less CPU overhead) | Increases (wait for multi-step) | Bursty (tokens arrive in groups) |
| `enable_chunked_prefill` | Increases (decode-maximal) | May increase (chunking overhead) | Decreases (less decode stalling) |
| `enable_prefix_caching` | Depends on workload | Decreases (cache hits) | Neutral |
| `gpu_memory_utilization` | Increases (more KV cache) | Neutral | Neutral |
| `block_size` (larger) | Slight decrease (fragmentation) | Neutral | Neutral |

### 7.4 Workload-Dependent Recommendations

| Workload Type | Key Parameters to Tune | Rationale |
|--------------|----------------------|-----------|
| Chatbot (interactive) | `num_scheduler_steps` (low), `enable_chunked_prefill=True`, `max_num_batched_tokens` (moderate) | Minimize TTFT and TPOT; chunked prefill prevents decode stalling |
| Batch processing | `max_num_seqs` (high), `scheduler_delay_factor` (high), `num_scheduler_steps` (high) | Maximize throughput; latency less important |
| RAG with shared context | `enable_prefix_caching=True`, `block_size=16` | Reuse KV cache for shared document context |
| Long-context | `gpu_memory_utilization` (high), `max_num_seqs` (low), `enable_chunked_prefill=True` | Memory-constrained; chunked prefill manages large prefills |
| Classification (short) | `max_num_seqs` (high), `scheduler_delay_factor` (low) | Short sequences; maximize concurrency |

### Concrete Implementation for vllm-tuner

```python
# Known constraints to encode in the search space
VLLM_KNOWN_CONSTRAINTS = [
    # max_num_batched_tokens >= max_num_seqs (always)
    lambda p: p['max_num_batched_tokens'] >= p['max_num_seqs'],

    # When chunked_prefill is disabled, max_num_batched_tokens >= max_model_len
    lambda p: p['enable_chunked_prefill'] or p['max_num_batched_tokens'] >= p.get('max_model_len', 4096),

    # Prefix caching + chunked prefill conflict (vLLM < 0.6 only)
    # For newer vLLM, this constraint is relaxed but suboptimal
    # lambda p: not (p['enable_chunked_prefill'] and p['enable_prefix_caching']),

    # Memory constraint: rough estimate
    # max_num_seqs * avg_kv_per_seq <= available_kv_memory
    lambda p: p['max_num_seqs'] * p.get('est_kv_per_seq', 1e8) <= \
              p.get('available_kv_memory', 60e9),
]

def validate_config(params, vllm_version="0.6.0"):
    """Check all known constraints before evaluation."""
    for constraint in VLLM_KNOWN_CONSTRAINTS:
        if not constraint(params):
            return False, "Known constraint violated"
    return True, "OK"
```

---

## 8. Framework Comparison: Optuna vs SMAC vs HEBO vs BoTorch

### Detailed Feature Comparison

| Feature | Optuna | SMAC3 | HEBO | BoTorch/Ax |
|---------|--------|-------|------|-----------|
| **Surrogate Model** | TPE | Random Forest (C++) or GP | GP (Matern 5/2) + warping | GP (flexible kernels) |
| **Acquisition Function** | EI (implicit in TPE) | EI + log transform | Multi-objective ensemble | qEI, qNEHVI, qParEGO, custom |
| **Multi-Objective** | NSGA-II, MOTPE | Yes (native) | Yes (GeneralBO) | qNEHVI (state-of-the-art) |
| **Multi-Fidelity** | HyperbandPruner | BOHB, Hyperband (native) | No built-in | MOMF, multi-fidelity GP |
| **Constraint Handling** | Pruning, constraint sampling | Constrained EI | Penalty-based | Constrained EHVI (native) |
| **Parallelism** | Distributed (native) | Multi-threading (native) | Constant liar | qEI (native batch) |
| **Speed per Trial** | Fast (O(n log n)) | Fast (RF) or Moderate (GP) | Moderate (GP O(n^3)) | Moderate to Slow |
| **Sample Efficiency (low-dim, < 30 trials)** | Good | Very Good | Very Good | Best |
| **Sample Efficiency (high-dim, > 20 params)** | Very Good | Very Good | Moderate | Moderate |
| **Ecosystem** | Largest (MLflow, W&B, etc.) | Good (AutoML ecosystem) | Small | Good (Meta ecosystem) |
| **Ease of Use** | Best (define-by-run) | Good | Good | Moderate (PyTorch-based) |
| **Active Development** | Very Active | Active | Moderate | Very Active (Meta) |
| **License** | MIT | BSD-3 | MIT | MIT |

### Benchmark Results (Ax/AutoML 2025 Paper)

From the Ax paper benchmarks (AutoML Conference 2025):

- **Overall**: Ax achieves state-of-the-art performance across synthetic and real-world black-box optimization tasks.
- **Multi-objective, constrained, high-dimensional**: Ax substantially better early on.
- **Mixed/discrete settings**: Ax comparable to others until ~20 trials, then pulls ahead.
- **Speed**: Optuna and SMAC-HPO (non-GP methods) are substantially faster than Ax, HEBO, and SMAC-BB (GP methods).

From the systematic HPO study (Kégl, Medium):

- **HEBO** is one of three engines that perform significantly better than the rest.
- **Performance with moderate trials (10-50)**: HEBO > SMAC > Optuna.
- **Performance with many trials (50+)**: Differences narrow; Optuna competitive.

### Recommendation for vllm-tuner

**Best overall choice: Optuna + HEBOSampler**

Rationale:
1. **Sample efficiency matters most**: vllm-tuner evaluations are expensive (5-10 minutes each), so we can only afford 20-50 trials. In this regime, GP-based methods (HEBO, BoTorch) significantly outperform TPE.
2. **Ecosystem preservation**: Optuna's ecosystem (logging, visualization, distributed execution, pruning) is the best. By using HEBO as an Optuna sampler, we get HEBO's sample efficiency with Optuna's infrastructure.
3. **Multi-objective support**: HEBO's GeneralBO via OptunaHub supports multi-objective optimization for TTFT+TPOT tradeoffs.
4. **Constraint handling gap**: Neither HEBO nor Optuna has great native constraint handling. We must add SCOOT-style random forest constraint learning regardless of which framework we choose.

**If starting from scratch: BoTorch/Ax**

If vllm-tuner were being built from scratch, BoTorch/Ax would be the strongest choice because:
- Best-in-class constrained multi-objective BO (qNEHVI with constraints).
- Native cost-aware BO (InverseCostWeightedUtility).
- Multi-fidelity BO (MOMF).
- Used at scale at Meta for system optimization.

However, migrating from Optuna to BoTorch/Ax is a significant rewrite, and the marginal benefit over Optuna+HEBO doesn't justify the cost.

**SMAC3 as alternative for constraint-heavy problems:**

SMAC3's random forest surrogate naturally handles mixed categorical/continuous spaces (like vLLM's parameter mix) and is available as an OptunaHub sampler. Its random forest can capture non-smooth parameter interactions better than GP. Worth testing as an alternative sampler for vllm-tuner via OptunaHub.

### Implementation Plan

```python
# Phase 1: Drop-in HEBO replacement (immediate)
import optunahub
hebo_module = optunahub.load_module("samplers/hebo")
sampler = hebo_module.HEBOSampler(seed=42)

# Phase 2: Add SMAC as comparison sampler (short-term)
smac_module = optunahub.load_module("samplers/smac_sampler")
smac_sampler = smac_module.SMACSampler(seed=42)

# Phase 3: Constraint-aware wrapper (medium-term)
# Wraps any sampler with known + hidden constraint checking
constraint_sampler = ConstraintAwareSampler(
    base_sampler=hebo_module.HEBOSampler(seed=42),
    constraint_learner=HiddenConstraintLearner(),
    known_constraints=VLLM_KNOWN_CONSTRAINTS
)

# Phase 4: LLAMBO warmstart + HEBO main (medium-term)
hybrid_sampler = HybridSampler(
    warmstart_sampler=llambo_module.LLAMBOSampler(),
    main_sampler=constraint_sampler,
    switch_trial=5
)
```

---

## 9. Consolidated Recommendations

### Priority Ranking by Impact/Effort Ratio

| Priority | Technique | Expected Impact | Implementation Effort | Effort (Days) |
|----------|-----------|----------------|----------------------|---------------|
| **P0** | HEBO as Optuna sampler | 20-40% fewer trials to converge | Drop-in replacement | 0.5 |
| **P0** | Known constraint encoding | Eliminate 30%+ infeasible suggestions | Configuration + validation | 1 |
| **P1** | Hidden constraint learning (RF) | Save 5-10 crashed evaluations per session | Sklearn RF + integration | 2-3 |
| **P1** | Multi-fidelity (Hyperband pruner) | 3x speedup in total tuning time | Configurable benchmark duration | 2-3 |
| **P2** | Warm-starting from past results | 2-4x faster convergence for similar tasks | JSON knowledge base | 1-2 |
| **P2** | Parameter interaction map | Better search space definition | Configuration encoding | 1 |
| **P2** | Parallel suggestion (k configs) | Linear speedup with available GPUs | Multi-process evaluation | 2 |
| **P3** | LLAMBO zero-shot warmstarting | Better initial suggestions | OptunaHub integration | 1 |
| **P3** | Cost-aware acquisition | 30-50% better GPU-hour efficiency | Cost model + modified acquisition | 3-4 |
| **P3** | SMAC sampler comparison | May find better optima for mixed spaces | OptunaHub integration | 0.5 |
| **P4** | Simulation-in-the-loop | 5-10x more configs explored | LLMServingSim integration | 5-10 |
| **P4** | GP ensemble transfer learning | Cross-task knowledge transfer | Research-level implementation | 5-7 |

### Recommended Implementation Roadmap

**Week 1: Quick Wins (P0)**
1. Swap Optuna's default TPE sampler for HEBOSampler.
2. Encode all known vLLM constraints as search space pruning rules.
3. Add parameter interaction documentation to guide search space definitions.

**Week 2-3: Core Improvements (P1)**
4. Implement random forest hidden constraint learner.
5. Add multi-fidelity support with configurable benchmark durations.
6. Implement Hyperband-style pruning for early termination of bad configs.

**Week 4-5: Advanced Features (P2)**
7. Build tuning knowledge base for warm-starting.
8. Add parallel suggestion support for multi-GPU evaluation.
9. Implement workload-aware default configurations.

**Week 6+: Research Features (P3-P4)**
10. LLAMBO integration for zero-shot warmstarting.
11. Cost-aware acquisition function.
12. Simulation-in-the-loop with analytical or LLMServingSim surrogate.

### Key Metrics to Track

When implementing these improvements, measure:
1. **Trials to target**: How many evaluations to reach within 5% of the best-known configuration.
2. **Crash rate**: Percentage of evaluations that fail (hidden constraints).
3. **Wall-clock time**: Total time from start to finding the optimal configuration.
4. **GPU-hours**: Total compute resource consumption during tuning.
5. **Final quality**: Best throughput/latency achieved vs. baseline methods.

---

## References

### Bayesian Optimization Frameworks
1. HEBO: https://github.com/huawei-noah/HEBO, https://arxiv.org/abs/2012.03826
2. SMAC3: https://github.com/automl/SMAC3, https://www.jmlr.org/papers/v23/21-0888.html
3. BoTorch: https://botorch.org/docs/multi_objective/
4. Ax: https://ax.dev/, AutoML 2025 paper: https://openreview.net/forum?id=U1f6wHtG1g
5. OptunaHub (HEBO sampler): https://hub.optuna.org/samplers/hebo/
6. OptunaHub (SMAC sampler): https://hub.optuna.org/samplers/smac_sampler/

### Constraint Handling
7. "Local Bayesian Optimization for Controller Tuning with Crash Constraints": https://arxiv.org/abs/2411.16267
8. "Boundary Exploration for Bayesian Optimization With Unknown Physical Constraints": https://arxiv.org/abs/2402.07692
9. "Surrogate-Based Optimization of System Architectures Subject to Hidden Constraints": https://arxiv.org/abs/2504.08721
10. "Constrained Bayesian Optimization under Partial Observations" (AAAI 2024): https://arxiv.org/abs/2312.03212

### Transfer Learning and Meta-Learning
11. "Transfer Learning for Bayesian Optimization: A Survey": https://arxiv.org/abs/2302.05927
12. "Meta-Learning Acquisition Functions for Transfer Learning in BO" (ICLR 2020): https://arxiv.org/abs/1904.02642
13. LLAMBO (ICLR 2024): https://arxiv.org/abs/2402.03921
14. Restune (SIGMOD 2021): by Xinyi Zhang et al.

### Multi-Fidelity Optimization
15. BOHB: https://www.automl.org/blog_bohb/
16. "Practical Multi-fidelity Bayesian Optimization" (UAI 2019): https://arxiv.org/abs/1903.04703
17. MFES-HB (AAAI 2021): https://ojs.aaai.org/index.php/AAAI/article/view/17031

### Cost-Aware Optimization
18. "Cost-aware Bayesian Optimization" (Amazon Science): https://arxiv.org/abs/2003.10870
19. EvolCAF (2024): https://arxiv.org/abs/2404.16906
20. "Cost-aware Stopping for Bayesian Optimization" (2025): https://arxiv.org/abs/2507.12453

### Simulation and Performance Modeling
21. LLMServingSim: https://arxiv.org/abs/2408.05499, https://github.com/casys-kaist/LLMServingSim
22. "Simulation Based Bayesian Optimization": https://arxiv.org/abs/2401.10811

### vLLM Documentation
23. vLLM Optimization and Tuning: https://docs.vllm.ai/en/stable/configuration/optimization/
24. vLLM Engine Arguments: https://docs.vllm.ai/en/stable/configuration/engine_args/
25. vLLM v0.6.0 Performance Update: https://blog.vllm.ai/2024/09/05/perf-update.html
26. vLLM Parameter Discussion: https://github.com/vllm-project/vllm/issues/2492
27. vLLM Multi-Step Scheduling: https://github.com/vllm-project/vllm/issues/6854
28. vLLM Prefix Caching + Chunked Prefill: https://discuss.vllm.ai/t/should-vllm-consider-prefix-caching-when-chunked-prefill-is-enabled/903
