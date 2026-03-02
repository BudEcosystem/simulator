# 14. Hardware, Training, Validation, and CPU Subsystem Analysis

This document provides a comprehensive analysis of seven subsystems within the `llm-memory-calculator` package, covering their current capabilities, API surfaces, extension points, and potential enhancements from research.

---

## 1. Hardware Layer

### Files Analyzed
- `hardware/configs.py`
- `hardware/gpu_specs.py`
- `hardware/cpu_specs.py`
- `hardware/cost_utils.py`
- `hardware/device_matcher.py`
- `hardware/manager.py`
- `hardware/db_connection.py`
- `hardware/__init__.py`

### A. Current Capabilities

**Static Hardware Configuration Database (`configs.py`)**
- Central `HARDWARE_CONFIGS` dictionary with 50+ hardware entries spanning NVIDIA GPUs (V100 through GB200), Google TPUs (v4-v6), AMD accelerators (MI300X, MI325X), Intel devices (MAX 1550, ARC, Gaudi3), cloud ASICs (AWS Trainium/Inferentia, Cerebras, Groq, SambaNova), and CPU configs.
- Each entry stores: `Flops` (peak TFLOPS), `Memory_size` (GB), `Memory_BW` (GB/s), `ICN` (interconnect bandwidth GB/s), `cost` (per-provider pricing), `pci_ids`, `aliases`, `manufacturer`, `type`.
- Merges CPU configurations from `cpu_specs.py` via `**CPU_CONFIGS` to provide a unified lookup.
- Helper functions filter by manufacturer (`get_hardware_by_manufacturer`) and type (`get_hardware_by_type`).

**GPU Architecture Specifications (`gpu_specs.py`)**
- `GPUArchitecture` dataclass with vendor, process node, description.
- Architecture dictionaries for NVIDIA (Pascal through Blackwell, 7 generations), AMD (GCN through CDNA3), and Intel.
- `PCI_ARCHITECTURE_MAP`: ~70 PCI device IDs mapped to architecture, model name, and variant.
- `ARCHITECTURE_FEATURES`: per-architecture feature sets including tensor core types, memory technologies, and key capabilities (e.g., FP8 support, HBM3e).
- Functions: `get_architecture_info()`, `get_gpu_info_by_pci_id()`, `get_compute_capability()`.

**CPU Specifications (`cpu_specs.py`)**
- Detailed SKU-level specifications for Intel Sapphire Rapids (5 SKUs), Emerald Rapids (2 SKUs), AMD Milan (3 SKUs), Genoa (4 SKUs), and Bergamo (2 SKUs).
- Each SKU: cores, base/turbo frequency, calculated FLOPS, TDP, memory bandwidth, sockets, memory channels.
- `GENERATION_TO_SKUS` mapping for backward-compatible lookups (e.g., "sapphire_rapids" -> list of SKUs).

**Cost Utilities (`cost_utils.py`)**
- `get_best_rate(cost_data, allow_spot, preferred_provider)`: finds cheapest rate across 7 cloud providers (AWS, GCP, Azure, Lambda Labs, CoreWeave, RunPod, Vast.ai), with spot/on-demand tiers.
- `has_cost_data(config)`: validates that non-zero pricing exists.
- `get_all_provider_rates(cost_data)`: organizes all available rates by provider.

**Device Matching (`device_matcher.py`)**
- `DeviceIdentity` dataclass with gpu_architecture, compute_capability, pci_device_id.
- `DeviceParser`: regex-based parsing of device names, PCI IDs (hex format), memory sizes from strings.
- `DeviceMatcher`: multi-strategy matching pipeline:
  1. CPU ID match (exact)
  2. PCI device ID match
  3. Architecture + memory match
  4. Name-based fuzzy match
- `match_device(device_info)`: unified entry point returning best hardware config key + confidence score.

**Hardware Manager (`manager.py`)**
- `HardwareManager` class with LRU caching, alias map, device matching cache.
- Methods: `get_hardware_config()`, `get_all_hardware()`, `search_hardware()`, `match_cluster_device()`, `get_cluster_hardware_specs()`.
- Module-level convenience functions wrapping singleton pattern for simple API access.

**Database Connection (`db_connection.py`)**
- `ReadOnlyDatabaseConnection` using SQLite URI mode (`?mode=ro`) for safety.
- GenZ-compatible field mapping (lowercase DB columns -> PascalCase config keys: `flops -> Flops`, `memory_size -> Memory_size`).

### B. API Surface

```python
# configs.py
HARDWARE_CONFIGS: Dict[str, Dict[str, Any]]  # Central config dict
get_hardware_names() -> List[str]
get_hardware_by_manufacturer(manufacturer: str) -> Dict[str, Dict]
get_hardware_by_type(hw_type: str) -> Dict[str, Dict]

# gpu_specs.py
get_architecture_info(arch_name: str) -> Optional[GPUArchitecture]
get_gpu_info_by_pci_id(pci_id: str) -> Optional[Dict]
get_compute_capability(gpu_name: str) -> Optional[str]

# cost_utils.py
get_best_rate(cost_data, allow_spot, preferred_provider) -> Tuple[float, str, str]
has_cost_data(config: Dict) -> bool
get_all_provider_rates(cost_data: Dict) -> Dict[str, List]

# device_matcher.py
match_device(device_info: Dict) -> Tuple[str, float]  # (config_key, confidence)

# manager.py
HardwareManager.get_hardware_config(name: str) -> Dict
HardwareManager.get_all_hardware() -> Dict
HardwareManager.search_hardware(query: str) -> List[Dict]
HardwareManager.match_cluster_device(device_info: Dict) -> str
HardwareManager.get_cluster_hardware_specs(devices: List[Dict]) -> List[Dict]
```

### C. Extension Points

1. **Adding new hardware**: Add entry to `HARDWARE_CONFIGS` dict with required keys (Flops, Memory_size, Memory_BW, ICN, cost, aliases).
2. **New cloud providers**: Extend cost dictionaries within each hardware entry and update `cost_utils.py` parsing.
3. **New GPU architectures**: Add to `NVIDIA_ARCHITECTURES`/`AMD_ARCHITECTURES` dicts and add PCI IDs to `PCI_ARCHITECTURE_MAP`.
4. **New CPU SKUs**: Add to `CPU_SPECS` dict in `cpu_specs.py` with full parameter set.
5. **Custom matching strategies**: Extend `DeviceMatcher._match_by_name()` or add new matching stages.
6. **Database overlay**: `ReadOnlyDatabaseConnection` can overlay static configs with database entries.

### D. Research Enhancements

- **Power modeling (LLMServingSim)**: The hardware layer has no power consumption modeling. LLMServingSim's dynamic power model could be integrated by adding per-hardware TDP curves and utilization-dependent power draw.
- **CXL/PIM memory tiers**: Current configs model only HBM/GDDR. CXL-attached memory and Processing-in-Memory devices from emerging hardware could be added as additional memory tiers with separate bandwidth/latency parameters.
- **Heterogeneous device pools**: `match_cluster_device` currently matches to a single config. Research on heterogeneous clusters (e.g., mixed A100/H100 nodes) would benefit from multi-device pool modeling.
- **Interconnect topology modeling**: ICN is a single number. Research shows that ring, tree, and mesh topologies have different collective communication characteristics. Adding topology descriptors would improve multi-node accuracy.
- **Thermal throttling**: No thermal model exists. Adding junction temperature curves and throttling thresholds would improve sustained workload predictions.

---

## 2. Training Subsystem

### Files Analyzed
- `training/calculator.py`
- `training/advanced_calculator.py`
- `training/distributed.py`
- `training/cluster_optimizer.py`
- `training/tco_calculator.py`
- `training/time_estimator.py`
- `training/scaling_laws.py`

### A. Current Capabilities

**Memory Calculation (`calculator.py`)**
- `TrainingMemoryCalculator` uses a peak-of-phases model:
  ```
  forward_peak  = weight_memory + activation_memory
  backward_peak = weight_memory + activation_memory + gradient_memory
  optimizer_peak = weight_memory + gradient_memory + optimizer_memory
  peak_memory = max(forward_peak, backward_peak, optimizer_peak)
  ```
- Weight memory: bytes_per_param * num_params, with full/LoRA/QLoRA paths.
- LoRA trainable parameters: GQA-aware calculation targeting 7 modules (q, k, v, o, gate, up, down) with proper KV head scaling.
- QLoRA memory: NF4 quantization at ~0.516 bytes/param (4-bit + double quantization overhead) plus dequantization buffer.
- Activation memory: config-aware Megatron-LM formula incorporating hidden_size, num_heads, seq_length, batch_size, num_layers with FlashAttention support (eliminates O(s^2) attention score storage).
- Gradient checkpointing: stores ceil(sqrt(L)) layers instead of all L layers.
- Optimizer state: full mapping for 25+ optimizers (AdamW=12 bytes, SGD=4 bytes, 8-bit Adam=6 bytes, LOMO=0 bytes, etc.).

**Advanced Calculator (`advanced_calculator.py`)**
- `AdvancedTrainingCalculator` extends base calculator with stage awareness (SFT, DPO, PPO, KTO, RM, PT).
- `AdvancedTrainingEstimate` dataclass includes fit analysis (fits_on_gpu, fit_details), TPS estimation, and time/cost projections.
- DPO/PPO stages add reference model memory (frozen copy); PPO adds reward model; RM adds value head.
- `calculate_training_with_genz()`: integrates GenZ roofline analysis for per-layer timing simulation.
- `list_supported_configurations()`: enumerates all supported hardware, optimizers, training types, and precision options.

**Distributed Training (`distributed.py`)**
- `DeepSpeedConfig`: ZeRO stages 0-3 with CPU/NVMe offload variants.
- `FSDPConfig`: FSDP sharding strategies with `equivalent_zero` mapping to ZeRO stages.
- `ParallelismConfig`: TP, PP, DP, EP, CP with communication overhead estimation per dimension.
- `DistributedMemoryEstimate`: per-GPU memory breakdown after distribution.
- `recommend_distributed_strategy()`: recommends ZeRO stage / FSDP strategy based on memory constraints and GPU count.

**Cluster Optimization (`cluster_optimizer.py`)**
- `ClusterOptimizer` with two core algorithms:
  1. `select_top_k_clusters()`: evaluates hardware x parallelism configurations, scores on normalized metrics (throughput, cost, memory utilization), returns top-K.
  2. `design_optimal_cluster()`: given a hardware type and budget, finds optimal GPU count and parallelism configuration.
- Integrates GenZ `training_modeling()` for accurate per-config simulation when available.
- Pareto frontier optimization via `paretoset` library for multi-objective selection.
- Early constraint pruning (memory check) to avoid expensive simulations on infeasible configs.
- Wrapper functions: `rank_clusters()`, `predict_requirements()`, `generate_comprehensive_config()`.

**TCO Calculator (`tco_calculator.py`)**
- `GPUPricing` dataclass with per-provider rates across 7 cloud providers.
- `GPU_PRICING` database with 11 GPU types and their cloud pricing.
- `calculate_tco()`: computes total cost = GPU compute + power + network + storage + operations overhead.
- `compare_provider_costs()`: side-by-side cost comparison across all providers for given training duration.

**Time Estimator (`time_estimator.py`)**
- `TrainingTimeEstimator` with:
  - `BASELINE_TPS`: empirical tokens/second lookup tables for 10 GPU types x 5 model sizes.
  - `GPU_MFU`: per-GPU Model FLOPs Utilization estimates (H100=0.50, RTX3060=0.12).
  - First-principles fallback: `tps = (peak_tflops * MFU * num_gpus * 1e12) / (6 * params)`.
  - Interconnect-aware scaling: NVLink (0.92^log2(dp)) vs PCIe (0.70^log2(dp)).
  - `OPTIMIZER_OVERHEAD`: compute overhead factors for 15+ optimizers.
  - MFU calculation: `actual_tflops / peak_tflops`.

**Scaling Laws (`scaling_laws.py`)**
- `ScalingCoefficients`: 14 calibrated parameters fitted from Meta/NVIDIA benchmark data.
- Sub-models:
  - `ModelSizeScaling`: superlinear penalty with memory pressure threshold.
  - `ScaleEfficiencyModel`: log-scaling decay per parallelism dimension (TP, PP, DP, EP).
  - `NetworkCongestionModel`: topology-aware bandwidth degradation.
  - `StragglerModel`: sqrt(gpus/1000) overhead at scale.
  - `PipelineBubbleModel`: scale-aware with ZB-V (zero-bubble) support.
  - `DynamicEfficiencyBounds`: hardware-aware efficiency floor/ceiling.
- `compute_composite_efficiency()`: combines all scaling factors into a single efficiency multiplier.

### B. API Surface

```python
# calculator.py
TrainingMemoryCalculator.calculate_training_memory(
    model_config, training_config, hardware_config
) -> TrainingMemoryEstimate

# advanced_calculator.py
AdvancedTrainingCalculator.calculate_advanced_training(
    model_config, training_config, hardware_config
) -> AdvancedTrainingEstimate
AdvancedTrainingCalculator.calculate_training_with_genz(
    model_config, training_config, hardware_config
) -> AdvancedTrainingEstimate
AdvancedTrainingCalculator.list_supported_configurations() -> Dict[str, List]

# distributed.py
DeepSpeedConfig(zero_stage, offload_device, ...)
FSDPConfig(sharding_strategy, ...)
ParallelismConfig(tp, pp, dp, ep, cp)
DistributedMemoryEstimate(per_gpu_memory, ...)
recommend_distributed_strategy(model_params, gpu_memory, num_gpus) -> Dict

# cluster_optimizer.py
ClusterOptimizer.select_top_k_clusters(model_config, training_config, k) -> List[Dict]
ClusterOptimizer.design_optimal_cluster(model_config, hardware, budget) -> Dict
rank_clusters(model_config, training_config) -> List[Dict]
predict_requirements(model_config, training_config) -> Dict
generate_comprehensive_config(model_config, training_config) -> Dict

# tco_calculator.py
calculate_tco(gpu_type, num_gpus, hours, provider) -> Dict
compare_provider_costs(gpu_type, num_gpus, hours) -> Dict

# time_estimator.py
TrainingTimeEstimator.estimate_training_time(
    model_config, dataset_tokens, batch_size, gradient_accumulation,
    epochs, hardware, num_gpus, seq_length, parallelism, optimizer
) -> TrainingTimeEstimate

# scaling_laws.py
compute_composite_efficiency(model_size, num_gpus, parallelism, hardware) -> float
```

### C. Extension Points

1. **New optimizer states**: Add byte counts to `OPTIMIZER_STATE_BYTES` in `calculator.py` and overhead to `OPTIMIZER_OVERHEAD` in `time_estimator.py`.
2. **New training stages**: Add `TrainingStageType` enum value and `TrainingStageConfig` in `training_stages.py`; the advanced calculator picks them up automatically.
3. **New hardware in time estimator**: Add `BASELINE_TPS` entries and `GPU_MFU` values.
4. **Custom scaling models**: Each sub-model in `scaling_laws.py` is independent and can be replaced or extended.
5. **New cloud providers**: Add pricing to `GPU_PRICING` in `tco_calculator.py` and cost data in `configs.py`.
6. **Distributed strategies**: Add new `DeepSpeedConfig` or `FSDPConfig` variants.

### D. Research Enhancements

- **Activation recomputation strategies**: Current gradient checkpointing uses a simple sqrt(L) heuristic. Research on selective recomputation (e.g., recomputing only cheap ops) could reduce memory while minimizing recomputation overhead.
- **Communication-computation overlap**: The scaling laws model communication overhead as a multiplicative penalty. More accurate models would track overlap between gradient AllReduce and backward pass computation.
- **Mixed-precision training dynamics**: The calculator treats precision as a static bytes-per-param multiplier. Research on dynamic loss scaling, precision switching during training, and FP8 training could be integrated.
- **Sequence parallelism and context parallelism**: The parallelism config supports CP but the memory calculator does not model ring attention or sequence-parallel activation sharding.
- **ZeRO++ optimizations**: Hierarchical partitioning and quantized communication from ZeRO++ are not modeled.
- **Expert parallelism for MoE**: EP is in the parallelism config but the memory calculator does not model expert-specific activation and routing memory.

---

## 3. GenZ LLM Training / Simulation Layer

### Files Analyzed
- `genz/LLM_training/training_modeling.py` (first 200 lines)
- `genz/LLM_training/training_parallelization.py` (first 200 lines)
- `genz/LLM_training/training_stages.py` (first 200 lines)

### A. Current Capabilities

**Training Modeling (`training_modeling.py`)**
- End-to-end training simulation using GenZ roofline analysis.
- `soft_bound()`: sigmoid-based clamping that replaces hard min/max to preserve gradient-like smoothness in parameter sweeps.
- `_detect_attention_type_from_config()`: auto-detects MLA (Multi-Latent Attention), GQA (Grouped Query), MQA (Multi-Query), or standard MHA from model config.
- `calculate_effective_overlap()`: models compute-communication overlap bounded by the compute window size.
- `apply_scale_aware_overlap_degradation()`: straggler degradation at 4K+ GPU scale using sqrt model.
- Per-operator roofline analysis produces per-layer forward/backward timing, memory access patterns, and communication volumes.

**Training Parallelization (`training_parallelization.py`)**
- `TrainingParallelismConfig` dataclass with ZeRO stage support.
- `_estimate_memory_per_gpu()`: quick memory feasibility check for parallelism search.
- Extended `precision_bytes_map`: includes FP32, BF16, FP16, FP8_E4M3, FP8_E5M2, FP6, INT8, INT4, INT2.
- Complete optimizer bytes mapping covering 25+ optimizers including experimental ones (LOMO, ADALomo, BAdam, Muon, Apollo, GaLore).

**Training Stages (`training_stages.py`)**
- `TrainingStageType` enum with 13 training methods: SFT, DPO, PPO, KTO, RM, PT, ORPO, SimPO, GRPO, IPO, RLOO, REINFORCE, CPO.
- `TrainingStageConfig` dataclass defining per-stage:
  - Forward/backward compute multipliers (e.g., DPO=2x forward for policy+reference, PPO=4x).
  - Model requirements (reference model, reward model, value head).
  - Generation phase flag (PPO/GRPO/RLOO/REINFORCE need on-policy generation).
  - Memory multipliers.
- `TRAINING_STAGE_CONFIGS`: pre-defined configurations for all 13 stages.
- `validate_all_stage_configs()`: consistency validation.

### B. API Surface

```python
# training_modeling.py
training_modeling(model_config, system_config, parallelism_config,
                  training_config) -> TrainingResult
soft_bound(x, low, high, sharpness) -> float

# training_parallelization.py
TrainingParallelismConfig(tp, pp, dp, ep, cp, zero_stage)
predict_training_parallelism(model_config, hardware_config) -> TrainingParallelismConfig

# training_stages.py
TrainingStageType  # Enum
TrainingStageConfig  # Dataclass
TRAINING_STAGE_CONFIGS: Dict[TrainingStageType, TrainingStageConfig]
validate_all_stage_configs() -> bool
```

### C. Extension Points

1. **New training stages**: Add enum value to `TrainingStageType` and config to `TRAINING_STAGE_CONFIGS`.
2. **Custom attention types**: Extend `_detect_attention_type_from_config()` for new architectures.
3. **Custom overlap models**: Replace `calculate_effective_overlap()` with more sophisticated models.
4. **New precision types**: Add to `precision_bytes_map` in `training_parallelization.py`.
5. **New optimizers**: Add byte count to optimizer mapping.

### D. Research Enhancements

- **LLMServingSim's 0.97% accuracy**: The current roofline model is analytical. Integrating LLMServingSim's cycle-accurate simulation for critical path analysis could dramatically improve accuracy.
- **Pipeline schedule modeling**: Current PP overhead is a flat 0.85 multiplier. Research on 1F1B, interleaved, and zero-bubble schedules would provide schedule-specific bubble calculations.
- **Activation checkpointing strategies**: Beyond sqrt(L), selective checkpointing and offloading strategies from recent research (e.g., DTR - Dynamic Tensor Rematerialization) could be modeled.
- **Async pipeline parallelism**: Emerging techniques like DAPPLE and PipeDream-2BW use asynchronous weight updates to reduce pipeline bubbles.

---

## 4. Validation Subsystem

### Files Analyzed
- `validation/benchmark_validator.py`
- `validation/calibration_engine.py` (first 150 lines)
- `validation/accuracy_metrics.py` (first 150 lines)
- `validation/accuracy_reporter.py` (first 150 lines)

### A. Current Capabilities

**Benchmark Validator (`benchmark_validator.py`)**
- `BenchmarkValidator` with 7 validation rule categories:
  1. Memory bounds (per-hardware limits)
  2. MFU plausibility (per-training-type expected ranges)
  3. Parallelism consistency (TP*PP*DP == num_gpus)
  4. Throughput sanity (tokens/sec within reasonable bounds)
  5. Cost validation (non-negative, reasonable ranges)
  6. Configuration completeness
  7. Multi-model validation (PPO needs policy + reference + reward)
- `HARDWARE_MEMORY_LIMITS` and `HARDWARE_PEAK_TFLOPS` dictionaries for bounds checking.
- `MFU_EXPECTED_RANGES` per training type (e.g., SFT: 0.30-0.55, DPO: 0.20-0.45).
- `ValidationResult` with scoring: each error subtracts 0.2, each warning subtracts 0.05 from a 1.0 baseline.

**Calibration Engine (`calibration_engine.py`)**
- `CalibrationFactors` dataclass with:
  - Hardware efficiency factors (20+ hardware types).
  - Parallelism overhead multipliers per dimension (TP, PP, DP).
  - ZeRO stage overhead factors.
  - Method efficiency factors (SFT, DPO, PPO, etc.).
  - Model size scaling coefficients.
  - MoE-specific factors.
- Uses `scipy.optimize` to fit calibration parameters against published benchmark data.
- Calibration targets include Meta Llama training runs and NVIDIA Megatron benchmarks.

**Accuracy Metrics (`accuracy_metrics.py`)**
- `AccuracyMetrics` dataclass:
  - MAE (Mean Absolute Error)
  - MRE (Mean Relative Error)
  - Pearson and Spearman correlation
  - R-squared
  - Systematic bias
  - Percentile bounds (P5, P25, P75, P95)
  - Within-tolerance percentage
- `is_production_ready()`: returns True when MRE < 15%, correlation > 0.8, bias < 5%, tolerance > 85%.
- `get_improvement_suggestions()`: generates actionable recommendations based on which metrics fail.

**Accuracy Reporter (`accuracy_reporter.py`)**
- `AccuracyReport` with breakdown tables, outlier detection, category-level analysis.
- Markdown report generation with:
  - Executive summary with production-readiness assessment.
  - Error distribution analysis.
  - Per-category breakdown (by hardware, model size, training type).
  - Outlier identification and root cause hypotheses.
  - Improvement recommendations.

### B. API Surface

```python
# benchmark_validator.py
BenchmarkValidator.validate(simulation_result: Dict) -> ValidationResult
BenchmarkValidator.validate_batch(results: List[Dict]) -> List[ValidationResult]
ValidationResult.score: float
ValidationResult.errors: List[str]
ValidationResult.warnings: List[str]

# calibration_engine.py
CalibrationEngine.calibrate(benchmark_data: List[Dict]) -> CalibrationFactors
CalibrationEngine.apply_calibration(result: Dict, factors: CalibrationFactors) -> Dict

# accuracy_metrics.py
AccuracyMetrics.from_predictions(predicted, actual) -> AccuracyMetrics
AccuracyMetrics.is_production_ready() -> bool
AccuracyMetrics.get_improvement_suggestions() -> List[str]

# accuracy_reporter.py
AccuracyReporter.generate_report(metrics, breakdown) -> AccuracyReport
AccuracyReport.to_markdown() -> str
```

### C. Extension Points

1. **New validation rules**: Add methods to `BenchmarkValidator` and register in the validation pipeline.
2. **Custom calibration targets**: Add benchmark datasets for new hardware or training methods.
3. **New accuracy metrics**: Extend `AccuracyMetrics` dataclass with additional statistical measures.
4. **Report formats**: Add export methods beyond Markdown (JSON, HTML, PDF).
5. **Hardware-specific bounds**: Extend `HARDWARE_MEMORY_LIMITS` and `HARDWARE_PEAK_TFLOPS`.

### D. Research Enhancements

- **LLMServingSim validation targets**: LLMServingSim achieves 0.97% average error. The current validation framework's production-readiness threshold is MRE < 15%. Integrating LLMServingSim's validation methodology could guide calibration to achieve sub-1% accuracy.
- **Automated calibration pipeline**: Current calibration requires manual benchmark data collection. An automated pipeline that scrapes published training reports (MLPerf, papers) could keep calibration factors current.
- **Cross-validation across hardware families**: The calibration engine fits globally. Per-hardware-family calibration (NVIDIA vs AMD vs TPU) could improve accuracy within each family.
- **Temporal drift detection**: Training hardware and software evolve. Adding drift detection (comparing recent predictions to actual benchmarks) would flag when recalibration is needed.
- **Uncertainty quantification**: The current system produces point estimates. Adding confidence intervals or prediction distributions would better inform cluster planning decisions.

---

## 5. CPU Modeling Subsystem

### Files Analyzed
- `genz/cpu/cpu_system.py`
- `genz/cpu/cpu_operator.py` (first 150 lines)
- `genz/cpu/cpu_configs.py` (first 150 lines)
- `genz/cpu/cache_model.py`
- `genz/cpu/numa_model.py`

### A. Current Capabilities

**CPU System Model (`cpu_system.py`)**
- `CPUConfig` dataclass: cores, sockets, cache hierarchy (L1i, L1d, L2, L3 as `CacheConfig` objects), NUMA topology, ISA support, frequency curves, memory channels.
- `CPUSystem` wraps the base GenZ `System` with CPU-specific components:
  - `CacheHierarchy`: multi-level cache simulation.
  - `NUMATopology`: NUMA distance-aware memory access modeling.
  - `ISASelector`: selects optimal instruction set (AMX, AVX-512, AVX2, SVE2, NEON) based on operation characteristics.
  - `FrequencyGovernor`: models dynamic frequency scaling based on active cores and thermal state.
  - `ThreadingModel`: estimates threading efficiency with hyper-threading and core contention.
- `get_effective_bandwidth()`: returns cache-level-aware bandwidth with NUMA distance penalty and batch contention modeling.
- `get_effective_flops()`: returns ISA-aware, frequency-aware, vectorization-aware compute throughput.

**CPU Operator Mixin (`cpu_operator.py`)**
- `CPUOperatorMixin`: mixin class for base operators to add CPU-specific timing.
- `get_cpu_memory_time()`: simulates cache hierarchy hit/miss behavior for operator memory accesses. Models L1 -> L2 -> L3 -> DRAM cascade with per-level latency.
- `get_cpu_compute_time()`: selects optimal ISA, applies threading efficiency, and calculates compute-bound time.
- Integrates with the roofline model: `max(compute_time, memory_time)`.

**CPU Presets (`cpu_configs.py`)**
- Three detailed presets:
  - `intel_xeon_8380` (Ice Lake): 40 cores, AMX+AVX-512, 6-channel DDR4.
  - `amd_epyc_7763` (Zen3): 64 cores, AVX2 (no AVX-512), 8-channel DDR4.
  - `aws_graviton3` (Neoverse V1): 64 cores, SVE2+NEON, 8-channel DDR5.
- Each preset includes full cache hierarchy (L1i/L1d/L2/L3 with sizes, associativity, line sizes, latencies), NUMA distance matrix, ISA support flags, and frequency-vs-core-count curves.

**Cache Model (`cache_model.py`)**
- `CacheLevel` class: simulates a single cache level with set-associative structure.
  - Uses numpy arrays for tags, valid bits, and LRU counters.
  - `access(address, size, is_write)` -> `(hit: bool, cycles: int)`.
  - LRU replacement policy with eviction tracking.
  - Statistics: hits, misses, evictions.
- `CacheHierarchy` class: models complete L1d -> L2 -> L3 -> DRAM cascade.
  - `simulate_data_access(accesses: List[MemoryAccess])` -> timing breakdown with per-level hit rates.
  - `analyze_operator_access_pattern(operator)` -> generates `MemoryAccess` sequences from operator types:
    - **GEMM**: samples M*K + K*N access patterns with sqrt-scaled sampling for large matrices.
    - **Logit (prefill)**: quadratic Q*K^T access pattern with sqrt-scaled random sampling, capped at 50K samples.
    - **Logit (decode)**: models full KV cache read with batch-aware sampling.
    - **Attend**: similar to Logit with separate prefill/decode paths.
    - **FC**: smaller matrix access patterns.
  - Uses `MemoryAccess` dataclass (address, size, is_read, stride) and `AccessPattern` enum (SEQUENTIAL, RANDOM, STRIDED, TILED).

**NUMA Model (`numa_model.py`)**
- `NUMATopology` class: models NUMA memory access costs.
  - `get_numa_node(core_id)`: maps core to NUMA node.
  - `get_memory_node(address)`: maps memory address to NUMA node via allocation tracking.
  - `get_access_penalty(core_id, memory_address)`: calculates NUMA penalty factor (local=1.0x, remote=~2.1x).
  - `allocate_memory(size, preferred_node)`: tracks allocations with round-robin default policy.
  - `optimize_thread_placement(memory_footprint)`: optimizes thread-to-core mapping to minimize cross-NUMA access.

### B. API Surface

```python
# cpu_system.py
CPUConfig(cores, sockets, l1i_config, l1d_config, l2_config, l3_config, ...)
CPUSystem(cpu_config: CPUConfig)
CPUSystem.get_effective_bandwidth(data_size, cache_level, numa_distance) -> float
CPUSystem.get_effective_flops(op_type, data_type) -> float

# cpu_operator.py
CPUOperatorMixin.get_cpu_memory_time(data_size) -> float
CPUOperatorMixin.get_cpu_compute_time(flops) -> float

# cpu_configs.py
CPU_PRESETS: Dict[str, CPUConfig]  # 'intel_xeon_8380', 'amd_epyc_7763', 'aws_graviton3'

# cache_model.py
CacheLevel(config: CacheConfig)
CacheLevel.access(address, size, is_write) -> Tuple[bool, int]
CacheHierarchy(cpu_config)
CacheHierarchy.simulate_data_access(accesses) -> Dict[str, float]
CacheHierarchy.analyze_operator_access_pattern(operator) -> List[MemoryAccess]

# numa_model.py
NUMATopology(cpu_config)
NUMATopology.get_access_penalty(core_id, memory_address) -> float
NUMATopology.allocate_memory(size, preferred_node) -> Tuple[int, int]
NUMATopology.optimize_thread_placement(memory_footprint) -> Dict[int, int]
```

### C. Extension Points

1. **New CPU presets**: Add to `CPU_PRESETS` with full cache hierarchy and ISA support.
2. **New ISA types**: Extend `ISASelector` with new instruction sets (e.g., AVX10, SME for ARM).
3. **Cache replacement policies**: `CacheLevel` uses LRU; could be extended with RRIP, BIP, DRRIP.
4. **NUMA policies**: `allocate_memory` uses round-robin; could add first-touch, interleave, or bind policies.
5. **New operator access patterns**: Extend `analyze_operator_access_pattern` for new operator types.
6. **Frequency governor models**: Add new governor policies (performance, powersave, schedutil curves).

### D. Research Enhancements

- **Prefetch modeling**: The cache model does not simulate hardware prefetching. Adding stride-based and next-line prefetchers would improve accuracy for sequential access patterns.
- **TLB modeling**: No TLB (Translation Lookaside Buffer) simulation exists. Large matrix operations can be TLB-limited, especially with huge pages.
- **Memory controller contention**: The NUMA model tracks distance penalties but not memory controller bandwidth saturation under concurrent access.
- **SIMD auto-vectorization modeling**: The ISA selector picks the best ISA but does not model partial vectorization or scalar fallbacks for non-aligned data.
- **Power-performance curves**: CPU frequency scaling has non-linear power-performance tradeoffs. Integrating race-to-idle vs sustained compute analysis would improve energy estimates.
- **Intel AMX tiling**: The current model treats AMX as a FLOPS multiplier. Modeling tile register constraints and data layout requirements would improve AMX accuracy.

---

## 6. Features System

### Files Analyzed
- `genz/features/registry.py`
- `genz/features/base.py`
- `genz/features/decorators.py`

### A. Current Capabilities

**Base Feature Interface (`base.py`)**
- `FeatureCategory` enum: INFERENCE, HARDWARE, MODEL, PARALLELISM, OPTIMIZATION.
- `FeatureMetadata` dataclass: name, version, category, description, dependencies, incompatible_with, required_params, optional_params, min/max_genz_version.
- `BaseFeature` abstract class with:
  - `validate_config(config)` -> bool (abstract)
  - `apply(simulation_context)` -> Dict (abstract)
  - `initialize(config)` and `cleanup()` lifecycle methods.
  - `is_compatible_with(other_feature)` compatibility check.
- Specialized abstract subclasses:
  - `InferenceFeature`: adds `run_inference(model_config, system_config, simulation_params)`.
  - `ModelFeature`: adds `modify_model(model_operations)`.
  - `HardwareFeature`: adds `optimize_system(system_config)`.
  - `ParallelismFeature`: adds `configure_parallelism(parallelism_config)`.

**Feature Registry (`registry.py`)**
- `FeatureRegistry` class with auto-discovery via `pkgutil.iter_modules`.
- Scans the features package directory for classes extending `BaseFeature`.
- Registers 10 built-in features (pseudo-features mapping to existing GenZ functionality):
  - Inference: prefill, decode, chunked (mutually exclusive).
  - Model: lora.
  - Optimization: flash_attention, memory_offload, speculative_decode.
  - Parallelism: tensor_parallel, pipeline_parallel.
  - Hardware: cpu_optimization.
- `validate_feature_combination(features)`: checks incompatibilities and dependency resolution.
- `create_feature(name, config)`: factory method for feature instantiation.
- `get_features_by_category(category)`: category-based feature listing.

**Decorators (`decorators.py`)**
- `@register_feature(name, version, category, ...)`: class decorator for automatic registration with the global registry.
- `@feature_compatibility(*features)`: declares compatible features.
- `@requires_params(*params)`: declares required configuration parameters.
- `@optional_params(*params)`: declares optional parameters.
- `@depends_on(*features)`: declares feature dependencies.
- `@incompatible_with(*features)`: declares incompatibilities.
- `@validate_config(func)`: adds custom validation logic.
- `@post_process_results(func)`: adds post-processing to `apply()` results.

### B. API Surface

```python
# base.py
FeatureCategory  # Enum
FeatureMetadata  # Dataclass
BaseFeature.validate_config(config) -> bool  # Abstract
BaseFeature.apply(simulation_context) -> Dict  # Abstract
BaseFeature.initialize(config) -> None
BaseFeature.cleanup() -> None
BaseFeature.is_compatible_with(other) -> bool
InferenceFeature.run_inference(model, system, params) -> Dict  # Abstract
ModelFeature.modify_model(operations) -> List  # Abstract
HardwareFeature.optimize_system(system_config) -> Dict  # Abstract
ParallelismFeature.configure_parallelism(config) -> Dict  # Abstract

# registry.py
FeatureRegistry()
FeatureRegistry.get_available_features() -> List[str]
FeatureRegistry.get_feature_metadata(name) -> FeatureMetadata
FeatureRegistry.get_features_by_category(category) -> List[str]
FeatureRegistry.is_builtin_feature(name) -> bool
FeatureRegistry.create_feature(name, config) -> BaseFeature
FeatureRegistry.validate_feature_combination(features) -> bool
FeatureRegistry.get_feature_info(name) -> Dict

# decorators.py
@register_feature(name, version, category, ...)
@feature_compatibility(*features)
@requires_params(*params)
@optional_params(*params)
@depends_on(*features)
@incompatible_with(*features)
@validate_config(func)
@post_process_results(func)
```

### C. Extension Points

1. **New feature classes**: Create a Python file in the features directory with a class extending the appropriate `BaseFeature` subclass; auto-discovered by registry.
2. **Decorator-based registration**: Use `@register_feature` decorator for zero-config registration.
3. **Custom validation**: Use `@validate_config` decorator to add domain-specific validation.
4. **Post-processing hooks**: Use `@post_process_results` to transform feature outputs.
5. **New feature categories**: Extend `FeatureCategory` enum.
6. **Version constraints**: Use `min_genz_version` / `max_genz_version` for version-gated features.

### D. Research Enhancements

- **Feature composition engine**: The current system validates compatibility but does not compose features automatically. A composition engine could chain features in dependency order and manage shared state.
- **Dynamic feature selection**: Based on model/hardware characteristics, automatically enable/disable features (e.g., enable flash_attention only when sequence length > threshold).
- **Feature performance profiling**: Add instrumentation to measure per-feature simulation overhead, enabling cost-benefit analysis.
- **Conditional features**: Some optimizations are only beneficial under certain conditions (e.g., speculative decode only helps when draft model exists). Adding conditional activation logic would prevent misconfiguration.
- **Feature versioning and migration**: The version fields exist but no migration logic handles breaking changes between versions.

---

## 7. LoRA Subsystem

### Files Analyzed
- `lora/config.py`
- `lora/injection.py`

### A. Current Capabilities

**LoRA Configuration (`config.py`)**
- `LoraConfig` dataclass with:
  - `enabled`: toggle flag.
  - `rank`: LoRA rank for performance simulation.
  - `target_modules`: list of module types to target (e.g., `['attn', 'ffn']`).
  - `strategy`: `'dynamic'` (runtime A*B computation) or `'merge'` (one-time merge into base weights).
  - vLLM-style parameters: `max_loras` (concurrent adapters), `max_lora_rank`, `lora_dtype`, `fully_sharded_loras`.

**LoRA Injection (`injection.py`)**
- `inject_lora_ops(operations, model_config, parallelism_config, sequence_length)`: main entry point that processes an operation list and injects LoRA operations where appropriate.
- Two injection strategies:
  - **Merge strategy** (`inject_lora_merge`): inserts a `LORA_MERGE` operation before the base GEMM. Cost proportional to rank. Models one-time adapter merging.
  - **Dynamic strategy** (`inject_lora_dynamic`): replaces base GEMM with 4-operation sequence:
    1. Base GEMM (unchanged)
    2. `GEMM_LORA_A`: input x A (down projection, k -> rank)
    3. `GEMM_LORA_B`: lora_a_output x B (up projection, rank -> n)
    4. `ADD`: base_output + lora_output
  - LoRA weights use `ResidencyInfo.All_offchip` (not cached on-chip).
- `should_inject_lora(layer_name, target_modules)`: determines if a given layer name matches target modules using a mapping table (qkv/out_proj -> attn, up/gate/down/w1/w2/w3 -> ffn).
- TP-aware: reads `parallelism_config.tensor_parallel` for dimension adjustment.

### B. API Surface

```python
# config.py
LoraConfig(enabled, rank, target_modules, strategy, max_loras,
           max_lora_rank, lora_dtype, fully_sharded_loras)

# injection.py
inject_lora_ops(operations, model_config, parallelism_config, seq_length) -> List[List]
inject_lora_merge(base_gemm, model_config, parallelism_config, layer_name) -> List[List]
inject_lora_dynamic(base_gemm, model_config, parallelism_config, layer_name, seq_length) -> List[List]
should_inject_lora(layer_name, target_modules) -> bool
```

### C. Extension Points

1. **New LoRA variants**: Add QLoRA, DoRA, or PiSSA injection patterns alongside merge/dynamic.
2. **New target modules**: Extend the `op_to_module` mapping in `should_inject_lora`.
3. **Custom residency models**: Change `ResidencyInfo.All_offchip` to model cached adapters.
4. **Multi-adapter serving**: `max_loras` exists in config but injection logic currently handles single adapter; extend for concurrent adapter simulation.
5. **Adapter combination strategies**: Model adapter stacking, mixing, or switching overhead.

### D. Research Enhancements

- **QLoRA/GGUF quantization in simulation**: The injection currently models LoRA at full precision. Adding NF4/INT4 quantized adapter simulation would better model QLoRA serving workloads.
- **Multi-LoRA batching (S-LoRA)**: Research on serving multiple LoRA adapters simultaneously (S-LoRA, Punica) uses unified paging and custom CUDA kernels. Modeling the batching overhead and memory sharing would improve serving estimates.
- **DoRA (Weight-Decomposed Low-Rank Adaptation)**: DoRA decomposes into magnitude and direction components. Adding DoRA-specific injection would model its additional normalization overhead.
- **LoRA rank adaptation**: Dynamic rank selection during serving (different ranks per layer) is not modeled.
- **Adapter caching and swapping**: For multi-tenant serving, adapter swapping between GPU and CPU memory has latency costs that are not captured.

---

## Cross-Subsystem Integration Observations

### Existing Integration Points
1. **Training calculator -> Hardware configs**: Memory calculator uses `HARDWARE_CONFIGS` for GPU memory limits.
2. **Advanced calculator -> GenZ training modeling**: `calculate_training_with_genz()` bridges the memory calculator with the GenZ roofline engine.
3. **Cluster optimizer -> Training modeling**: Uses `training_modeling()` for accurate per-config timing.
4. **Cluster optimizer -> TCO calculator**: Integrates cost estimation into cluster ranking.
5. **LoRA injection -> GenZ operator framework**: Injects operations into the GenZ operator pipeline.
6. **CPU system -> Base System**: `CPUSystem` wraps GenZ `System` with CPU-specific enhancements.
7. **Features registry -> All subsystems**: Feature flags control which capabilities are enabled.

### Integration Gaps
1. **Validation <-> Training**: The calibration engine and benchmark validator are not automatically invoked after training simulations. Adding a validation pipeline would flag implausible results before they reach users.
2. **CPU modeling <-> Training**: CPU-specific training (CPU-only or CPU-offload scenarios) is not fully integrated with the training memory calculator.
3. **LoRA <-> Training calculator**: The memory calculator handles LoRA memory but the injection logic and memory calculator are not cross-referenced for consistency.
4. **Features system <-> Everything**: The feature registry exists but is not wired into the main simulation pipeline as a mandatory gating mechanism. Features are currently more of a catalog than an active configuration system.
5. **Scaling laws <-> Time estimator**: Both estimate training time but use different models. The scaling laws module's composite efficiency is not fed into the time estimator's TPS calculation.
