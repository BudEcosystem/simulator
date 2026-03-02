# BudSimulator API & SDK Comprehensive Analysis

## 1. Complete API Inventory

### 1.1 Root & Health Endpoints (apis/main.py, apis/health.py)

| Endpoint | Method | Prefix | Description | GenZ Function Called |
|---|---|---|---|---|
| `/` | GET | - | Root endpoint, returns API info | None |
| `/api/health` | GET | /api | Comprehensive health check (system, DB, dependencies) | None |
| `/api/health/diagnostics` | GET | /api | Detailed system diagnostics (CPU, memory, disk) | None |
| `/api/health/ready` | GET | /api | Simple readiness check (DB connectivity) | None |

### 1.2 Models Endpoints (apis/routers/models.py)

| Endpoint | Method | Prefix | Description | GenZ/SDK Function Called |
|---|---|---|---|---|
| `/api/models/validate` | POST | /api/models | Validate model URL/ID accessibility | `HuggingFaceConfigLoader.fetch_model_config()`, `ModelManager.get_model()` |
| `/api/models/{model_id}/config` | GET | /api/models | Get model config + metadata + analysis | `ModelManager.get_model_config()`, `ModelMemoryCalculator.detect_model_type()`, `detect_attention_type()` |
| `/api/models/calculate` | POST | /api/models | Calculate memory requirements | `estimate_memory()` from llm_memory_calculator |
| `/api/models/compare` | POST | /api/models | Compare memory across multiple models | `estimate_memory()` per model |
| `/api/models/analyze` | POST | /api/models | Analyze efficiency across sequence lengths | `estimate_memory()` per sequence length |
| `/api/models/config/submit` | POST | /api/models | Submit gated model config + full analysis pipeline | `ModelMemoryCalculator.detect_model_type()`, `detect_attention_type()`, `call_bud_LLM()` |
| `/api/models/popular` | GET | /api/models | Get popular models with metadata | `MODEL_DICT`, `ModelManager`, `HuggingFaceConfigLoader.get_model_info()` |
| `/api/models/list` | GET | /api/models | List all models (MODEL_DICT + database) | `MODEL_DICT.models`, `ModelManager.list_models()` |
| `/api/models/filter` | GET | /api/models | Filter models by criteria | Uses `list_all_models()` + filters |
| `/api/models/add/huggingface` | POST | /api/models | Add model from HuggingFace | `HuggingFaceModelImporter.import_model()` |
| `/api/models/add/config` | POST | /api/models | Add model from config dict | `ModelMemoryCalculator.detect_model_type()`, `detect_attention_type()` |

### 1.3 Hardware Endpoints (apis/routers/hardware.py)

| Endpoint | Method | Prefix | Description | GenZ/SDK Function Called |
|---|---|---|---|---|
| `/api/hardware` | POST | /api/hardware | Create new hardware entry | `BudHardware.add_hardware()` |
| `/api/hardware` | GET | /api/hardware | List hardware with filters | `BudHardware.search_hardware()` |
| `/api/hardware/filter` | GET | /api/hardware | Advanced hardware filtering | `BudHardware.search_hardware_extended()` |
| `/api/hardware/{name}` | GET | /api/hardware | Get detailed hardware info + vendors + clouds | `BudHardware.get_hardware_by_name()`, `get_hardware_vendors()`, `get_hardware_clouds()` |
| `/api/hardware/{name}` | PUT | /api/hardware | Update hardware | `BudHardware.update_hardware()` |
| `/api/hardware/{name}` | DELETE | /api/hardware | Delete hardware (soft/hard) | `BudHardware.delete_hardware()` |
| `/api/hardware/recommend` | POST | /api/hardware | Hardware recommendations by memory | `HardwareRecommendation.recommend_hardware()` |
| `/api/hardware/vendors/{name}` | GET | /api/hardware | Get vendor info + pricing | `BudHardware.get_hardware_vendors()` |
| `/api/hardware/clouds/{name}` | GET | /api/hardware | Get cloud availability + pricing | `BudHardware.get_hardware_clouds()` |

### 1.4 Usecase Endpoints (apis/routers/usecases.py)

| Endpoint | Method | Prefix | Description | GenZ/SDK Function Called |
|---|---|---|---|---|
| `/api/usecases` | POST | /api/usecases | Create usecase | `BudUsecases.add_usecase()` |
| `/api/usecases` | GET | /api/usecases | List usecases with filters | `BudUsecases.search_usecases()` |
| `/api/usecases/search` | GET | /api/usecases | Advanced usecase search | `BudUsecases.search_usecases()` |
| `/api/usecases/stats` | GET | /api/usecases | Usecase statistics | `BudUsecases.get_stats()` |
| `/api/usecases/{id}` | GET | /api/usecases | Get specific usecase | `BudUsecases.get_usecase()` |
| `/api/usecases/{id}` | PUT | /api/usecases | Update usecase | `BudUsecases.update_usecase()` |
| `/api/usecases/{id}` | DELETE | /api/usecases | Delete usecase | `BudUsecases.delete_usecase()` |
| `/api/usecases/import` | POST | /api/usecases | Import from JSON | `BudUsecases.import_from_json()` |
| `/api/usecases/export` | POST | /api/usecases | Export to JSON | `BudUsecases.export_to_json()` |
| `/api/usecases/industries/list` | GET | /api/usecases | List industries | `BudUsecases.get_all_usecases()` |
| `/api/usecases/tags/list` | GET | /api/usecases | List tags with counts | `BudUsecases.get_stats()` |
| `/api/usecases/{id}/recommendations` | POST | /api/usecases | Model+hardware recommendations for usecase | `ModelMemoryCalculator.calculate_total_memory()`, `_estimate_performance()`, `_get_hardware_recommendations()` |

### 1.5 Usecase Optimization Endpoints (apis/routers/usecases_optimization.py)

| Endpoint | Method | Prefix | Description | GenZ/SDK Function Called |
|---|---|---|---|---|
| `/api/usecases/{id}/optimize-hardware` | POST | /api/usecases | Full GenZ-based hardware optimization | `find_best_hardware_for_usecase()` -> `estimate_prefill_performance()`, `estimate_decode_performance()` |
| `/api/usecases/{id}/quick-optimization` | GET | /api/usecases | Quick single-config optimization | `find_best_hardware_for_usecase()` |

### 1.6 Training/Simulator Endpoints (apis/routers/training.py)

| Endpoint | Method | Prefix | Description | GenZ/SDK Function Called |
|---|---|---|---|---|
| `/api/simulator/estimate-training` | POST | /api/simulator | Training memory estimation | `TrainingMemoryCalculator.calculate_training_memory()` |
| `/api/simulator/recommend-cluster` | POST | /api/simulator | Cluster recommendations for training | `TrainingClusterSelector.recommend_clusters()` |
| `/api/simulator/check-fit` | POST | /api/simulator | Check if training fits in cluster | `TrainingClusterSelector.check_fit()` |
| `/api/simulator/estimate-time` | POST | /api/simulator | Training time + cost estimation | `TrainingTimeEstimator.estimate_training_time()` |
| `/api/simulator/hardware` | GET | /api/simulator | List available hardware profiles | `TrainingClusterSelector.list_available_hardware()` |

---

## 2. SDK Public API Surface (llm-memory-calculator)

### 2.1 Exported Classes

| Class | Module | Description |
|---|---|---|
| `ModelMemoryCalculator` | calculator.py | Core memory calculator for inference. Methods: `calculate_memory()`, `detect_model_type()`, `detect_attention_type()`, `calculate_total_memory()`, `calculate_parameters()` |
| `UniversalParameterCounter` | parameter_counter.py | Accurate parameter counting across architectures |
| `HuggingFaceConfigLoader` | huggingface_loader.py | Load configs from HuggingFace Hub. Methods: `get_model_config()`, `fetch_model_config()`, `get_model_info()`, `analyze_model()`, `compare_models()` |
| `MemoryReport` | types.py | Dataclass for memory breakdown. Properties: `total_memory_gb`, `weight_memory_gb`, `kv_cache_gb`, `activation_memory_gb`, `state_memory_gb`, `image_memory_gb`, `extra_work_gb`, `lora_adapter_memory_gb`, `recommended_gpu_memory_gb`, `can_fit_24gb_gpu`, `can_fit_80gb_gpu` |
| `HardwareManager` | hardware.py | Hardware configuration management with DB backend |
| `ConfigNormalizer` | config_normalizer.py | Normalizes diverse model configs (MoE keys, quantization, per-layer attention) |

### 2.2 Exported Functions

#### Memory & Analysis Functions

| Function | Signature | Description |
|---|---|---|
| `calculate_memory` | `(model_id_or_config, **kwargs) -> MemoryReport` | Calculate memory from model ID or config |
| `estimate_memory` | `(config, batch_size=1, seq_length=2048, num_images=0, precision='fp16', include_gradients=False, decode_length=0) -> MemoryReport` | Estimate memory from config dict |
| `analyze_hf_model` | `(model_id, seq_length=2048, precision='fp16', token=None) -> Dict` | Full model analysis from HuggingFace |
| `compare_models` | `(model_ids, seq_length=2048, precision='fp16', **kwargs) -> List[Dict]` | Compare memory across models |
| `estimate_max_batch_size` | `(config, gpu_memory_gb, seq_length, precision) -> int` | Find max batch size for given GPU memory |
| `analyze_attention_efficiency` | `(config, seq_lengths, batch_size, precision) -> Dict` | Analyze attention memory efficiency |

#### Hardware Functions

| Function | Signature | Description |
|---|---|---|
| `get_hardware_config` | `(hardware_name: str) -> Dict` | Get hardware config by name from DB/system_configs |
| `get_all_hardware` | `() -> List[Dict]` | Get all hardware configs |
| `get_hardware_by_type` | `(hw_type: str) -> List[Dict]` | Filter hardware by type (gpu/cpu/accelerator/asic) |
| `get_hardware_by_manufacturer` | `(manufacturer: str) -> List[Dict]` | Filter hardware by manufacturer |
| `search_hardware` | `(**filters) -> List[Dict]` | Search hardware with filters |
| `set_hardware_db_path` | `(path: str) -> None` | Set custom database path |

#### Performance Estimation Functions (conditional import)

| Function | Signature | Description |
|---|---|---|
| `estimate_prefill_performance` | `(model, batch_size, input_tokens, system_name, bits, tensor_parallel, pipeline_parallel, expert_parallel, debug) -> Dict` | Prefill phase performance (TTFT, throughput) via GenZ `prefill_moddeling()` |
| `estimate_decode_performance` | `(model, batch_size, beam_size, input_tokens, output_tokens, system_name, bits, tensor_parallel, pipeline_parallel, expert_parallel, debug) -> Dict` | Decode phase performance (TPOT, throughput) via GenZ `decode_moddeling()` |
| `estimate_end_to_end_performance` | `(model, batch_size, beam_size, input_tokens, output_tokens, system_name, bits, ...) -> Dict` | Combined prefill+decode E2E performance |
| `estimate_chunked_performance` | `(model, batch_size, input_tokens, output_tokens, chunk_size, system_name, bits, ...) -> Dict` | Chunked prefill processing via GenZ `chunked_moddeling()` |
| `compare_performance_configurations` | `(model, configurations, batch_size, input_tokens, output_tokens) -> List[Dict]` | Compare across hardware+parallelism configs |

#### Parallelism Optimization Functions (conditional import)

| Function | Signature | Description |
|---|---|---|
| `get_various_parallelization` | `(model, total_nodes) -> Set[Tuple[int,int]]` | Get valid TP/PP combinations |
| `get_best_parallelization_strategy` | `(stage, model, total_nodes, batch_size, beam_size, input_tokens, output_tokens, system_name, bits) -> DataFrame/Dict` | Find optimal parallelism strategy |
| `get_pareto_optimal_performance` | `(model, total_nodes, ...) -> Dict` | Get Pareto-optimal performance configs |
| `get_minimum_system_size` | `(model, ...) -> int` | Find minimum nodes for model |

### 2.3 Training Module (llm_memory_calculator.training)

| Class | Description |
|---|---|
| `TrainingMemoryCalculator` | Calculate training memory (weights, gradients, optimizer states, activations). Methods: `calculate_training_memory(config, batch_size, seq_length, precision, method, optimizer, gradient_checkpointing, lora_rank, lora_alpha, freeze_layers, deepspeed_stage, tensor_parallel, data_parallel, framework_overhead_percent)` |
| `TrainingClusterSelector` | Recommend GPU clusters. Methods: `recommend_clusters(training_estimate, prefer_cost, max_budget_per_hour, available_hardware, max_gpus)`, `check_fit(training_estimate, hardware, num_gpus)`, `list_available_hardware()` |
| `TrainingTimeEstimator` | Estimate training duration. Methods: `estimate_training_time(model_config, dataset_tokens, batch_size, gradient_accumulation, epochs, hardware, num_gpus, seq_length, parallelism)` |

---

## 3. Core Service Layer

### 3.1 BudSimulator (src/bud_sim.py)

**Class**: `BudSimulator`
- `SimType` enum: USECASE_SIM, BEST_MODEL_SIM, BEST_HARDWARE_SIM, PARALLELISATION_STRATEGY_SIM, HETEROGENEOUS_SIM, POWER_CONSUMPTION_SIM, COST_SIM, YTD_SIM
- `SimulationConfig` Pydantic model: models, batch_size, precision, decode_length, usecases, hardwares, features
- `run(**kwargs) -> Dict`: Delegates to GenZ `SimulationEngine.simulate()`, returns latency, throughput, runtime_breakdown, memory_usage, hardware_utilization, feature_metrics
- `get_supported_features() -> List[Dict]`: Queries GenZ feature registry
- NOT yet exposed via API endpoints

### 3.2 BudModels (src/bud_models.py)

**Re-exports from llm_memory_calculator**: `UniversalParameterCounter`, `HuggingFaceConfigLoader`, `ModelMemoryCalculator`, `MemoryReport`, `estimate_memory`, `analyze_hf_model`, `estimate_max_batch_size`, `analyze_attention_efficiency`

**Additional functions**:
- `estimate_max_sequence_length(config, gpu_memory_gb, batch_size, precision) -> int` - Binary search for max seq length
- `get_model_config_from_hf(model_id, token, add_param_count) -> Dict` - Convenience wrapper
- `compare_hf_models(model_ids, seq_length, precision, token, print_results) -> List[Dict]` - Compare with formatted output

### 3.3 BudAI (src/bud_ai.py)

- `call_bud_LLM(prompt, model, system_prompt, temperature, max_tokens, ...) -> str` - Calls LLM API via OpenAI client
- `update_llm_url_and_model(base_url, model) -> Dict` - Update LLM settings in DB
- `get_llm_url_and_model() -> Dict` - Get current LLM settings

### 3.4 BudHardware (src/hardware.py)

**Extends** `HardwareManager` from llm_memory_calculator. Adds:
- Database-backed CRUD operations (add, update, delete)
- JSON import/export
- Vendor and cloud pricing management
- Schema migration support
- `calculate_price_indicator(flops, memory_gb, bandwidth_gbs) -> float` (static method)
- `search_hardware_extended(**filters) -> List[Dict]` - Advanced filtering with sorting

### 3.5 HardwareRecommendation (src/hardware_recommendation.py)

- `recommend_hardware(total_memory_gb, model_params_b) -> Dict` - Intelligent HW recommendation with CPU/GPU separation, batch recommendations for small models

### 3.6 HardwareRegistry (src/hardware_registry.py)

- Singleton that syncs DB hardware into GenZ `system_configs` at startup
- `initialize()`, `get_hardware_instance()`, `get_cpu_configs()`, `get_hardware_by_type()`, `refresh()`

### 3.7 HardwareOptimizer (src/optimization/hardware_optimizer.py)

- `find_best_hardware_for_usecase(usecase, batch_size, model_sizes) -> List[Dict]` - Tests multiple model sizes against hardware types, uses GenZ performance estimation
- `find_optimal_configuration(model_id, hardware_type, batch_size, usecase) -> Optional[Dict]` - Finds optimal config for specific model+hardware pair
- `evaluate_with_genz_direct(model_id, system, parallelism, batch_size, usecase, num_nodes) -> Optional[Dict]` - Direct GenZ performance evaluation
- `rank_by_cost_effectiveness(configurations) -> List[Dict]` - Rank by cost per request + SLO headroom

---

## 4. Database Schema

### 4.1 Models Schema (schema.py) - Version 2
- **models**: id, model_id (unique), model_name, source, config_json, model_type, attention_type, parameter_count, logo, model_analysis, created_at, updated_at, is_active
- **model_cache**: id, model_id, cache_key, cache_value, expires_at
- **model_versions**: id, model_id, version, config_json, change_description
- **model_quality_metrics**: id, model_id, metric_name, metric_value, shots, metadata
- **user_model_configs**: id, model_uri (unique), config_json, validated
- **schema_version**: version, applied_at

### 4.2 Hardware Schema (hardware_schema.py) - Version 2
- **hardware**: id, name (unique), manufacturer, type, flops, memory_size, memory_bw, icn, icn_ll, power, real_values, url, description, source, is_active
- **hardware_advantages**: id, hardware_id, advantage
- **hardware_disadvantages**: id, hardware_id, disadvantage
- **hardware_on_prem_vendors**: id, hardware_id, vendor_name, price_lower, price_upper, currency
- **hardware_clouds**: id, hardware_id, cloud_name
- **hardware_cloud_instances**: id, cloud_id, instance_name, price_lower, price_upper, price_unit, currency
- **hardware_cloud_regions**: id, cloud_id, region
- **hardware_legacy_pricing**: migration table
- **Views**: hardware_with_on_prem_vendors, hardware_with_clouds, hardware_cloud_instance_details, hardware_performance_metrics

### 4.3 Usecase Schema (usecase_schema.py) - Version 1
- **usecases**: id, unique_id (unique), name, industry, description, batch_size, beam_size, input_tokens_min/max, output_tokens_min/max, ttft_min/max, e2e_min/max, inter_token_min/max, source, is_active
- **usecase_tags**: id, usecase_id, tag
- **usecase_versions**: id, usecase_id, version, config_json, change_description
- **usecase_cache**: id, usecase_id, cache_key, cache_value, expires_at
- **Views**: usecases_with_tags, usecase_performance_profiles, usecases_by_industry

### 4.4 Additional Tables (created at runtime)
- **model_recommendation_categories**: id, category, model_id, display_order, is_active (created by usecases.py)
- **llm_settings**: (used by bud_ai.py via LLMSettingsManager)

---

## 5. Backward Compatibility Constraints

### 5.1 MUST NOT Break

1. **Endpoint URLs and methods**: All existing endpoints must remain at their current paths. The frontend depends on `/api/models/*`, `/api/hardware/*`, `/api/usecases/*`, `/api/simulator/*`.

2. **Response schema fields**: Existing response fields MUST NOT be removed or renamed. New fields can be added with `Optional` defaults.

3. **Request schema fields**: Existing required fields must remain required. Existing optional fields must remain optional. New fields must be optional with defaults.

4. **MODEL_DICT interface**: The `MODEL_DICT` object must maintain `.models`, `.get_model()`, `.list_models()`, `.get_model_metadata()` methods.

5. **estimate_memory() signature**: `(config, batch_size, seq_length, num_images, precision, include_gradients, decode_length)` - all downstream callers depend on this.

6. **MemoryReport properties**: All properties (`total_memory_gb`, `weight_memory_gb`, `kv_cache_gb`, etc.) used by API response builders.

7. **GenZ function signatures**: `estimate_prefill_performance()`, `estimate_decode_performance()` - called from HardwareOptimizer with specific parameter names.

8. **Database schema**: Existing tables and columns must not be removed. Migrations must handle upgrades gracefully.

9. **Usecase unique_id**: The codebase uses `unique_id` (string), NOT numeric `id` for usecase operations. This is a documented critical constraint.

10. **Hardware key normalization**: The `_KEY_MAP` in hardware.py that maps PascalCase GenZ keys (Flops, Memory_size, Memory_BW) to lowercase Pydantic model fields.

### 5.2 Backward Compatibility Strategy for Adding New Features

- **Add new endpoints** under existing router prefixes with new paths (e.g., `/api/models/power-profile`)
- **Add new optional fields** to existing Pydantic models with `= None` or `= Field(None, ...)`
- **Add new response classes** that extend existing ones (e.g., `class EnhancedMemoryBreakdown(MemoryBreakdown)`)
- **Add new routers** for entirely new domains (e.g., `prefix="/api/simulation"`)
- **Extend MemoryReport** with new optional properties; keep existing properties unchanged
- **Database additions**: Use Alembic migrations, add new tables/columns with nullable defaults

---

## 6. Extension Points for New Features

### 6.1 Architecture Extension Points

1. **Router-level**: Add new routers in `apis/routers/` and register in `apis/main.py` with `app.include_router()`
2. **Service-level**: Add new service classes in `src/` following BudHardware/BudUsecases pattern
3. **SDK-level**: Add new modules in `llm-memory-calculator/src/llm_memory_calculator/` and export from `__init__.py`
4. **Schema-level**: Add new schema files in `src/db/` following existing pattern, bump schema version
5. **BudSimulator**: The `BudSimulator` class and its `SimType` enum can be extended with new simulation types

### 6.2 Specific New Endpoints Needed

#### A. Power Modeling

**New endpoints** (extend hardware router or add new router):

```
POST /api/hardware/{name}/power-profile
  Request: { batch_sizes: [1,8,32], model_id: str, precision: str }
  Response: { hardware_name, idle_power_w, peak_power_w, power_per_batch: [{batch_size, power_w, efficiency_tflops_per_watt}] }

POST /api/models/power-estimate
  Request: { model_id, hardware_name, batch_size, seq_length, precision }
  Response: { total_power_w, compute_power_w, memory_power_w, interconnect_power_w, pue_adjusted_power_w, cost_per_kwh }
```

**Schema extension**: Add `power_tdp`, `power_idle`, `pue_factor` to hardware table.

**SDK extension**: Add `estimate_power_consumption()` to performance_estimator.py. GenZ's System class already has a `Power` field that is currently passed through but not deeply modeled.

#### B. Prefix Caching Effects

**New endpoints** (extend models router):

```
POST /api/models/prefix-cache-analysis
  Request: { model_id, prefix_length, total_seq_length, batch_size, cache_hit_rate, hardware_name }
  Response: { without_caching: {ttft, memory_gb, kv_cache_gb}, with_caching: {ttft, memory_gb, kv_cache_gb, cache_memory_gb}, speedup_factor, memory_savings_gb }
```

**SDK extension**: Add `estimate_prefix_cache_memory()` and `estimate_prefix_cache_performance()` to performance_estimator.py. This requires extending GenZ's KV cache calculation to account for shared prefix tokens.

#### C. PD Disaggregation (Prefill-Decode Disaggregation)

**New endpoints** (add new router `apis/routers/disaggregation.py`):

```
POST /api/simulation/pd-disaggregation
  Request: { model_id, prefill_hardware, decode_hardware, prefill_tp, decode_tp, batch_size, input_tokens, output_tokens, kv_transfer_bandwidth_gbps }
  Response: { prefill_latency, decode_latency, kv_transfer_latency, total_latency, prefill_utilization, decode_utilization, prefill_memory, decode_memory, cost_comparison_vs_unified }

POST /api/simulation/pd-disaggregation/optimize
  Request: { model_id, usecase_id, available_hardware, budget_per_hour }
  Response: { optimal_prefill_config, optimal_decode_config, kv_transfer_plan, total_cost, vs_unified_cost_savings }
```

**SDK extension**: Add `estimate_disaggregated_performance()` to performance_estimator.py. This requires modeling the KV cache transfer overhead between prefill and decode nodes. GenZ already models prefill and decode separately -- the new component is the inter-phase transfer.

#### D. Auto-Tuning

**New endpoints** (add new router `apis/routers/autotuning.py`):

```
POST /api/autotuning/optimize
  Request: { model_id, hardware_name, target_metric: "throughput"|"latency"|"cost", constraints: {max_ttft, max_tpot, max_memory_gb, max_gpus}, search_space: {batch_sizes, parallelism_configs, precisions, chunk_sizes} }
  Response: { best_config: {batch_size, tp, pp, precision, chunk_size}, performance: {ttft, tpot, throughput, memory}, search_results: [{config, performance, score}], pareto_frontier: [{config, throughput, latency}] }

POST /api/autotuning/sweep
  Request: { model_id, hardware_configs: [{name, count}], usecase_id }
  Response: { sweep_results: [{hardware, config, performance, cost, meets_slo}], optimal_by_cost, optimal_by_performance }
```

**SDK extension**: Add `auto_tune()` and `parameter_sweep()` to parallelism_optimizer.py. GenZ already has `get_best_parallelization_strategy()` and `get_pareto_optimal_performance()` - these can be composed into a higher-level auto-tuning loop.

#### E. Runtime Simulation

**New endpoints** (expose BudSimulator via new router `apis/routers/simulation.py`):

```
POST /api/simulation/run
  Request: SimulationConfig (the existing Pydantic model from bud_sim.py)
  Response: { latency, throughput, runtime_breakdown, memory_usage, hardware_utilization, feature_metrics }

GET /api/simulation/features
  Response: { features: [{name, description, parameters}] }

POST /api/simulation/batch
  Request: { simulations: [SimulationConfig, ...] }
  Response: { results: [SimulationResult, ...], comparison_table }

POST /api/simulation/what-if
  Request: { base_config: SimulationConfig, variations: [{param, values}] }
  Response: { base_result, variations: [{param_value, result, delta_vs_base}] }
```

**Implementation**: The `BudSimulator` class already exists in `src/bud_sim.py` with full GenZ `SimulationEngine` integration. It just needs API endpoint exposure. The `SimType` enum already covers 8 simulation types.

---

## 7. Schema Extensions Needed

### 7.1 Hardware Table Extensions

```sql
ALTER TABLE hardware ADD COLUMN power_tdp REAL;          -- TDP in watts
ALTER TABLE hardware ADD COLUMN power_idle REAL;          -- Idle power in watts
ALTER TABLE hardware ADD COLUMN pue_factor REAL DEFAULT 1.2;  -- Power Usage Effectiveness
ALTER TABLE hardware ADD COLUMN interconnect_type TEXT;   -- NVLink, InfiniBand, PCIe, etc.
ALTER TABLE hardware ADD COLUMN interconnect_version TEXT; -- e.g., "NVLink 4.0"
ALTER TABLE hardware ADD COLUMN gpu_architecture TEXT;     -- e.g., "Hopper", "Ada Lovelace"
ALTER TABLE hardware ADD COLUMN compute_capability TEXT;   -- e.g., "9.0"
ALTER TABLE hardware ADD COLUMN fp8_flops REAL;           -- FP8 TFLOPS (separate from bf16)
ALTER TABLE hardware ADD COLUMN sparsity_flops REAL;      -- Sparsity-accelerated TFLOPS
```

### 7.2 New Simulation Results Table

```sql
CREATE TABLE simulation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    simulation_type TEXT NOT NULL,
    model_id TEXT NOT NULL,
    hardware_name TEXT NOT NULL,
    config_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    latency_ms REAL,
    throughput_tps REAL,
    memory_used_gb REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES models(model_id)
);
```

### 7.3 New Usecase Extensions

```sql
ALTER TABLE usecases ADD COLUMN concurrency_target INTEGER DEFAULT 1;
ALTER TABLE usecases ADD COLUMN throughput_target_tps REAL;  -- Tokens/second target
ALTER TABLE usecases ADD COLUMN cost_budget_per_hour REAL;   -- Max $/hour budget
ALTER TABLE usecases ADD COLUMN prefix_sharing_enabled BOOLEAN DEFAULT 0;
ALTER TABLE usecases ADD COLUMN common_prefix_length INTEGER DEFAULT 0;
```

### 7.4 New Model Optimization Cache Table

```sql
CREATE TABLE model_optimization_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    usecase_id TEXT,
    hardware_name TEXT NOT NULL,
    optimization_type TEXT NOT NULL,    -- 'auto_tune', 'pd_disagg', 'sweep'
    config_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES models(model_id)
);
```

---

## 8. Request/Response Schema Catalog

### 8.1 Models Router Schemas (apis/schemas.py)

**Request Schemas:**
- `ValidateModelRequest` { model_url: str }
- `CalculateMemoryRequest` { model_id, precision, batch_size, seq_length, num_images, include_gradients, decode_length }
- `CompareModelsRequest` { models: List[CompareModelConfig] }
- `CompareModelConfig` { model_id, precision, batch_size, seq_length }
- `AnalyzeModelRequest` { model_id, precision, batch_size, sequence_lengths: List[int] }
- `AddModelFromHFRequest` { model_uri, auto_import }
- `AddModelFromConfigRequest` { model_id, config: Dict, metadata: Optional[Dict] }
- `ConfigSubmitRequest` { model_uri, config: Dict }

**Response Schemas:**
- `ValidateModelResponse` { valid, error, error_code, model_id, requires_config, config_submission_url }
- `ModelConfigResponse` { model_id, model_type, attention_type, parameter_count, architecture, logo, model_analysis, config: ModelConfig, metadata: ModelMetadata }
- `CalculateMemoryResponse` { model_type, attention_type, precision, parameter_count, memory_breakdown: MemoryBreakdown, total_memory_gb, recommendations: MemoryRecommendations }
- `CompareModelsResponse` { comparisons: List[ModelComparison] }
- `AnalyzeModelResponse` { model_id, attention_type, analysis: Dict[str, SequenceAnalysis], insights: AnalysisInsights }
- `PopularModelsResponse` { models: List[PopularModel] }
- `ListModelsResponse` { total_count, model_dict_count, database_count, models: List[ModelSummary] }
- `FilterModelsResponse` { total_count, filters_applied, models: List[ModelSummary] }
- `AddModelResponse` { success, model_id, message, source, already_existed }
- `ConfigSubmitResponse` { success, message, model_id, validation, error_code, missing_fields }

**Nested Schemas:**
- `ModelConfig` { hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, intermediate_size, vocab_size, max_position_embeddings, activation_function }
- `ModelMetadata` { downloads, likes, size_gb, tags }
- `MemoryBreakdown` { weight_memory_gb, kv_cache_gb, activation_memory_gb, state_memory_gb, image_memory_gb, extra_work_gb }
- `MemoryRecommendations` { recommended_gpu_memory_gb, can_fit_24gb_gpu, can_fit_80gb_gpu, min_gpu_memory_gb }
- `ModelComparison` { model_id, model_name, total_memory_gb, memory_breakdown, recommendations }
- `SequenceAnalysis` { total_memory_gb, kv_cache_gb, kv_cache_percent }
- `AnalysisInsights` { memory_per_token_bytes, efficiency_rating, recommendations }
- `PopularModel` { model_id, name, parameters, model_type, attention_type, downloads, likes, description, logo }
- `ModelSummary` { model_id, name, author, model_type, attention_type, parameter_count, logo, model_analysis, source, in_model_dict, in_database }

### 8.2 Hardware Router Schemas

**Request Schemas:**
- `HardwareCreate` { name, type, manufacturer, flops, memory_size, memory_bw, icn, icn_ll, power, real_values, url, description, on_prem_vendors, clouds }
- `HardwareUpdate` { type, manufacturer, flops, memory_size, memory_bw, icn, icn_ll, power, url, description }
- `RecommendationRequest` { total_memory_gb, model_params_b }

**Response Schemas:**
- `HardwareResponse` { name, type, manufacturer, flops, memory_size, memory_bw, icn, icn_ll, power, real_values, url, description, on_prem_vendors, clouds, min_on_prem_price, max_on_prem_price, source, price_approx }
- `HardwareDetailResponse` { ..., vendors: List[Dict], clouds: List[Dict] }
- `RecommendationResponse` { hardware_name, nodes_required, memory_per_chip, manufacturer, type, optimality, utilization, total_memory_available, batch_recommendations, price_approx }
- `HardwareRecommendationResponse` { cpu_recommendations, gpu_recommendations, model_info, total_recommendations }

### 8.3 Usecase Router Schemas

**Request Schemas:**
- `UsecaseCreate` { unique_id, name, industry, description, batch_size, beam_size, input_tokens_min/max, output_tokens_min/max, ttft_min/max, e2e_min/max, inter_token_min/max, tags }
- `UsecaseUpdate` { (all fields optional) }
- `RecommendationRequest` { batch_sizes, model_categories, precision, include_pricing }

**Response Schemas:**
- `UsecaseResponse` { id, unique_id, name, industry, description, batch_size, beam_size, input/output_tokens_min/max, ttft/e2e/inter_token_min/max, tags, source, created_at, updated_at, is_active, latency_profile, input_length_profile }
- `RecommendationsResponse` { usecase: UsecaseResponse, recommendations: List[CategoryRecommendation] }
- `CategoryRecommendation` { model_category, recommended_models: List[ModelRecommendation] }
- `ModelRecommendation` { model_id, parameter_count, model_type, attention_type, batch_configurations }
- `BatchConfiguration` { batch_size, memory_required_gb, meets_slo, estimated_ttft, estimated_e2e, hardware_options }

### 8.4 Usecase Optimization Schemas

- `OptimizationRequest` { batch_sizes, model_sizes, max_results, optimization_mode }
- `OptimizationResult` { model_id, model_size, hardware_type, num_nodes, parallelism, batch_size, achieved_ttft, achieved_e2e, required_ttft, required_e2e, meets_slo, cost_per_hour, cost_per_request, throughput, utilization, efficiency_score }
- `OptimizationResponse` { usecase, configurations, optimization_mode, summary }

### 8.5 Training Router Schemas

**Request Schemas:**
- `TrainingEstimateRequest` { model, method, batch_size, seq_length, precision, optimizer, gradient_checkpointing, lora_rank, lora_alpha, freeze_layers, deepspeed_stage, tensor_parallel, data_parallel, framework_overhead_percent }
- `ClusterRecommendRequest` { model, method, batch_size, seq_length, precision, optimizer, gradient_checkpointing, lora_rank, deepspeed_stage, prefer_cost, max_budget_per_hour, available_hardware, max_gpus }
- `CheckFitRequest` { model, method, batch_size, seq_length, precision, optimizer, gradient_checkpointing, lora_rank, deepspeed_stage, hardware, num_gpus }
- `TimeEstimateRequest` { model, dataset_tokens, batch_size, gradient_accumulation, epochs, hardware, num_gpus, seq_length, parallelism }

**Response Schemas:**
- `TrainingEstimateResponse` { model, method, precision, optimizer, batch_size, seq_length, memory_breakdown, total_params, trainable_params, trainable_percent, gradient_checkpointing, lora_rank, deepspeed_stage, fits_single_gpu_24gb/40gb/80gb, min_gpus_80gb }
- `ClusterRecommendResponse` { recommendations: List[ClusterRecommendationResponse], total_options }
- `CheckFitResponse` { fits, memory_per_gpu_gb, utilization_percent, parallelism, reason, min_gpus_required, estimated_cost_per_hour }
- `TimeEstimateResponse` { total_steps, tokens_per_second, estimated_hours, estimated_cost, hardware, num_gpus, parallelism, model_flops_utilization }

---

## 9. Key Observations and Gaps

### 9.1 BudSimulator Not Exposed

The `BudSimulator` class in `src/bud_sim.py` is fully implemented with 8 simulation types and GenZ `SimulationEngine` integration, but has **no API endpoints**. This is the single largest gap -- the core simulation capability is unreachable from the frontend.

### 9.2 Performance Estimation Disconnect

The `usecases.py` router uses a **heuristic-based** `_estimate_performance()` function (line 634) with hardcoded multipliers instead of calling GenZ's actual performance modeling. Only `usecases_optimization.py` uses the real GenZ-backed `find_best_hardware_for_usecase()`. This means the basic recommendations endpoint gives rough estimates while the optimization endpoint gives GenZ-accurate results.

### 9.3 Missing Simulation Features

GenZ already supports features that have no API exposure:
- Chunked prefill (`estimate_chunked_performance`)
- End-to-end performance combining prefill+decode (`estimate_end_to_end_performance`)
- Multi-configuration comparison (`compare_performance_configurations`)
- Pareto-optimal performance (`get_pareto_optimal_performance`)

### 9.4 No Power Modeling API

Hardware records store a `power` field (watts), and the hardware_schema has a `hardware_performance_metrics` view with `perf_per_watt`, but there are no API endpoints that utilize power data for modeling or optimization.

### 9.5 Training Module Is Well-Structured

The training router (`/api/simulator/*`) is the most modern and cleanest router. It uses proper Pydantic v2 `field_validator`, clean separation of concerns, and delegates entirely to the SDK's `TrainingMemoryCalculator`, `TrainingClusterSelector`, and `TrainingTimeEstimator`.

### 9.6 Dual Source Model Management

Models exist in two places: `MODEL_DICT` (GenZ static registry) and SQLite database. The `list_all_models` endpoint merges these with a `source` field ("model_dict", "database", "both"). This creates complexity but is well-handled. The `apply_model_dict_patch()` at startup bridges these via `DynamicModelCollection`.
