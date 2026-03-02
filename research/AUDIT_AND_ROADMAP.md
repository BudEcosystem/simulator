# BudSimulator Comprehensive Audit & Strategic Roadmap

**Date**: 2026-03-01
**Scope**: Full codebase audit (inference, training, core engine, hardware, API) + SOTA research + industry analysis
**Sources**: 5 parallel agents auditing code + web research across 40+ papers, production traces, and industry reports

---

## TABLE OF CONTENTS

1. [Critical Bugs (Fix Immediately)](#1-critical-bugs-fix-immediately)
2. [High-Priority Bugs & Accuracy Issues](#2-high-priority-bugs--accuracy-issues)
3. [Medium-Priority Improvements](#3-medium-priority-improvements)
4. [Hardware Config Fixes](#4-hardware-config-fixes)
5. [New Features - Quick Wins](#5-new-features---quick-wins-1-2-weeks-each)
6. [New Features - Core Differentiators](#6-new-features---core-differentiators-2-6-weeks-each)
7. [New Features - Strategic Moats](#7-new-features---strategic-moats-1-3-months-each)
8. [Competitive Analysis](#8-competitive-analysis)
9. [Implementation Priority Matrix](#9-implementation-priority-matrix)

---

## 1. CRITICAL BUGS (Fix Immediately)

These produce wrong results and must be fixed before any feature work.

### BUG-C1: GQA Weight Memory Over-Estimation in Simulator
**File**: `genz/serving/simulator.py:411`
**Impact**: For Llama 3.1 70B (8 KV heads vs 64 Q heads), K/V weight size overestimated by ~8x
**Fix**: `attn_params = (2 * hidden * n_heads * head_dim + 2 * hidden * num_kv_heads * head_dim) * layers`

### BUG-C2: Decode Latency Double-Count in Disaggregation
**File**: `genz/serving/disaggregation.py:49, 159`
**Impact**: `decode_moddeling()` already models KV growth for output_tokens; multiplying by output_tokens again overestimates decode time 2-10x
**Fix**: Either pass `output_tokens=1` to get per-token latency, or use returned latency directly without multiplication

### BUG-C3: Simulation Only Processes One Batch at a Time
**File**: `genz/serving/simulator.py:286-292`
**Impact**: PP>1 throughput underestimated by ~PP factor; even PP=1 underestimates due to missing scheduler-execution overlap
**Fix**: Model pipeline stages as separate resources; allow next batch formation during current batch execution

### BUG-C4: ZeRO-1 Not Supported in Basic Training Calculator
**File**: `training/calculator.py:95-96, 384-386`
**Impact**: `deepspeed_stage="zero1"` silently gives zero memory savings (should shard optimizer states)
**Fix**: Add `if deepspeed_stage in ('zero1', 'zero2', 'zero3')` for optimizer sharding

### BUG-C5: Inconsistent LoRA Params Between Basic and Advanced Calculators
**File**: `training/calculator.py:238-265` vs `training/advanced_calculator.py:437-450`
**Impact**: Advanced calculator underestimates LoRA trainable params by ~30-40% due to `// 2` integer division bug
**Fix**: Align advanced calculator with basic calculator's 7-target formula

### BUG-C6: GH200 Flops Value is 2x Too High
**File**: `hardware/configs.py:184`
**Impact**: `'Flops': 1979` is FP8 TFLOPS, not BF16. All GH200 perf estimates are 2x too optimistic
**Fix**: Change to `'Flops': 989`

### BUG-C7: `release_onchip_mem` Uses `max` Instead of `min`
**File**: `genz/system.py:177`
**Impact**: Released memory can exceed physical on-chip memory size
**Fix**: Change `max` to `min`

### BUG-C8: Decode Memory Check Missing Expert Parallelism Division
**File**: `genz/LLM_inference/llm_decode.py:77`
**Impact**: MoE models with EP>1 falsely trigger "params won't fit" errors during decode
**Fix**: Change to `per_chip_memory < total_memory_req/(pipeline_parallel * expert_parallel)`

### BUG-C9: Attention TP*EP Over-Sharding for MoE Models
**File**: `genz/Models/attention.py:11`
**Impact**: With TP=4, EP=8, attention heads divided by 32 instead of 4
**Fix**: Use `tp = parallelism_config.tensor_parallel` for attention (EP only in FFN layers)

### BUG-C10: Gradient Checkpointing Memory Uses sqrt(L) Instead of 2*sqrt(L)
**File**: `training/calculator.py:449-453`
**Impact**: Activation memory underestimated by ~2x with gradient checkpointing
**Fix**: Use `effective_layers = 2 * math.ceil(math.sqrt(L))`

---

## 2. HIGH-PRIORITY BUGS & ACCURACY ISSUES

### BUG-H1: Token Throughput Includes Input Tokens
**File**: `genz/serving/simulator.py:275`
**Impact**: Throughput inflated by input/output ratio (40x for RAG workloads)
**Fix**: Use `total_tokens_generated += req.tokens_generated` only

### BUG-H2: Power Model Only Tracks GPU Active Energy in Sim Loop
**File**: `genz/serving/simulator.py:278-281`
**Impact**: Total power underestimated by ~45% (missing DRAM, CPU, cooling, interconnect, standby)
**Fix**: Call `estimate_from_simulation_result()` after each batch with utilization metrics

### BUG-H3: Eviction Doesn't Notify Requests (Silent KV Cache Corruption)
**File**: `genz/serving/memory_model.py:155-190`
**Impact**: Evicted requests continue decoding with invalid KV cache
**Fix**: Return affected request IDs; scheduler moves them back to pending for re-prefill

### BUG-H4: First Output Token Requires Extra Decode Iteration
**File**: `genz/serving/batch_scheduler.py:168-170`
**Impact**: TTFT overestimated by one full decode iteration (50-200ms)
**Fix**: Generate first token during prefill completion

### BUG-H5: Pipeline Parallel Latency Completely Commented Out
**File**: `genz/LLM_inference/llm_prefill.py:99-107`, `llm_decode.py:163-170`
**Impact**: PP>1 missing pipeline bubble; latency/throughput estimates incorrect
**Fix**: Re-enable pipeline latency with 1F1B schedule model

### BUG-H6: Decode KV Cache Weighting is Arbitrary (80/20)
**File**: `genz/LLM_inference/llm_decode.py:129`
**Impact**: Average TPOT inaccurate for long outputs
**Fix**: Use `(initial_latency + final_latency) / 2` for linear KV growth

### BUG-H7: Reference Model Memory Uses Incorrect 0.8 Factor
**File**: `training/advanced_calculator.py:314`
**Impact**: DPO reference model memory underestimated by 20% (28 GB for 70B models)
**Fix**: `reference_memory = weight_memory` (eval mode = same weights, no gradients)

### BUG-H8: DPO Activation Memory Not Doubled for Chosen+Rejected
**File**: `training/advanced_calculator.py:298-306`
**Impact**: DPO activation memory underestimated by ~2x
**Fix**: Multiply activation memory by `stage_config.forward_multiplier`

### BUG-H9: Missing FP8 Precision in Basic Training Calculator
**File**: `training/calculator.py:37-45`
**Impact**: FP8 training memory estimates 2x too high (falls through to 2 bytes)
**Fix**: Add `'fp8': 1` to PRECISION_BYTES

### BUG-H10: TPUv5e Memory Bandwidth Wrong (820 vs 1600 GB/s)
**File**: `hardware/configs.py:651`
**Impact**: TPUv5e performance ~50% too pessimistic for memory-bound ops
**Fix**: Change to `'Memory_BW': 1600`

### BUG-H11: Missing H200 Hardware Configuration
**File**: `hardware/configs.py`
**Impact**: Cannot simulate H200 workloads; training code references it causing potential KeyError
**Fix**: Add H200 config (989 TFLOPS BF16, 141GB HBM3e, 4800 GB/s)

### BUG-H12: Missing Standalone B200 Configuration
**File**: `hardware/configs.py`
**Impact**: Cannot simulate B200 without Grace CPU
**Fix**: Add B200 config (2250 TFLOPS BF16, 192GB HBM3e, 8000 GB/s)

---

## 3. MEDIUM-PRIORITY IMPROVEMENTS

### Inference Simulation
| ID | Issue | File | Impact |
|----|-------|------|--------|
| M1 | Chunked prefill doesn't actually chunk (only affects token budget) | batch_scheduler.py:104-106 | Head-of-line blocking not modeled |
| M2 | Radix tree checks only first token per block | prefix_cache.py:57-65 | False positive prefix matches |
| M3 | Config optimizer truncates grid by iteration order | config_optimizer.py:288-308 | May miss optimal configs with high TP/PP |
| M4 | Cluster scaling model too optimistic (94% at 8 GPUs) | cluster.py:47-51 | Throughput projections inflated |
| M5 | Missing batch scheduling overhead (CUDA graph, block tables) | batch_scheduler.py:185-214 | Latency underestimated 2-40% |
| M6 | LRU eviction can partially evict a request's blocks | memory_model.py:173-180 | Invalid inference state |
| M7 | Spill doesn't update per-request block tracking | memory_model.py:192-221 | Memory accounting corruption |
| M8 | DRAM energy double-counted in estimate_from_simulation_result | power_model.py:206-218 | Slight DRAM energy overestimate |
| M9 | Prefill-decode interference not modeled in continuous batching | batch_scheduler.py:201-213 | Batch latency overestimated |

### Training Simulation
| ID | Issue | File | Impact |
|----|-------|------|--------|
| M10 | Gradient checkpointing recomputation FLOPs too low | training_modeling.py:1188-1196 | Training time underestimated 15-30% |
| M11 | Pipeline bubble uses flat 15% instead of 1F1B formula | distributed.py:337-339 | PP throughput 3-5x overestimated at high PP |
| M12 | Adafactor memory overestimated (4 bytes/param vs ~1) | optimizers.py:137-150 | 2-4x Adafactor memory overestimate |
| M13 | Sequence parallelism not modeled for activation memory | calculator.py:390-457 | 30-40% activation overestimate with TP>1 |
| M14 | DoRA extra magnitude vector memory not modeled | calculator.py:238 | 5-10% underestimate for small LoRA ranks |
| M15 | GaLore low-rank memory capped at 50% unnecessarily | optimizers.py:72-76 | Pessimistic for aggressive rank configs |
| M16 | Context parallel not used in memory calculations | distributed.py:289 | No activation savings for CP>1 |

### Core Engine
| ID | Issue | File | Impact |
|----|-------|------|--------|
| M17 | FP8 compute multiplier applied to A100 (no FP8 support) | system.py:17 | Overly optimistic FP8 on pre-Hopper |
| M18 | INT4 multiplier assumes W4A4 (0.25x) vs common W4A16 (0.5x) | system.py:10 | 2x overestimate for GPTQ/AWQ |
| M19 | FC/GEMM `get_num_ops` returns MACs not FLOPs (naming) | operators.py:30-31 | Maintenance/confusion risk |
| M20 | Groq LPU 80 TB/s SRAM bandwidth in HBM roofline framework | configs.py:1061 | Nonsensical roofline for Groq |

---

## 4. HARDWARE CONFIG FIXES

### Incorrect Values
| GPU | Field | Current | Correct | Source |
|-----|-------|---------|---------|--------|
| GH200 | Flops | 1979 | 989 | BF16 not FP8 |
| H100 SXM | Memory_BW | 3400 | 3350 | NVIDIA datasheet |
| B100 | Flops | 1750 | 1800 | NVIDIA spec |
| TPUv5e | Memory_BW | 820 | 1600 | Google spec |
| GH200 | Memory_BW | 4900 | 3350 (GPU HBM only) | Clarify GPU vs system |

### Missing Hardware (Add)
| GPU | Flops (BF16) | Memory | BW (GB/s) | Priority |
|-----|-------------|--------|-----------|----------|
| H200 | 989 | 141 GB HBM3e | 4800 | CRITICAL |
| B200 | 2250 | 192 GB HBM3e | 8000 | CRITICAL |
| H100 PCIe | 756 | 80 GB HBM3 | 2000 | HIGH |
| B300 | TBD | TBD | TBD | MEDIUM (when specs available) |
| MI325X | TBD | 288 GB HBM3e | 6000 | MEDIUM |
| Trainium2 | ~380 | 96 GB HBM | 1600 | MEDIUM |
| Gaudi3 | ~1835 | 128 GB HBM2e | 3700 | MEDIUM |

### Missing Models (Add)
| Model | Priority | Notes |
|-------|----------|-------|
| DeepSeek-V3/R1 | CRITICAL | MLA architecture, MoE, dominant models |
| Llama 3.3 70B | HIGH | Released Dec 2024, widely deployed |
| Qwen 2.5 series | HIGH | Major competitor to Llama |
| Llama 4 Behemoth | LOW | When released |

---

## 5. NEW FEATURES - QUICK WINS (1-2 weeks each)

### F1: What-If Playground (Interactive Scenario Simulator)
**Problem**: Teams can't answer "what if I switch to INT8?" without expensive hardware experiments
**Solution**: Web UI with sliders for model/hardware/precision/batch/parallelism; real-time roofline charts, latency breakdown, memory utilization update as params change; side-by-side comparison of up to 4 scenarios
**USP**: No existing tool provides this interactivity. Leverages existing React frontend + GenZ backend.
**Implementation**: Frontend work on existing React app + existing performance API
**Value**: Highest adoption driver. DevRel tool for cloud providers ($50K-$200K/year embedded).

### F2: Quantization Impact Analyzer
**Problem**: Proliferation of quantization options (FP16/BF16/INT8/INT4/GPTQ/AWQ/GGUF) confuses teams
**Solution**: Input a model, see side-by-side comparison of all quantization levels: memory, throughput, latency, estimated quality degradation. Map each to compatible hardware.
**USP**: Already modeled in GenZ engine. Just needs UI exposure.
**Implementation**: Frontend + precision multiplier data already exists
**Value**: Every team deploying LLMs needs this. Key free-tier adoption feature.

### F3: Migration Planner (Hardware Upgrade Advisor)
**Problem**: Hopper-to-Blackwell migration wave in 2026. Cannot predict performance/cost impact.
**Solution**: Input current deployment + target hardware. Simulate both, show delta. Recommend GPU count reduction.
**USP**: Timely. Addresses the massive 2026 migration cycle.
**Implementation**: Existing comparison capability + migration-specific UI
**Value**: Every enterprise upgrading GPUs. $10K-$50K per engagement.

### F4: Speculative Decoding Estimator
**Problem**: Speculative decoding gives 2-6x decode speedup but teams can't predict the gain for their model/hardware combo
**Solution**: Add draft model params (size, acceptance rate) to decode modeling. Calculate net speedup from verification overhead + token acceptance + draft throughput.
**USP**: No calculator handles this. EAGLE/Medusa widely deployed.
**Implementation**: Modify `llm_decode.py` + add parameters. Math is well-understood.
**Value**: Directly impacts latency predictions for modern deployments.

---

## 6. NEW FEATURES - CORE DIFFERENTIATORS (2-6 weeks each)

### F5: TCO Compass (Total Cost of Ownership Calculator)
**Problem**: Teams cannot compare build vs. buy costs. Hidden expenses (ML eng, monitoring, electricity) surprise organizations. Only 29% of executives can measure AI ROI.
**Solution**: Extend performance modeling with cost layers: hardware amortization, cloud hourly rates, egress fees, staffing, electricity, cooling. Input workload profile → output 1-3 year TCO comparison across self-hosted, cloud API, and hybrid.
**USP**: No existing tool combines performance simulation with full TCO analysis.
**Value**: Enterprise teams spending $100K-$10M/year on GPU infra. $5K-$50K/year subscription.

### F6: GPU Scout (Intelligent Hardware Recommender)
**Problem**: "Which GPU should I use?" is the #1 community question with no authoritative answer. 30-50% GPU budget waste.
**Solution**: Input model + workload + budget → simulate across all supported hardware → rank by cost-efficiency → recommend optimal GPU(s) with parallelism strategy. Include consumer (RTX 4090/5090), data center (A100/H100/H200/B200), and alternatives (Inferentia2, TPU).
**USP**: Goes beyond VRAM calculators by modeling actual throughput, latency, and cost-per-token.
**Value**: Cloud providers, GPU resellers, enterprises. Affiliate/referral revenue potential.

### F7: SLA Predictor (Latency and Throughput Guarantor)
**Problem**: Teams cannot predict SLA compliance before provisioning hardware
**Solution**: Input model + hardware + parallelism + SLA targets. Simulate under various load patterns (steady, bursty, diurnal). Report SLA compliance probability. Recommend minimum hardware for target SLA.
**USP**: Bridges simulation and SLA engineering. No existing tool does this.
**Value**: Enterprise SRE/Platform teams. $10K-$50K/year. Critical for $1M+ infrastructure decisions.

### F8: MLA + MoE Architecture Support
**Problem**: DeepSeek V3/R1 use MLA (28x KV cache reduction); all frontier models are MoE. Current simulator cannot model either correctly.
**Solution**: Add `attention_type` (mha/gqa/mla) with MLA latent dim. Add MoE params (num_experts, active_experts, expert_parallelism, routing overhead). Model total_memory = all experts, active_compute = active_experts only.
**USP**: First simulator to handle MLA + MoE correctly.
**Value**: Critical for anyone working with DeepSeek, Llama 4, Mixtral, Qwen3-MoE.

### F9: Disaggregated Serving Simulator
**Problem**: Disaggregated inference (separate prefill/decode pools) is the default production architecture. Current simulator only models co-located.
**Solution**: Add `serving_mode`: colocated vs disaggregated. Model separate GPU pools with independent parallelism. Add KV cache transfer latency. Support heterogeneous hardware (H100 prefill + H200 decode).
**USP**: Only Frontier (unreleased) and Splitwise-sim model this. First open tool.
**Value**: High -- unlocks accurate modeling of all major production deployments.

### F10: Green Compute Dashboard (Energy + Carbon Estimator)
**Problem**: EU AI Act (Aug 2025) requires energy reporting for GPAI providers. No simulation tool estimates energy from model/hardware specs.
**Solution**: Extend existing power model to full reporting. Energy per inference request + per training run. Carbon footprint by grid region. Compliance-ready reports for EU AI Act.
**USP**: Regulatory compliance is mandatory, not optional. First sim tool with built-in energy/carbon reporting.
**Value**: Every EU-serving AI company needs this. $10K-$50K/year. Regulatory tailwind.

---

## 7. NEW FEATURES - STRATEGIC MOATS (1-3 months each)

### F11: Agent Capacity Planner
**Problem**: Agentic AI multiplies token consumption 20-30x. 50-100B agents projected in 2026. Only 11% of orgs have agentic AI in production. No tools help plan infrastructure.
**Solution**: Model agentic workloads: multi-turn conversations, tool calling, reasoning traces, parallel agent execution. Input agent architecture → estimate persistent KV cache, concurrent inference capacity, cost per agent-task.
**USP**: First tool for agentic AI infrastructure planning. Massive differentiator.
**Value**: Every company deploying agentic AI. $20K-$100K/year enterprise. TAM growing exponentially.

### F12: Model Routing Simulator
**Problem**: Routing 70% of queries to cheap models saves 63% costs. Teams can't simulate optimal routing without deployment.
**Solution**: Input multiple model configs + workload complexity distribution. Simulate routing ratios and cascade strategies. Output cost savings, quality impact, optimal routing.
**USP**: Connects simulation to emerging model routing/cascading paradigm.
**Value**: Any company running multiple models. $10K-$30K/year.

### F13: Cluster Architect (Multi-GPU Topology Optimizer)
**Problem**: Scaling from 1 GPU to clusters is where teams get burned. TP/PP/DP/EP/CP decisions are complex.
**Solution**: Input model + target throughput + available inventory. Simulate all viable parallelism configs including heterogeneous clusters. Recommend optimal topology.
**USP**: Supports heterogeneous clusters (mixed GPU types). Most tools don't.
**Value**: HPC teams and cloud architects managing $1M+ clusters. $20K-$100K/year.

### F14: Provider Price Tracker + Cost-Per-Token Comparator
**Problem**: GPU cloud pricing is opaque. Hidden fees inflate bills 20-40%.
**Solution**: Aggregate pricing from AWS/GCP/Azure/CoreWeave/Lambda/RunPod. Combine with BudSimulator performance estimates for actual cost-per-token across providers.
**USP**: Real cost-per-token, not just hourly rates. Performance-informed comparisons.
**Value**: Freemium drives traffic. Affiliate revenue. Premium enterprise tier.

### F15: Multi-Stage Pipeline Simulator (RAG + Inference)
**Problem**: RAG at scale requires sizing both retrieval and LLM layers. No tool does both.
**Solution**: Model end-to-end pipeline: embedding → retrieval → reranking → LLM. Simulate component interactions and end-to-end latency.
**USP**: First tool to size complete RAG stacks. Following MIST simulator's approach.
**Value**: Every RAG deployment. Growing TAM as RAG becomes standard.

---

## 8. COMPETITIVE ANALYSIS

| Capability | BudSim | Vidur | TokenSim | Frontier | LLMPerf | TRT-LLM | VRAM Calc |
|------------|--------|-------|----------|----------|---------|---------|-----------|
| Analytical simulation | **Yes** | Yes | Yes | Yes | No (real HW) | No (real HW) | No |
| Multi-vendor hardware | **Yes** | Limited | Limited | Limited | Yes | NVIDIA only | Limited |
| Web UI | **Yes** | No | No | No | No | No | Yes |
| Training estimation | **Yes** | No | No | No | No | No | No |
| Cost/TCO analysis | Planned | No | No | No | No | No | No |
| Hardware recommendation | Planned | No | No | No | No | No | No |
| MoE support | Partial | No | Yes | **Yes** | N/A | Yes | No |
| Disaggregated serving | Partial | No | No | **Yes** | N/A | Yes | No |
| Energy/carbon | **Yes** | No | No | No | No | No | No |
| Speculative decoding | No | No | Yes | No | N/A | Yes | No |
| LoRA/QLoRA modeling | **Yes** | No | No | No | No | No | No |
| MLA support | No | No | No | No | N/A | Yes | No |

**BudSimulator's unique moat**: Only tool combining analytical simulation + multi-vendor hardware + web UI + training estimation + energy modeling. Adding TCO/cost layer makes it unmatched.

---

## 9. IMPLEMENTATION PRIORITY MATRIX

### Phase 1: Foundation Fixes (Week 1-2)
**Goal**: Fix all critical bugs, update hardware configs

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| P0 | Fix 10 critical bugs (C1-C10) | 3 days | Correctness |
| P0 | Fix 12 high bugs (H1-H12) | 3 days | Accuracy |
| P0 | Hardware config fixes (GH200, H100 BW, B100, TPUv5e) | 1 day | Accuracy |
| P0 | Add H200, B200, H100 PCIe configs | 1 day | Coverage |
| P0 | Add DeepSeek-V3/R1, Llama 3.3 model definitions | 1 day | Coverage |

### Phase 2: Quick Win Features (Week 3-4)
**Goal**: Ship high-visibility features that leverage existing engine

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| P1 | F1: What-If Playground | 1 week | Adoption |
| P1 | F2: Quantization Impact Analyzer | 3 days | Adoption |
| P1 | F3: Migration Planner | 3 days | Revenue |
| P1 | F4: Speculative Decoding Estimator | 1 week | Accuracy |

### Phase 3: Core Differentiators (Week 5-10)
**Goal**: Build the features that make BudSimulator unmatched

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| P1 | F8: MLA + MoE Architecture Support | 2 weeks | Critical gap |
| P1 | F5: TCO Compass | 2 weeks | Revenue |
| P1 | F6: GPU Scout | 2 weeks | Adoption |
| P2 | F9: Disaggregated Serving | 2 weeks | Accuracy |
| P2 | F7: SLA Predictor | 1 week | Enterprise |
| P2 | F10: Green Compute Dashboard | 1 week | Compliance |

### Phase 4: Strategic Moats (Month 3-6)
**Goal**: Build defensible competitive advantages

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| P2 | F11: Agent Capacity Planner | 4 weeks | USP |
| P2 | F13: Cluster Architect | 3 weeks | Enterprise |
| P3 | F12: Model Routing Simulator | 2 weeks | Cost optimization |
| P3 | F14: Provider Price Tracker | 4 weeks | Revenue |
| P3 | F15: Multi-Stage Pipeline Simulator | 4 weeks | Full-stack |

### Phase 5: Medium Fixes (Ongoing)
**Goal**: Incrementally improve accuracy

| Priority | Items | Notes |
|----------|-------|-------|
| P2 | M1-M9: Inference medium fixes | Chunked prefill, eviction, scheduling overhead |
| P2 | M10-M16: Training medium fixes | Checkpointing, pipeline bubble, SP, DoRA |
| P3 | M17-M20: Core engine fixes | FP8/INT4 multipliers, Groq modeling |

---

## KEY MARKET SIGNALS SUPPORTING THIS ROADMAP

1. **$690B AI capex in 2026** → massive demand for infrastructure planning tools
2. **42% project abandonment rate** → TCO/ROI tools desperately needed
3. **20-30x token multiplication from agentic AI** → new infrastructure planning challenges
4. **EU AI Act energy reporting (Aug 2025)** → regulatory demand for energy estimation
5. **Hopper-to-Blackwell migration wave** → migration planning tools valuable now
6. **Heterogeneous GPU clusters becoming standard** → multi-hardware comparison essential
7. **Only 29% of executives can measure AI ROI** → TCO tools have massive market
8. **95% of enterprise GenAI implementations fail** → better planning tools = direct ROI
9. **MoE + MLA now dominant architecture** → simulator must model these correctly
10. **Disaggregated serving is production default** → must model to stay relevant

---

## TOTAL ISSUES FOUND

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| Inference/Serving | 3 | 6 | 9 | 3 | 21 |
| Training | 3 | 4 | 7 | 3 | 17 |
| Core Engine/Hardware | 4 | 5 | 4 | 6 | 19 |
| **Total** | **10** | **15** | **20** | **12** | **57** |

## NEW FEATURES PROPOSED: 15

| Category | Count |
|----------|-------|
| Quick Wins | 4 |
| Core Differentiators | 6 |
| Strategic Moats | 5 |
