from .utils import ModdelingOutput, get_inference_system, get_offload_system
from llm_memory_calculator.genz.unit import Unit
from llm_memory_calculator.genz.operators import *

from llm_memory_calculator.genz.analyse_model import *
import warnings
from llm_memory_calculator.genz.collective_times import *
from llm_memory_calculator.genz.utils.plot_rooflines import *
from llm_memory_calculator.genz.Models import create_full_decode_model
from math import ceil

unit = Unit()

# CPU decode memory-bandwidth floor (category calibration, generalizable).
# Decode is weight-streaming bound: every token reads the resident matmul weights (+ batch KV) from
# DRAM, so TPOT cannot beat resident_bytes/(eta*BW). The operator roofline sums per-op memory time but
# lands optimistic on CPU, so we clamp decode latency to this floor. `eta` is the sustained DECODE
# memory efficiency RELATIVE TO THE MODEL'S OWN resident-byte accounting (summary_table weights+KV),
# fit from one measured device per category, then reused across that category (ISA class x vendor).
# It is lower than the STREAM band (~0.80) because (a) GEMV weight-streaming interleaved with compute
# sustains less than pure STREAM and (b) it absorbs the residual roofline optimism, self-consistently.
# A measured per-device block (system.decode_mem_efficiency) overrides it.
# One category constant: sustained decode memory efficiency vs the model's own resident-byte
# accounting (weights + KV), against the device's SOURCED datasheet peak bandwidth. Joint-fit from a
# measured x86-AMX device (Intel Xeon 6767P / Qwen3-8B, DDR5-6400) across BOTH socket counts:
#   1 socket  (409.6 GB/s): 11.0 GiB / (eta * BW) = 102 ms  -> eta = 0.257
#   2 sockets (819  GB/s): 11.0 GiB / (eta * BW) =  60 ms  -> eta = 0.219
#   joint (geomean)                                          -> eta = 0.237
# ~0.3 of the STREAM band (~0.80): GEMV weight-streaming interleaved with compute sustains less. Because
# it is defined against the SOURCED peak BW (not a projected number), it is a true efficiency that
# transfers across the (ISA x vendor) category. The residual TP=1(+8%)/TP=2(-8%) split is the NUMA
# scaling loss (real 2-socket is ~1.7x, not 2x) -- a separate interconnect term, not this constant.
# NOTE: a separate higher eta for KV-streaming was tried and OVER-corrected, so one blended constant
# generalizes better. A measured per-device block (system.decode_mem_efficiency) overrides it.
_ETA_MEM_DECODE_CPU_X86 = 0.237
_X86_CPU_VENDORS = ('intel', 'amd')

# NUMA / tensor-parallel interconnect latency for CPU decode. Megatron TP does 2 all-reduces per layer
# (post-attention, post-FFN). GenZ's collective models the all-reduce DATA VOLUME (a few KB/token ->
# negligible), but on CPU the cost is dominated by the per-all-reduce LATENCY of many small blocking
# gloo/UPI reductions, which GenZ omits. Adding it makes 2-socket decode scale ~1.7x (measured), not the
# ideal 2x. Category constant (x86 2-socket UPI + gloo), fit so Xeon 6767P batch-1 TP=2 decode = 60 ms:
# 55.4 ms floor + 2*36 all-reduces * L = 60 ms  ->  L ~= 0.064 ms/all-reduce.
_CPU_TP_ALLREDUCE_LAT_MS = 0.064


def _cpu_decode_mem_efficiency(system, system_name):
    """Sustained decode memory efficiency for a CPU, or None if not a CPU.
    Per-device override (system.decode_mem_efficiency) beats the category default."""
    is_cpu = isinstance(system_name, dict) and str(system_name.get('type', '')).lower() == 'cpu'
    if not is_cpu:
        return None
    return float(getattr(system, 'decode_mem_efficiency', None) or _ETA_MEM_DECODE_CPU_X86)

def decode_moddeling(model = 'BERT', batch_size = 1, input_tokens = 4096,
    output_tokens = 0,   Bb = 4 ,           ## Only for Decode
    system_name = 'A100_40GB_GPU', system_eff = 1, bits='bf16', debug= False, model_profilling = False,
    tensor_parallel = 1, pipeline_parallel = 1,
    expert_parallel = 1,
    collective_strategy='GenZ', network_config=None,
    parallelism_heirarchy = "TP{1}_EP{1}_PP{1}",
    model_offload = False, ceff = None, meff = None,
    model_characterstics = False):

    if pipeline_parallel > 1:
        ub = max(batch_size // pipeline_parallel, 1)
        num_micro_batches = batch_size // ub
        if batch_size < pipeline_parallel:
            warnings.warn(f"Batch size is divided into micro batches for pipeline parallel, micro batch size:{ub}, consider increasing batch size")
    else:
        ub = batch_size

    ##################################################################################################
    ### System Declaration
    ##################################################################################################

    system = get_inference_system(system_name = system_name, bits = bits, ceff=system_eff , meff=system_eff,
                                network_config=network_config,
                                collective_strategy=collective_strategy,
                                parallelism_heirarchy=parallelism_heirarchy, phase='decode' )

    ##################################################################################################
    ### Model Characterization Calculation
    ##################################################################################################
    # if is_moe:
    model_decode = create_full_decode_model(name=model,
                                            input_sequence_length=input_tokens,
                                            output_gen_tokens = output_tokens ,
                                            tensor_parallel=tensor_parallel,
                                            pipeline_parallel=pipeline_parallel,
                                            expert_parallel=expert_parallel)

    # Get model dataframe, which will include detailed metrics if model_characterstics is True
    model_df = get_model_df(model_decode, system=system, batch_size= ub*Bb, intermediate_on_chip=True , beam_merge= (Bb > 1), beam_size= Bb, model_characterstics=model_characterstics)
    
    # The summary table's columns will differ based on the flag, but latency is always in model_df
    summary_table = get_summary_table(model_df, unit, model_characterstics=model_characterstics)
    
    # Get latency directly from the detailed model_df, which always contains it
    decode_latency = model_df[f'Latency ({unit.unit_time})'].sum()

    # Get memory requirements
    if f'Total Weights ({unit.unit_mem})' in summary_table.columns:
        model_weights = summary_table[f'Total Weights ({unit.unit_mem})'].values[0]
        kv_cache = summary_table[f'KV Cache ({unit.unit_mem})'].values[0]
        total_memory_req = model_weights + kv_cache
    else:
        # Fallback for characterization mode
        total_memory_req = model_df[f'Total Data ({unit.unit_mem})'].sum()
        
    num_nodes = pipeline_parallel * tensor_parallel * expert_parallel

    #################################################################################
    ### Offloading calculations
    #################################################################################
    is_offloaded = False
    per_chip_memory = system.get_off_chip_mem_size()   ## MB
    memory_parallelism = pipeline_parallel * expert_parallel
    if  per_chip_memory < total_memory_req/memory_parallelism:
        if model_offload:
            system = get_offload_system(system=system, total_memory_req = total_memory_req/memory_parallelism , debug=debug)
            warnings.warn(f"Some Parameter offloaded, effective Memory BW:{unit.raw_to_unit(system.offchip_mem_bw, type='BW')} ")
            is_offloaded = True
        elif model_profilling:
            warnings.warn(f"All params would not fit on chip. System Memory Cap:{per_chip_memory/1024} GB , Weights : {model_weights/1024} GB, KV Cache:{kv_cache/1024} ")
        else:
            raise ValueError(f"All params would not fit on chip. System Memory Cap:{per_chip_memory/1024} GB , Weights : {model_weights/1024} GB, KV Cache:{kv_cache/1024}. \n System:{system_name}")

    ## for tensor shareding per layer.
    assert pipeline_parallel >= 1, "Pipeline parallel must be >= 1"
    assert tensor_parallel >= 1, f"Tensor parallel must be >= 1, {tensor_parallel}"

    if model_profilling:
        return model_df, summary_table

    ##################################################################################################
    ### Token generation time with KV cache growth modeling
    ##################################################################################################
    
    # Model decode with growing KV cache to get accurate TPOT
    if output_tokens > 1:
        # For longer outputs, use simple two-point sampling to avoid expensive computation
        # Model initial decode step with current KV cache
        model_decode_initial = create_full_decode_model(name=model,
                                                      input_sequence_length=input_tokens,
                                                      output_gen_tokens=1,
                                                      tensor_parallel=tensor_parallel,
                                                      pipeline_parallel=pipeline_parallel,
                                                      expert_parallel=expert_parallel)
        
        initial_df = get_model_df(model_decode_initial, system, unit, ub*Bb,
                                intermediate_on_chip=True, beam_merge=(Bb > 1), beam_size=Bb, model_characterstics=model_characterstics)
        initial_summary = get_summary_table(initial_df, unit, model_characterstics=model_characterstics)
        initial_latency = initial_summary[f'Latency ({unit.unit_time})'].values[0]
        
        # Model the final decode step with the grown KV cache (context = input + output_tokens) and take
        # the 50/50 average — the expected per-token latency over uniform KV growth. M2 fix: this now
        # applies for ALL output_tokens > 1. The previous `output_tokens > 10` gate fell back to a fixed
        # `initial*(1 + 0.1*output_tokens/10)` bump that ignored the actual context length (a 5-token
        # generation got +5% regardless of whether the prompt was 128 or 128k). The two-point average is
        # grounded in the real KV-cache growth; the extra model build for short outputs is cheap.
        model_decode_final = create_full_decode_model(name=model,
                                                    input_sequence_length=input_tokens + output_tokens,
                                                    output_gen_tokens=1,
                                                    tensor_parallel=tensor_parallel,
                                                    pipeline_parallel=pipeline_parallel,
                                                    expert_parallel=expert_parallel)

        final_df = get_model_df(model_decode_final, system, unit, ub*Bb,
                              intermediate_on_chip=True, beam_merge=(Bb > 1), beam_size=Bb, model_characterstics=model_characterstics)
        final_summary = get_summary_table(final_df, unit, model_characterstics=model_characterstics)
        final_latency = final_summary[f'Latency ({unit.unit_time})'].values[0]

        # 50/50 average for expected value over uniform KV cache growth
        decode_latency = (initial_latency + final_latency) / 2.0
        
        # Use the initial model and summary for debugging output
        model_df = initial_df
        summary_table = initial_summary
    else:
        # Single token decode - use original logic
        model_decode = create_full_decode_model(name=model,
                                                input_sequence_length=input_tokens,
                                                output_gen_tokens=output_tokens,
                                                tensor_parallel=tensor_parallel,
                                                pipeline_parallel=pipeline_parallel,
                                                expert_parallel=expert_parallel)

        model_df = get_model_df(model_decode, system, unit, ub*Bb,  
                              intermediate_on_chip=True, beam_merge=(Bb > 1), beam_size=Bb, model_characterstics=model_characterstics)
        summary_table = get_summary_table(model_df, unit, model_characterstics=model_characterstics)
        decode_latency = summary_table[f'Latency ({unit.unit_time})'].values[0]

    if debug:
        display_df(simplify_df(model_df))
        display(summary_table)
        if output_tokens > 1:
            print(f"Modeled KV cache growth for {output_tokens} output tokens")
            print(f"TPOT: {decode_latency:.2f} ms")

    ##################################################################################################
    ### Final Latency and Thrpt Calculation
    ##################################################################################################

    # Per-hardware inference calibration (default no-op): add the fixed per-step kernel-launch cost
    # (N_ops · t_launch) and the per-stream runtime overhead (c_stream · layers · (B-1)) that the
    # throughput roofline omits. Guarded so the default path (both constants 0) is byte-identical.
    if system.kernel_launch_latency_ms or system.per_stream_overhead_ms:
        n_ops = count_repeat_aware_ops(model_df)
        _repeats = [model_df.loc[i, 'Dimension'] for i in range(len(model_df))
                    if model_df.loc[i, 'Op Type'] == 'Repeat']
        n_layers = max(_repeats) if _repeats else 1
        decode_latency += (system.kernel_launch_latency_ms * n_ops
                           + system.per_stream_overhead_ms * n_layers * max(0, batch_size - 1))

    # CPU decode memory-bandwidth floor: a token cannot be produced faster than the resident weights
    # (+ batch KV) can stream from DRAM. The operator roofline under-counts this traffic on CPU, so
    # clamp to the physical floor  (weight+KV bytes / tensor_parallel) / (eta_decode * offchip_BW).
    # Generalizable: uses the device's own bytes + bandwidth with a category efficiency (see
    # _cpu_decode_mem_efficiency). No-op on GPU / when the model already predicts above the floor.
    _eta_dec = _cpu_decode_mem_efficiency(system, system_name)
    if _eta_dec and not is_offloaded:
        # summary_table weights+KV are ALREADY per-rank (GenZ shards at tensor_parallel), streamed at the
        # per-rank (per-socket) bandwidth -> do NOT divide by tensor_parallel again.
        _bytes = total_memory_req * 1024 * 1024  # MiB -> bytes, per rank
        _floor_ms = _bytes / (_eta_dec * system.offchip_mem_bw) * 1000.0
        if _floor_ms > decode_latency:
            decode_latency = _floor_ms
        # NUMA/TP all-reduce latency (see _CPU_TP_ALLREDUCE_LAT_MS): 2 reductions/layer, added on top of
        # the memory floor because comm and weight-streaming are sequential phases of a decode step.
        if tensor_parallel > 1:
            _reps = [model_df.loc[i, 'Dimension'] for i in range(len(model_df))
                     if model_df.loc[i, 'Op Type'] == 'Repeat']
            _n_layers = max(_reps) if _reps else 1
            decode_latency += _CPU_TP_ALLREDUCE_LAT_MS * 2 * _n_layers * (tensor_parallel - 1)

    ## 1000x because the latency is in milli seconds. thrpt is in Token/s
    # M1: steady-state pipelined throughput is gated by the per-token work, which is CONSERVED across
    # pipeline stages (inter-stage comm is already in decode_latency). Pipeline parallelism raises
    # memory capacity and adds fill/drain BUBBLE LATENCY, but does not reduce steady-state token
    # throughput. The prior formula divided throughput by the one-shot fill/drain latency
    # (~(2 - 1/PP)×), under-counting PP throughput ~1.5-2×. Fill/drain belongs to first-token latency.
    thrpt = 1000 * batch_size / decode_latency  # Requests per second (steady-state, PP-independent)
    # For decode, each request generates one token per iteration
    tokens_per_sec = thrpt  # Since decode generates 1 token per request per iteration


    linear_time = summary_table[f'Linear Latency ({unit.unit_time})'].values[0]                ## In milliseconds
    attn_time = summary_table[f'Attn Latency ({unit.unit_time})'].values[0]                    ## In milliseconds
    total_communication_delay = summary_table[f'Comm Latency ({unit.unit_time})'].values[0]    ## In milliseconds
    total_time = linear_time + attn_time + total_communication_delay
    # runtime_breakdown = [linear_time, attn_time, total_communication_delay]
    runtime_breakdown = get_runtime_breakdown(model_df)
    ##################################################################################################
    ### Output Generation
    ##################################################################################################

    return ModdelingOutput(
                        Latency=decode_latency,
                        Throughput=thrpt,
                        Throughput_tokens_per_sec=tokens_per_sec,
                        Runtime_breakdown=runtime_breakdown,
                        is_offload=is_offloaded,
                        model_df = model_df,
                        summary_table = summary_table,
                )
