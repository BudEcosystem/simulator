from .utils import ModdelingOutput, apply_checkpoint_weight_precision, get_inference_system, get_offload_system
from llm_memory_calculator.genz.unit import Unit
from llm_memory_calculator.genz.operators import *

from llm_memory_calculator.genz.analyse_model import *
import warnings
from llm_memory_calculator.genz.collective_times import *
from llm_memory_calculator.genz.utils.plot_rooflines import *
from llm_memory_calculator.genz.Models import get_configs, create_full_prefill_model
from llm_memory_calculator.hardware.configs import (
    apply_inference_realism,
    resolve_inference_realism,
)
from math import ceil

unit = Unit()

def prefill_moddeling(model = 'BERT', batch_size = 1, input_tokens = 4096,
    system_name = 'A100_40GB_GPU', system_eff=1, bits='bf16', debug= False, model_profilling = False,
    tensor_parallel = 1, pipeline_parallel = 1, expert_parallel = 1,
    collective_strategy='GenZ', network_config=None,
    parallelism_heirarchy = "TP{1}_EP{1}_PP{1}",
    model_offload = False):

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

    # ACHIEVED EFFICIENCY + FIXED OVERHEAD are DECLARED BY THE HARDWARE, not assumed here — see the
    # matching block in llm_decode.py. `system_eff` defaulted to 1, i.e. 100% of datasheet FLOPs and
    # 100% of datasheet bandwidth with zero launch/scheduler cost; the device record now declares
    # what it actually sustains, and an explicit caller-supplied `system_eff` (!= 1) still wins.
    # `None` here means "nothing declared" -> get_inference_system's documented per-technology band.
    _realism = resolve_inference_realism(system_name, phase='prefill')
    _ceff = system_eff if system_eff != 1 else _realism['compute_efficiency']
    _meff = system_eff if system_eff != 1 else _realism['memory_efficiency']
    system = get_inference_system(system_name = system_name, bits = bits,
                                ceff=1 if _ceff is None else _ceff,
                                meff=1 if _meff is None else _meff, network_config=network_config,
                                collective_strategy=collective_strategy, parallelism_heirarchy=parallelism_heirarchy, phase='prefill')
    apply_inference_realism(system, _realism, compute_efficiency=_ceff, memory_efficiency=_meff)
    apply_checkpoint_weight_precision(system, model)
    ##################################################################################################
    ### Model Characterization Calculation
    ##################################################################################################
    model_prefill = create_full_prefill_model(  name=model,
                                                input_sequence_length=input_tokens,
                                                tensor_parallel=tensor_parallel,
                                                pipeline_parallel=pipeline_parallel,
                                                expert_parallel=expert_parallel)


    model_df = get_model_df(model_prefill, system=system, batch_size = ub, intermediate_on_chip=True , model_characterstics = True)
    summary_table = get_summary_table(model_df, unit, model_characterstics = True)

    model_weights = summary_table[f'Total Weights ({unit.unit_mem})'].values[0]        ## In MB
    kv_cache = summary_table[f'KV Cache ({unit.unit_mem})'].values[0]                  ## In MB

    total_memory_req = model_weights + kv_cache
    num_nodes = pipeline_parallel * tensor_parallel * expert_parallel

    #################################################################################
    ### Offloading calculations
    #################################################################################
    is_offloaded = False
    per_chip_memory = system.get_off_chip_mem_size()   ## MB
    # Phase 6 Fix: Divide by both PP and EP for MoE models
    # Each GPU only holds 1/PP of layers and 1/EP of experts
    memory_parallelism = pipeline_parallel * expert_parallel
    if  per_chip_memory  < total_memory_req/memory_parallelism:
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
    ### Prefill generation time
    ##################################################################################################
    # model_prefill = create_full_prefill_model(  name=model,
    #                                             input_sequence_length=input_tokens,
    #                                             tensor_parallel=tensor_parallel,
    #                                             pipeline_parallel=pipeline_parallel,
    #                                             expert_parallel=expert_parallel)
    system.parallelism_heirarchy = parallelism_heirarchy
    model_df = get_model_df(model_prefill, system, unit, ub, intermediate_on_chip=True )
    summary_table = get_summary_table(model_df, unit)
    prefill_latency = summary_table[f'Latency ({unit.unit_time})'].values[0]                 # Latency in millisec

    if debug:
        display_df(simplify_df(model_df))
        display(summary_table)

    ##################################################################################################
    ### Final Latency and Thrpt Calculation
    ##################################################################################################

    # FIXED PER-STEP COST the throughput roofline omits entirely (see llm_decode.py for the full
    # argument and the measurements). Prefill is ONE engine step over the whole prompt, so it pays
    # the per-operator dispatch cost, the per-step host cost once, and the per-sequence host cost for
    # each of the ub requests in the microbatch — but no per-stream (c_stream) term, which is a
    # decode-only, layer-scaled legacy term. Prefill launches eagerly (prompt shapes vary per
    # request, so engines do not CUDA-graph it), which is why the declared per-op default is larger
    # for this phase than for decode.
    _t_launch = system.kernel_launch_latency_ms
    _step_oh = getattr(system, 'step_overhead_ms', 0.0)
    _seq_oh = getattr(system, 'per_sequence_overhead_ms', 0.0)
    if _t_launch or _step_oh or _seq_oh:
        prefill_latency += (_t_launch * count_repeat_aware_ops(model_df)
                            + _step_oh
                            + _seq_oh * ub)

    # Pipeline-parallel TTFT semantics: prefill_latency above is ONE microbatch (ub requests)
    # filling the whole pipeline (all L layers + inter-stage comm). The remaining m-1 microbatches
    # drain behind it, one slowest-stage time T_stage = prefill_latency * ceil(L/PP)/L apart.
    # Latency reports the batch makespan (worst request's TTFT); TTFT_first / TTFT_mean expose the
    # first and mean request TTFT. PP=1 (m=1) is byte-identical.
    ttft_first = prefill_latency
    ttft_mean = prefill_latency
    if pipeline_parallel > 1:
        m = ceil(batch_size / ub)
        num_layers = max(1, get_configs(model).num_decoder_layers)
        t_stage = prefill_latency * ceil(num_layers / pipeline_parallel) / num_layers
        prefill_latency = prefill_latency + (m - 1) * t_stage
        ttft_mean = ttft_first + (m - 1) / 2 * t_stage

    ## 1000x because the latency is in milli seconds. thrpt is in Token/s
    # M1: steady-state pipelined throughput is gated by conserved per-token work (inter-stage comm is
    # already in prefill_latency), not the one-shot fill/drain latency. PP adds bubble LATENCY and
    # memory capacity but does not reduce steady-state throughput; the prior /(2 - 1/PP) under-counted it.
    # With PP>1, prefill_latency is the batch makespan (fill + m-1 stage-steps).
    thrpt = 1000 * batch_size / prefill_latency  # Requests per second
    tokens_per_sec = thrpt * input_tokens  # Tokens per second

    attn_time = summary_table[f'Attn Latency ({unit.unit_time})'].values[0]
    linear_time = summary_table[f'Linear Latency ({unit.unit_time})'].values[0]
    total_communication_delay = summary_table[f'Comm Latency ({unit.unit_time})'].values[0]
    # runtime_breakdown = [linear_time, attn_time, total_communication_delay]
    runtime_breakdown = get_runtime_breakdown(model_df)
    ##################################################################################################
    ### Output Generation
    ##################################################################################################

    return ModdelingOutput(
                        Latency=prefill_latency,
                        Throughput=thrpt,
                        Throughput_tokens_per_sec=tokens_per_sec,
                        Runtime_breakdown=runtime_breakdown,
                        is_offload=is_offloaded,
                        model_df = model_df,
                        summary_table = summary_table,
                        TTFT_first=ttft_first,
                        TTFT_mean=ttft_mean,
                )
