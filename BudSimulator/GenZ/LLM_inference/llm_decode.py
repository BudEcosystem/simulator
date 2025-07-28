from .utils import ModdelingOutput, get_inference_system, get_offload_system
from GenZ.unit import Unit
from GenZ.operators import *

from GenZ.analyse_model import *
import warnings
from GenZ.collective_times import *
from GenZ.utils.plot_rooflines import *
from GenZ.Models import create_full_decode_model
from math import ceil

unit = Unit()

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
                                parallelism_heirarchy=parallelism_heirarchy )

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
    if  per_chip_memory < total_memory_req/pipeline_parallel:
        if model_offload:
            system = get_offload_system(system=system, total_memory_req = total_memory_req/pipeline_parallel , debug=debug)
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
        
        # Model final decode step with grown KV cache (only if output is significant)
        if output_tokens > 10:
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
            
            # Conservative average: weight initial more heavily since most generation happens early
            decode_latency = initial_latency * 0.8 + final_latency * 0.2
        else:
            # For short outputs, just add small growth factor
            decode_latency = initial_latency * (1.0 + 0.1 * output_tokens / 10)
        
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

    ## 1000x because the latency is in milli seconds. thrpt is in Token/s
    # if pipeline_parallel > 1:
    #     micro_batch_latency = decode_latency
    #     ## If the N micro batches, then the total latency is (N-1)*stage latency + initial_latency
    #     ## We make the assumption that the pipeline is balanced and the latency is same for all stages
    #     total_latency = ((num_micro_batches-1) * (decode_latency / pipeline_parallel)) + micro_batch_latency
    #     thrpt = 1000 * batch_size / total_latency
    # else:
    thrpt = 1000 * batch_size / decode_latency  # Requests per second
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
