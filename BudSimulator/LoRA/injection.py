"""LoRA injection logic for GenZ simulator."""

from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from BudSimulator.GenZ.Models.default_models import ModelConfig
    from BudSimulator.GenZ.parallelism import ParallelismConfig

# Import at runtime to avoid circular imports
def _get_op_types():
    from BudSimulator.GenZ.Models.utils import OpType, ResidencyInfo
    return OpType, ResidencyInfo


def inject_lora_merge(
    base_gemm: List,
    model_config: 'ModelConfig',
    parallelism_config: 'ParallelismConfig',
    layer_name: str
) -> List[List]:
    """Inject LoRA merge operation for merge strategy.
    
    For merge strategy, we add a LORA_MERGE operation before the base GEMM
    to simulate the one-time cost of merging adapters.
    """
    if not model_config.lora_config or not model_config.lora_config.enabled:
        return [base_gemm]
    
    if model_config.lora_config.strategy != 'merge':
        return [base_gemm]
    
    # Extract dimensions from base GEMM
    name, m, n, k, h, z1, z2, op_type = base_gemm
    
    # Create LORA_MERGE operation
    # The merge operation has cost proportional to the adapter size
    OpType, _ = _get_op_types()
    rank = model_config.lora_config.rank
    merge_op = [
        f"{name}_lora_merge",
        m,  # batch dimension
        n,  # output dimension
        rank,  # rank dimension (merge cost)
        h,  # same as base
        z1,  # same residency
        z2,  # same residency
        OpType.LORA_MERGE
    ]
    
    return [merge_op, base_gemm]


def inject_lora_dynamic(
    base_gemm: List,
    model_config: 'ModelConfig',
    parallelism_config: 'ParallelismConfig',
    layer_name: str,
    sequence_length: int
) -> List[List]:
    """Inject LoRA dynamic operations for dynamic strategy.
    
    For dynamic strategy, we replace the base GEMM with:
    1. Base GEMM (unchanged)
    2. GEMM_LORA_A: input @ A (down projection)
    3. GEMM_LORA_B: lora_a_output @ B (up projection)
    4. ADD: base_output + lora_output
    """
    if not model_config.lora_config or not model_config.lora_config.enabled:
        return [base_gemm]
    
    if model_config.lora_config.strategy != 'dynamic':
        return [base_gemm]
    
    # Extract dimensions from base GEMM
    name, m, n, k, h, z1, z2, op_type = base_gemm
    rank = model_config.lora_config.rank
    
    # Adjust dimensions for parallelism
    tp = parallelism_config.tensor_parallel
    
    # Get OpType and ResidencyInfo
    OpType, ResidencyInfo = _get_op_types()
    
    # Create operation sequence
    ops = []
    
    # 1. Base GEMM (unchanged)
    ops.append(base_gemm)
    
    # 2. GEMM_LORA_A: [m, k] @ [k, rank] = [m, rank]
    lora_a_op = [
        f"{name}_lora_a",
        m,  # batch/sequence dimension
        rank,  # output dimension (rank)
        k,  # input dimension
        h,  # same as base
        z1,  # same residency
        ResidencyInfo.All_offchip,  # LoRA weights typically offchip
        OpType.GEMM_LORA_A
    ]
    ops.append(lora_a_op)
    
    # 3. GEMM_LORA_B: [m, rank] @ [rank, n] = [m, n]
    lora_b_op = [
        f"{name}_lora_b",
        m,  # batch/sequence dimension
        n,  # output dimension (same as base)
        rank,  # input dimension (rank)
        h,  # same as base
        z1,  # same residency
        ResidencyInfo.All_offchip,  # LoRA weights typically offchip
        OpType.GEMM_LORA_B
    ]
    ops.append(lora_b_op)
    
    # 4. ADD: base_output + lora_output
    add_op = [
        f"{name}_add",
        m,  # batch/sequence dimension
        n,  # feature dimension
        1,  # element-wise operation
        h,  # same as base
        z1,  # same residency
        z2,  # same residency
        OpType.ADD
    ]
    ops.append(add_op)
    
    return ops


def should_inject_lora(layer_name: str, target_modules: List[str]) -> bool:
    """Check if LoRA should be injected for this layer based on target modules."""
    layer_name_lower = layer_name.lower()
    
    # Map common operation names to module types
    op_to_module = {
        'qkv': 'attn',
        'out proj': 'attn',
        'query': 'attn',
        'key': 'attn',
        'value': 'attn',
        'up+gate': 'ffn',
        'down': 'ffn',
        'up': 'ffn',
        'gate': 'ffn',
        'w1': 'ffn',
        'w2': 'ffn',
        'w3': 'ffn',
    }
    
    # Check if the operation name matches any known patterns
    for op_pattern, module_type in op_to_module.items():
        if op_pattern in layer_name_lower and module_type in target_modules:
            return True
    
    # Also check direct matches
    for target in target_modules:
        target_lower = target.lower()
        if target_lower in layer_name_lower:
            return True
    
    return False


def inject_lora_ops(
    operations: List[List],
    model_config: 'ModelConfig',
    parallelism_config: 'ParallelismConfig',
    sequence_length: int
) -> List[List]:
    """Inject LoRA operations into a list of operations.
    
    This function processes the operation list and injects LoRA operations
    where appropriate based on the configuration.
    """
    if not model_config.lora_config or not model_config.lora_config.enabled:
        return operations
    
    # Get OpType for comparison
    OpType, _ = _get_op_types()
    
    result = []
    
    for op in operations:
        # Check if this is a GEMM operation
        if len(op) >= 8 and op[7] == OpType.GEMM:
            op_name = op[0]
            
            # Check if we should inject LoRA for this operation
            if should_inject_lora(op_name, model_config.lora_config.target_modules):
                if model_config.lora_config.strategy == 'merge':
                    # Inject merge operation
                    result.extend(inject_lora_merge(op, model_config, parallelism_config, op_name))
                elif model_config.lora_config.strategy == 'dynamic':
                    # Inject dynamic operations
                    result.extend(inject_lora_dynamic(op, model_config, parallelism_config, op_name, sequence_length))
                else:
                    # Unknown strategy, keep original
                    result.append(op)
            else:
                # Not a target module, keep original
                result.append(op)
        else:
            # Not a GEMM operation, keep original
            result.append(op)
    
    return result 