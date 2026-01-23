"""
Training Operators for GenZ Framework.

Extends the base Operator class for training with backward pass operators.
These operators model the computational cost of backward passes (gradient computation)
and optimizer updates for training simulation.

Key concepts:
- Backward FLOPs are typically 2x forward FLOPs for each operation
- dL/dW (weight gradients): X^T @ dL/dY
- dL/dX (activation gradients): dL/dY @ W^T
- Optimizer updates have their own compute and memory patterns
"""

import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from .operator_base import Operator
from .operators import FC, GEMM, Logit, Attend, Sync
from .Models import OpType, CollectiveType
from .collective_times import (
    get_AR_time, get_AG_time, get_A2A_time,
    get_reduce_scatter_time, get_message_pass_time
)


# Extended op_type_dicts for training operators
training_op_type_dicts = {
    30: 'BackwardGEMM',
    31: 'BackwardFC',
    32: 'BackwardLogit',
    33: 'BackwardAttend',
    34: 'OptimizerUpdate',
    35: 'GradientAccumulate',
    36: 'GradientAllReduce',
    37: 'GradientReduceScatter',
    38: 'WeightAllGather',
    39: 'BackwardNorm',       # Phase 7: LayerNorm/RMSNorm backward
    40: 'BackwardEmbedding',  # Phase 7: Embedding backward
}


class TrainingOperator(Operator):
    """
    Base class extending Operator for training with backward pass support.

    Training operators compute both forward and backward pass times,
    and can model gradient computation, accumulation, and optimizer updates.
    """

    def __init__(self, dim, density=(1.0, 1.0, 1.0), trainable: bool = True):
        """
        Initialize training operator.

        Args:
            dim: Dimensions for the operation
            density: Sparsity factors for input, weight, output
            trainable: Whether this layer's weights are trained (affects backward pass)
        """
        self.trainable = trainable
        self._op_type = None  # Set by subclasses
        super().__init__(dim, density)

    def get_op_type(self, dim=None):
        """
        Get operator type, using _op_type if set.

        Training operators set _op_type directly rather than including it in dim,
        so we override this method to return the correct type.
        """
        if hasattr(self, '_op_type') and self._op_type is not None:
            return training_op_type_dicts.get(self._op_type, f'Backward{self._op_type}')
        return super().get_op_type(dim)

    def get_backward_num_ops(self) -> int:
        """
        Get number of FLOPs for backward pass.

        For most operations, backward pass requires:
        - dL/dW computation (same as forward)
        - dL/dX computation (same as forward)
        Total = 2x forward FLOPs
        """
        if not self.trainable:
            # Only compute dL/dX for non-trainable layers
            return self.get_num_ops()
        return 2 * self.get_num_ops()

    def get_backward_tensors(self) -> Tuple[Any, Any, Any]:
        """
        Get gradient tensors for backward pass.

        Returns:
            Tuple of (grad_input, grad_weight, grad_output) shapes
        """
        input_a, input_w, output = self.get_tensors()
        # Gradient shapes match original tensor shapes
        return output, input_w, input_a  # dL/dY, dW, dL/dX

    def get_training_num_ops(self) -> int:
        """Get total FLOPs for forward + backward."""
        return self.get_num_ops() + self.get_backward_num_ops()

    def get_training_roofline(self, system, unit) -> Dict[str, Any]:
        """
        Compute combined forward + backward roofline analysis.

        Args:
            system: System configuration
            unit: Unit conversion helper

        Returns:
            Dictionary with training roofline metrics
        """
        # Get forward roofline
        fwd = self.get_roofline(system, unit)

        # Compute backward-specific metrics
        # get_backward_num_ops already accounts for trainability (2x for trainable, 1x for frozen)
        # No additional multiplication needed - just convert to MACs (x2)
        bwd_flops = self.get_backward_num_ops()  # Already includes 2x for trainable
        bwd_num_ops = bwd_flops  # MACs = FLOPs for our purposes

        # Memory for backward pass:
        # - Read: activations (for dL/dX), weights (for dL/dW), grad_output
        # - Write: grad_input, grad_weight
        # For trainable layers: ~2x forward memory
        # For frozen layers: ~1.2x forward memory (only activation gradients)
        fwd_num_data = self.get_effective_num_data(system)
        if self.trainable:
            bwd_num_data = fwd_num_data * 2.0  # Read + write activations and weights
        else:
            bwd_num_data = fwd_num_data * 1.2  # Only activation gradients

        bwd_op_intensity = bwd_num_ops / bwd_num_data if bwd_num_data > 0 else 0

        # Compute time: FLOPs / (throughput * efficiency)
        bwd_compute_time = (bwd_num_ops * system.get_bit_multiplier(type='C') /
                           system.op_per_sec / system.compute_efficiency)

        # Memory time: bytes / bandwidth
        bwd_memory_time = (bwd_num_data / system.offchip_mem_bw / system.memory_efficiency)

        bwd_exec_time = max(bwd_compute_time, bwd_memory_time)
        bwd_boundedness = 'Compute' if bwd_compute_time > bwd_memory_time else 'Memory'

        # Combine forward and backward
        total_exec_time = fwd[f'Latency ({unit.unit_time})'] + unit.raw_to_unit(bwd_exec_time, type='T')
        total_num_ops = fwd[f'Num ops ({unit.unit_flop})'] + unit.raw_to_unit(bwd_num_ops, type='O')

        return {
            'Layer Name': self.name,
            'Op Type': self.get_op_type(self.dim),
            'Trainable': self.trainable,
            'Forward Bound': fwd['Bound'],
            'Backward Bound': bwd_boundedness,
            f'Forward Latency ({unit.unit_time})': fwd[f'Latency ({unit.unit_time})'],
            f'Backward Latency ({unit.unit_time})': unit.raw_to_unit(bwd_exec_time, type='T'),
            f'Total Latency ({unit.unit_time})': total_exec_time,
            f'Forward FLOPs ({unit.unit_flop})': fwd[f'Num ops ({unit.unit_flop})'],
            f'Backward FLOPs ({unit.unit_flop})': unit.raw_to_unit(bwd_num_ops, type='O'),
            f'Total FLOPs ({unit.unit_flop})': total_num_ops,
            'Forward Op Intensity': fwd['Op Intensity'],
            'Backward Op Intensity': bwd_op_intensity,
        }


class BackwardGEMM(TrainingOperator):
    """
    Backward pass for GEMM operation.

    Forward: Y = X @ W^T  (B, M, N) = (B, M, K) @ (N, K)^T
    Backward:
      - dL/dW = X^T @ dL/dY  ->  (N, K) from (K, M) @ (M, N)
      - dL/dX = dL/dY @ W    ->  (B, M, K) from (B, M, N) @ (N, K)

    Dimension format: [name, B, M, N, K] or [name, B, M, K] (K=N assumed)
    """

    def __init__(self, dim, density=(1.0, 1.0, 1.0), trainable=True):
        self.name = dim[0]
        super().__init__(dim=dim[1:], density=density, trainable=trainable)
        self._op_type = OpType.BACKWARD_GEMM

    def get_effective_dim_len(self):
        # Support both 3D (B, M, K) and 4D (B, M, N, K) formats
        return min(4, len(self.dim))

    def _get_dims(self):
        """Get B, M, N, K handling both 3D and 4D formats."""
        if len(self.dim) >= 4:
            B, M, N, K = self.dim[:4]
        elif len(self.dim) >= 3:
            # 3D format: (B, M, K), assume N=K (square projection)
            B, M, K = self.dim[:3]
            N = K
        else:
            # Fallback for malformed input
            B = self.dim[0] if len(self.dim) > 0 else 1
            M = self.dim[1] if len(self.dim) > 1 else 1
            K = M
            N = K
        return B, M, N, K

    def get_tensors(self):
        """Get tensors for backward pass: grad_output, weight, grad_input."""
        B, M, N, K = self._get_dims()
        grad_output = (B, M, N)  # dL/dY
        weight = (N, K)          # W for dL/dX computation
        grad_input = (B, M, K)   # dL/dX
        return grad_output, weight, grad_input

    def get_num_ops(self):
        """FLOPs for backward: dL/dW + dL/dX."""
        B, M, N, K = self._get_dims()
        # dL/dW: X^T @ dL/dY -> (K, M) @ (M, N) = (K, N), batched over B
        dW_ops = B * M * N * K
        # dL/dX: dL/dY @ W -> (M, N) @ (N, K) = (M, K), batched over B
        dX_ops = B * M * N * K
        if self.trainable:
            return dW_ops + dX_ops
        return dX_ops  # Only activation gradient if not trainable


class BackwardFC(TrainingOperator):
    """
    Backward pass for Fully Connected layer.

    Forward: Y = X @ W^T  (B, O) = (B, I) @ (O, I)^T
    """

    def __init__(self, dim, density=(1.0, 1.0, 1.0), trainable=True):
        self.name = dim[0]
        super().__init__(dim=dim[1:], density=density, trainable=trainable)
        self._op_type = OpType.BACKWARD_FC

    def get_effective_dim_len(self):
        return 3

    def get_tensors(self):
        B, O, I = self.dim[:self.get_effective_dim_len()]
        grad_output = (B, O)
        weight = (O, I)
        grad_input = (B, I)
        return grad_output, weight, grad_input

    def get_num_ops(self):
        B, O, I = self.dim[:self.get_effective_dim_len()]
        dW_ops = np.prod([B, O, I])  # X^T @ dL/dY
        dX_ops = np.prod([B, O, I])  # dL/dY @ W
        if self.trainable:
            return dW_ops + dX_ops
        return dX_ops


class BackwardLogit(TrainingOperator):
    """
    Backward pass for attention logit computation (Q @ K^T).

    Forward: Logits = Q @ K^T  (B, H, M, N) = (B, H, M, D) @ (B, Hkv, N, D)^T

    For GQA (Grouped Query Attention), H > Hkv:
    - Q has H heads, K/V have Hkv heads
    - Each group of (H/Hkv) Q heads shares one K/V head
    - Backward must account for gradient aggregation across groups

    Backward:
    - dL/dQ: (B, H, M, N) @ (B, Hkv, N, D) -> (B, H, M, D)
      FLOPs = B * H * M * N * D (broadcast K to H heads)
    - dL/dK: Q^T @ dL/dLogits -> (B, Hkv, N, D)
      FLOPs = B * Hkv * M * N * D (aggregate gradients from H/Hkv groups)
    """

    def __init__(self, dim, density=(1.0, 1.0, 1.0), trainable=True):
        self.name = dim[0]
        super().__init__(dim=dim[1:], density=density, trainable=trainable)
        self._op_type = OpType.BACKWARD_LOGIT

    def get_effective_dim_len(self):
        return 6

    def get_tensors(self):
        B, H, M, N, D, Hkv = self.dim[:self.get_effective_dim_len()]
        grad_output = (B, H, M, N)  # dL/dLogits
        key = (B, Hkv, N, D)        # K for dL/dQ
        grad_query = (B, H, M, D)   # dL/dQ
        return grad_output, key, grad_query

    def get_num_ops(self):
        B, H, M, N, D, Hkv = self.dim[:self.get_effective_dim_len()]
        # dL/dQ: dL/dLogits @ K -> (B, H, M, N) @ (B, Hkv, N, D) with K broadcast
        # FLOPs = B * H * M * N * D (computed for all H heads)
        dQ_ops = B * H * M * N * D

        # dL/dK: Q^T @ dL/dLogits with gradient aggregation for GQA
        # For each of Hkv heads, aggregate gradients from (H/Hkv) query head groups
        # FLOPs = B * Hkv * M * N * D (reduced from H to Hkv)
        dK_ops = B * Hkv * M * N * D

        return dQ_ops + dK_ops


class BackwardAttend(TrainingOperator):
    """
    Backward pass for attention value computation (Attention @ V).

    Forward: Output = Attention @ V  (B, H, M, D) = (B, H, M, N) @ (B, Hkv, N, D)

    For GQA (Grouped Query Attention), H > Hkv:
    - Attention has H heads, V has Hkv heads
    - Each group of (H/Hkv) attention heads shares one V head
    - Backward must account for gradient aggregation across groups

    Backward:
    - dL/dAttention: dL/dOutput @ V^T -> (B, H, M, N)
      FLOPs = B * H * M * N * D (broadcast V to H heads)
    - dL/dV: Attention^T @ dL/dOutput -> (B, Hkv, N, D)
      FLOPs = B * Hkv * M * N * D (aggregate gradients from H/Hkv groups)
    """

    def __init__(self, dim, density=(1.0, 1.0, 1.0), trainable=True):
        self.name = dim[0]
        super().__init__(dim=dim[1:], density=density, trainable=trainable)
        self._op_type = OpType.BACKWARD_ATTEND

    def get_effective_dim_len(self):
        return 6

    def get_tensors(self):
        B, H, M, N, D, Hkv = self.dim[:self.get_effective_dim_len()]
        grad_output = (B, H, M, D)   # dL/dOutput
        value = (B, Hkv, N, D)       # V for dL/dAttention
        grad_attention = (B, H, M, N) # dL/dAttention
        return grad_output, value, grad_attention

    def get_num_ops(self):
        B, H, M, N, D, Hkv = self.dim[:self.get_effective_dim_len()]
        # dL/dAttention: dL/dOutput @ V^T with V broadcast to H heads
        # FLOPs = B * H * M * N * D (computed for all H heads)
        dAttn_ops = B * H * M * N * D

        # dL/dV: Attention^T @ dL/dOutput with gradient aggregation for GQA
        # For each of Hkv heads, aggregate gradients from (H/Hkv) attention head groups
        # FLOPs = B * Hkv * M * N * D (reduced from H to Hkv)
        dV_ops = B * Hkv * M * N * D

        return dAttn_ops + dV_ops


class BackwardNorm(TrainingOperator):
    """
    Backward pass for LayerNorm/RMSNorm.

    LayerNorm backward pass requires:
    1. Computing gradient w.r.t. normalized input
    2. Computing gradient w.r.t. scale (gamma) parameter
    3. Computing gradient w.r.t. shift (beta) parameter (if using LayerNorm)

    For RMSNorm (most common in modern LLMs):
    - Forward: y = x * rsqrt(mean(x^2) + eps) * gamma
    - Backward: dL/dx = (dL/dy * gamma - mean(dL/dy * gamma * x_norm) * x_norm) * rsqrt(mean(x^2) + eps)

    FLOPs breakdown:
    - Forward: ~4 * B * S * D (compute mean, variance, normalize, scale)
    - Backward: ~6 * B * S * D (gradient + recomputation)
    - Backward/Forward ratio: ~1.5x (not exactly 2x due to reuse)
    """

    def __init__(self, dim, density=(1.0, 1.0, 1.0), trainable=True):
        """
        Initialize backward norm operator.

        Args:
            dim: [name, batch_size, seq_length, hidden_size]
            density: Sparsity factors
            trainable: Whether the norm parameters (gamma, beta) are trainable
        """
        self.name = dim[0]
        super().__init__(dim=dim[1:], density=density, trainable=trainable)
        self._op_type = OpType.BACKWARD_NORM

    def get_effective_dim_len(self):
        return 3  # batch_size, seq_length, hidden_size

    def get_tensors(self):
        B, S, D = self.dim[:self.get_effective_dim_len()]
        grad_output = (B, S, D)  # dL/dy
        gamma = (D,)             # Scale parameter
        grad_input = (B, S, D)   # dL/dx
        return grad_output, gamma, grad_input

    def get_num_ops(self):
        """
        Compute FLOPs for LayerNorm/RMSNorm backward.

        Backward operations:
        1. dL/dx computation: ~4 * B * S * D (chain rule through normalization)
        2. dL/dgamma computation: B * S * D (reduction over batch/sequence)
        3. dL/dbeta computation: B * S * D (for LayerNorm)

        Total: ~6 * B * S * D (approximately 1.5x forward)
        """
        B, S, D = self.dim[:self.get_effective_dim_len()]
        # Main backward computation: gradient through normalization
        dx_ops = 4 * B * S * D
        # Gradient for parameters (gamma, beta)
        if self.trainable:
            dgamma_ops = B * S * D
            dbeta_ops = B * S * D  # For LayerNorm (RMSNorm doesn't have beta)
            return dx_ops + dgamma_ops + dbeta_ops
        return dx_ops  # Only activation gradient if not trainable

    def get_backward_memory_multiplier(self) -> float:
        """
        Norm backward requires reading and writing similar-sized tensors.
        Memory access pattern: read grad_out, read x, read gamma, write grad_x
        """
        return 1.5


class BackwardSpecialFunc(TrainingOperator):
    """
    Backward pass for special functions (activation functions).

    Activation functions like GeLU, SiLU, ReLU, Softmax have backward passes
    that are typically element-wise operations.

    FLOPs breakdown:
    - Forward GeLU/SiLU: ~10-15 FLOPs per element (approximate functions)
    - Backward: ~5-10 FLOPs per element (gradient computation)

    For softmax specifically:
    - Forward: O(N) per row (exp, sum, divide)
    - Backward: O(N) per row (Jacobian-vector product simplification)
    """

    def __init__(self, dim, density=(1.0, 1.0, 1.0), trainable=True, func_type: str = 'gelu'):
        """
        Initialize backward special function operator.

        Args:
            dim: [name, batch_size, seq_length, hidden_size]
            density: Sparsity factors
            trainable: Whether this is part of trainable computation
            func_type: Type of activation ('gelu', 'silu', 'relu', 'softmax', etc.)
        """
        self.name = dim[0]
        self.func_type = func_type.lower()
        super().__init__(dim=dim[1:], density=density, trainable=trainable)
        self._op_type = OpType.Special_Func  # Reuse forward op type

    def get_effective_dim_len(self):
        return 3  # batch_size, seq_length, hidden_size

    def get_tensors(self):
        B, S, D = self.dim[:self.get_effective_dim_len()]
        grad_output = (B, S, D)  # dL/dy
        input_tensor = (B, S, D)  # x (needed for gradient)
        grad_input = (B, S, D)  # dL/dx
        return grad_output, input_tensor, grad_input

    def get_num_ops(self):
        """
        Compute FLOPs for activation backward.

        FLOPs per element vary by activation:
        - ReLU: 1 FLOP (comparison)
        - GeLU: ~8 FLOPs (derivative of approximate function)
        - SiLU: ~6 FLOPs (derivative)
        - Softmax: ~3 FLOPs (special Jacobian structure)
        """
        B, S, D = self.dim[:self.get_effective_dim_len()]
        num_elements = B * S * D

        flops_per_element = {
            'relu': 1,
            'gelu': 8,
            'gelu_new': 8,
            'gelu_pytorch_tanh': 10,
            'silu': 6,
            'tanh': 4,
            'softmax': 3,
            'gegelu': 10,
        }

        flops = flops_per_element.get(self.func_type, 6)
        return num_elements * flops


class BackwardEmbedding(TrainingOperator):
    """
    Backward pass for embedding lookup and output projection (lm_head).

    Embedding forward:
    - Input embedding: Lookup from embedding table (B, S) -> (B, S, D)
    - Output projection (lm_head): (B, S, D) @ (D, V) -> (B, S, V)

    Embedding backward:
    - Input embedding: Sparse gradient update (only touched token indices)
    - Output projection: Dense GEMM gradient

    Key insight: Embedding backward is SPARSE because only the tokens in the
    batch are updated. This is fundamentally different from dense GEMM backward.

    For a batch of B*S tokens, only ~B*S unique indices are updated (with duplicates
    if same tokens appear multiple times). This is much less work than a full
    V * D gradient computation.
    """

    def __init__(self, dim, density=(1.0, 1.0, 1.0), trainable=True):
        """
        Initialize backward embedding operator.

        Args:
            dim: [name, batch_size, seq_length, vocab_size, hidden_size]
            density: Sparsity factors
            trainable: Whether the embedding weights are trainable
        """
        self.name = dim[0]
        super().__init__(dim=dim[1:], density=density, trainable=trainable)
        self._op_type = OpType.BACKWARD_EMBEDDING

    def get_effective_dim_len(self):
        return 4  # batch_size, seq_length, vocab_size, hidden_size

    def get_tensors(self):
        B, S, V, D = self.dim[:self.get_effective_dim_len()]
        grad_output = (B, S, D)   # dL/dy (from first hidden layer)
        embedding_table = (V, D)  # Embedding table
        grad_input = (B, S)       # dL/dx (gradient indices, not tensor)
        return grad_output, embedding_table, grad_input

    def get_num_ops(self):
        """
        Compute FLOPs for embedding backward.

        For input embedding (sparse):
        - Only B * S tokens are updated
        - Each update is D elements (scatter-add)
        - FLOPs ≈ B * S * D (sparse, not V * D)

        For output projection (lm_head, dense):
        - dL/dW: X^T @ dL/dy -> (D, B*S) @ (B*S, V) = D * B * S * V
        - dL/dX: dL/dy @ W -> (B*S, V) @ (V, D) = B * S * V * D
        - Total: 2 * B * S * V * D

        Combined (treating as averaged):
        - Input emb backward: B * S * D
        - Output emb backward: 2 * B * S * V * D (this dominates)
        """
        B, S, V, D = self.dim[:self.get_effective_dim_len()]

        # Input embedding backward (sparse)
        # Scatter-add operations for unique tokens
        input_emb_ops = B * S * D

        # Output projection backward (dense lm_head)
        # dL/dW and dL/dX
        lm_head_ops = 2 * B * S * V * D

        if self.trainable:
            return input_emb_ops + lm_head_ops
        # If frozen, still need activation gradient for downstream layers
        return lm_head_ops // 2  # Only dL/dX

    def is_memory_bound(self) -> bool:
        """
        Embedding backward is typically memory-bound due to:
        1. Sparse access patterns (poor cache utilization)
        2. Large vocabulary table (random access)
        """
        return True


class OptimizerUpdate(Operator):
    """
    Optimizer update operation (e.g., AdamW update step).

    Models the compute and memory cost of applying optimizer updates to weights.
    Different optimizers have different memory and compute requirements:
    - SGD: 0 states, 1 FLOP per param
    - Adam/AdamW: 2 states (m, v), ~10 FLOPs per param
    - 8-bit Adam: 2 states (quantized), ~10 FLOPs per param
    - Lion: 1 state (momentum), ~6 FLOPs per param
    - LAMB/LARS: Layer-wise adaptive with 2 states
    - Sophia: Second-order with 2 states
    - GaLore/Flora: Low-rank with reduced states
    """

    # Comprehensive optimizer profiles
    OPTIMIZER_PROFILES = {
        # Standard optimizers
        'sgd': {'states': 0, 'flops': 1, 'memory_bytes': 0, 'description': 'Basic SGD'},
        'sgd_momentum': {'states': 1, 'flops': 3, 'memory_bytes': 4, 'description': 'SGD with momentum'},
        'adam': {'states': 2, 'flops': 10, 'memory_bytes': 8, 'description': 'Adam optimizer'},
        'adamw': {'states': 2, 'flops': 12, 'memory_bytes': 8, 'description': 'AdamW with decoupled weight decay'},

        # Memory-efficient optimizers
        'adam_8bit': {'states': 2, 'flops': 10, 'memory_bytes': 2, 'description': '8-bit Adam (bitsandbytes)'},
        'adamw_8bit': {'states': 2, 'flops': 12, 'memory_bytes': 2, 'description': '8-bit AdamW'},
        'paged_adamw_8bit': {'states': 2, 'flops': 10, 'memory_bytes': 2, 'description': 'Paged 8-bit AdamW'},
        'adafactor': {'states': 2, 'flops': 8, 'memory_bytes': 4, 'description': 'Adafactor (factorized states)'},

        # Newer optimizers
        'lion': {'states': 1, 'flops': 6, 'memory_bytes': 4, 'description': 'Lion optimizer (Google)'},
        'lamb': {'states': 2, 'flops': 15, 'memory_bytes': 8, 'description': 'LAMB (Layer-wise Adaptive)'},
        'lars': {'states': 1, 'flops': 8, 'memory_bytes': 4, 'description': 'LARS (Layer-wise Adaptive Rate Scaling)'},
        'sophia': {'states': 2, 'flops': 12, 'memory_bytes': 8, 'description': 'Sophia (Second-order)'},
        'schedule_free_adamw': {'states': 2, 'flops': 12, 'memory_bytes': 8, 'description': 'Schedule-free AdamW'},

        # Low-rank optimizers
        'galore': {'states': 2, 'flops': 10, 'memory_bytes': 2, 'rank_factor': 0.25, 'description': 'GaLore (Gradient Low-Rank Projection)'},
        'flora': {'states': 2, 'flops': 10, 'memory_bytes': 2, 'rank_factor': 0.1, 'description': 'Flora (Full-rank Low-rank Adaptation)'},

        # Additional optimizers
        'came': {'states': 2, 'flops': 12, 'memory_bytes': 6, 'description': 'CAME (Confidence-guided Adaptive)'},
        'adan': {'states': 3, 'flops': 15, 'memory_bytes': 12, 'description': 'Adan (Adaptive Nesterov)'},
        'prodigy': {'states': 2, 'flops': 14, 'memory_bytes': 8, 'description': 'Prodigy (Auto-tuning Adam)'},
        'shampoo': {'states': 2, 'flops': 20, 'memory_bytes': 16, 'description': 'Shampoo (full-matrix preconditioning)'},
        'muon': {'states': 1, 'flops': 5, 'memory_bytes': 4, 'description': 'Muon (momentum-based)'},
    }

    # Optimizer state sizes in bytes per parameter (backward compatibility)
    OPTIMIZER_STATE_BYTES = {k: v['memory_bytes'] for k, v in OPTIMIZER_PROFILES.items()}

    # FLOPs per parameter for optimizer update (backward compatibility)
    OPTIMIZER_FLOPS = {k: v['flops'] for k, v in OPTIMIZER_PROFILES.items()}

    def __init__(self, dim, density=(1.0, 1.0, 1.0), optimizer_type: str = 'adamw'):
        """
        Initialize optimizer update operation.

        Args:
            dim: [name, num_parameters, hidden_size]
            density: Sparsity factors
            optimizer_type: Type of optimizer
        """
        self.name = dim[0]
        self.num_parameters = int(dim[1])
        self.optimizer_type = optimizer_type.lower()
        super().__init__(dim=dim[1:], density=density)
        self._op_type = OpType.OPTIMIZER_UPDATE

    def get_tensors(self):
        # Parameters, gradients, updated parameters
        return (self.num_parameters,), (self.num_parameters,), (self.num_parameters,)

    def get_num_ops(self):
        flops_per_param = self.OPTIMIZER_FLOPS.get(self.optimizer_type, 10)
        return self.num_parameters * flops_per_param

    def get_memory_bytes(self) -> int:
        """Get total memory access for optimizer update."""
        # Read: parameters, gradients, optimizer states
        # Write: parameters, optimizer states
        state_bytes = self.OPTIMIZER_STATE_BYTES.get(self.optimizer_type, 8)
        param_bytes = 2  # bf16 parameters
        grad_bytes = 4   # fp32 gradients

        read_bytes = self.num_parameters * (param_bytes + grad_bytes + state_bytes)
        write_bytes = self.num_parameters * (param_bytes + state_bytes)

        return read_bytes + write_bytes


class GradientSync(Sync):
    """
    Gradient synchronization operation for distributed training.

    Extends Sync to handle gradient-specific communication patterns:
    - AllReduce: Standard data parallel gradient sync
    - ReduceScatter: ZeRO-2/3 gradient partitioning
    - AllGather: ZeRO-3 weight gathering
    """

    def __init__(self, dim, density=(1.0, 1.0, 1.0), zero_stage: int = 0):
        """
        Initialize gradient sync operation.

        Args:
            dim: [name, num_elements, element_size, num_nodes, collective_type]
            density: Sparsity factors
            zero_stage: ZeRO optimization stage (affects collective type)
        """
        self.zero_stage = zero_stage
        super().__init__(dim, density)

    def get_communication_time(self, system):
        """Get communication time based on ZeRO stage."""
        if self.get_op_type(self.dim) != 'Sync':
            return 0

        data_size = self.communication_data() * system.get_bit_multiplier(type='M', data='a')

        if self.collective_type == CollectiveType.ReduceScatter:
            return get_reduce_scatter_time(data_size, self.num_collective_nodes, system) / 1000
        elif self.collective_type == CollectiveType.AllGather:
            return get_AG_time(data_size, self.num_collective_nodes, system) / 1000
        else:
            # Default to parent implementation
            return super().get_communication_time(system)


def create_backward_layer_from_forward(
    forward_op: Operator,
    trainable: bool = True
) -> Optional[TrainingOperator]:
    """
    Create a backward operator from a forward operator.

    Args:
        forward_op: Forward pass operator
        trainable: Whether the layer is trainable

    Returns:
        Corresponding backward operator, or None if no backward needed
    """
    op_type = forward_op.get_op_type(forward_op.dim)

    if op_type == 'GEMM':
        return BackwardGEMM(
            [f"{forward_op.name}_backward"] + list(forward_op.dim),
            forward_op.get_density_list(),
            trainable=trainable
        )
    elif op_type == 'FC':
        return BackwardFC(
            [f"{forward_op.name}_backward"] + list(forward_op.dim),
            forward_op.get_density_list(),
            trainable=trainable
        )
    elif op_type == 'Logit':
        return BackwardLogit(
            [f"{forward_op.name}_backward"] + list(forward_op.dim),
            forward_op.get_density_list(),
            trainable=trainable
        )
    elif op_type == 'Attend':
        return BackwardAttend(
            [f"{forward_op.name}_backward"] + list(forward_op.dim),
            forward_op.get_density_list(),
            trainable=trainable
        )
    elif op_type == 'Norm':
        # Phase 7: Add backward for normalization layers
        # Extract dimensions from forward operator (B, S, D)
        fwd_dims = forward_op.get_dimensions()
        if isinstance(fwd_dims, tuple) and len(fwd_dims) >= 3:
            B, S, D = fwd_dims[0], fwd_dims[1], fwd_dims[2]
            return BackwardNorm(
                [f"{forward_op.name}_backward", B, S, D],
                forward_op.get_density_list(),
                trainable=trainable
            )
        return None
    elif op_type in ('Sync', 'Repeat', 'EndRepeat', 'Special_Func'):
        # These don't have meaningful backward operators in our model
        return None

    return None


def create_backward_embedding_op(
    name: str,
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    hidden_size: int,
    trainable: bool = True
) -> BackwardEmbedding:
    """
    Create a backward embedding operator for output projection (lm_head).

    Args:
        name: Layer name
        batch_size: Batch size
        seq_length: Sequence length
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension
        trainable: Whether embedding weights are trainable

    Returns:
        BackwardEmbedding operator
    """
    return BackwardEmbedding(
        [name, batch_size, seq_length, vocab_size, hidden_size],
        trainable=trainable
    )


def calculate_training_communication_time(
    num_parameters: int,
    num_dp_ranks: int,
    system,
    zero_stage: int = 0,
    precision_bytes: int = 4,  # FP32 gradients
) -> float:
    """
    Calculate total communication time for one training step.

    Args:
        num_parameters: Number of trainable parameters
        num_dp_ranks: Number of data parallel ranks
        system: GenZ System object
        zero_stage: ZeRO optimization stage
        precision_bytes: Bytes per gradient element

    Returns:
        Communication time in milliseconds
    """
    if num_dp_ranks <= 1:
        return 0.0

    gradient_bytes = num_parameters * precision_bytes

    if zero_stage <= 1:
        # Standard AllReduce for gradients
        return get_AR_time(gradient_bytes, num_dp_ranks, system)

    elif zero_stage == 2:
        # ReduceScatter for gradients (each rank keeps 1/N)
        return get_reduce_scatter_time(gradient_bytes, num_dp_ranks, system)

    else:  # zero_stage == 3
        # Forward: AllGather weights
        # Backward: AllGather weights (for recomputation) + ReduceScatter gradients
        weight_bytes = num_parameters * 2  # bf16 weights
        forward_ag = get_AG_time(weight_bytes, num_dp_ranks, system)
        backward_ag = get_AG_time(weight_bytes, num_dp_ranks, system)
        backward_rs = get_reduce_scatter_time(gradient_bytes, num_dp_ranks, system)

        return forward_ag + backward_ag + backward_rs


def estimate_backward_flops(
    forward_flops: int,
    trainable: bool = True,
    layer_type: str = 'ffn',
    num_heads: int = 32,
    num_kv_heads: int = 32,
) -> int:
    """
    Estimate backward FLOPs from forward FLOPs with layer-specific multipliers.

    Research-backed multipliers (Phase 1 Improvements):
    - Attention (FlashAttention-2): 2.5× forward (5 matmuls vs 2)
    - FFN: 2.0× forward (standard)
    - Non-trainable: 1× forward (dL/dX only)

    For GQA (num_kv_heads < num_heads), attention backward has additional
    overhead from broadcasting gradients.

    Args:
        forward_flops: Forward pass FLOPs
        trainable: Whether the layer is trainable
        layer_type: Type of layer ('attention', 'ffn', 'norm', 'embedding')
        num_heads: Number of attention heads (for GQA overhead)
        num_kv_heads: Number of KV heads (for GQA overhead)

    Returns:
        Estimated backward FLOPs
    """
    if not trainable:
        return forward_flops  # Only activation gradients

    if layer_type == 'attention':
        # FlashAttention-2 backward: 2.5× forward
        base_multiplier = 2.5

        # GQA overhead: gradient broadcasting from H to Hkv
        if num_kv_heads < num_heads:
            gqa_overhead = 1.0 + 0.1 * (num_heads / num_kv_heads - 1)
            return int(forward_flops * base_multiplier * gqa_overhead)

        return int(forward_flops * base_multiplier)

    elif layer_type == 'ffn':
        return int(forward_flops * 2.0)

    elif layer_type == 'norm':
        return int(forward_flops * 2.0)

    elif layer_type == 'embedding':
        return int(forward_flops * 2.0)

    else:
        # Default to 2× for unknown layer types
        return int(forward_flops * 2.0)


def list_optimizers() -> Dict[str, str]:
    """
    List all supported optimizers with their descriptions.

    Returns:
        Dictionary mapping optimizer name to description
    """
    return {
        name: profile['description']
        for name, profile in OptimizerUpdate.OPTIMIZER_PROFILES.items()
    }


def get_optimizer_profile(optimizer: str) -> Dict[str, Any]:
    """
    Get detailed profile for an optimizer.

    Args:
        optimizer: Optimizer name

    Returns:
        Dictionary with states, flops, memory_bytes, description, and optional rank_factor
    """
    optimizer_lower = optimizer.lower()
    if optimizer_lower not in OptimizerUpdate.OPTIMIZER_PROFILES:
        raise ValueError(
            f"Unknown optimizer: {optimizer}. "
            f"Available: {list(OptimizerUpdate.OPTIMIZER_PROFILES.keys())}"
        )
    return OptimizerUpdate.OPTIMIZER_PROFILES[optimizer_lower].copy()


def calculate_optimizer_memory(
    num_parameters: int,
    optimizer: str,
    master_weights_fp32: bool = True,
) -> Dict[str, float]:
    """
    Calculate memory requirements for optimizer states.

    Args:
        num_parameters: Number of trainable parameters
        optimizer: Optimizer type
        master_weights_fp32: Whether to store FP32 master weights (common with mixed precision)

    Returns:
        Dictionary with memory breakdown in bytes
    """
    profile = get_optimizer_profile(optimizer)

    # Optimizer state memory
    state_bytes = num_parameters * profile['memory_bytes']

    # Master weights (FP32 copy for mixed precision training)
    master_weight_bytes = num_parameters * 4 if master_weights_fp32 else 0

    # Low-rank optimizers reduce state memory
    rank_factor = profile.get('rank_factor', 1.0)
    state_bytes = int(state_bytes * rank_factor)

    return {
        'optimizer_states_bytes': state_bytes,
        'master_weights_bytes': master_weight_bytes,
        'total_bytes': state_bytes + master_weight_bytes,
        'total_gb': (state_bytes + master_weight_bytes) / 1e9,
    }


def get_memory_efficient_optimizers() -> List[str]:
    """Get list of memory-efficient optimizers (< 4 bytes per param for states)."""
    return [
        name for name, profile in OptimizerUpdate.OPTIMIZER_PROFILES.items()
        if profile['memory_bytes'] <= 4
    ]


def get_optimizers_by_category() -> Dict[str, List[str]]:
    """Get optimizers organized by category."""
    return {
        'standard': ['sgd', 'sgd_momentum', 'adam', 'adamw'],
        'memory_efficient': ['adam_8bit', 'adamw_8bit', 'paged_adamw_8bit', 'adafactor', 'lion'],
        'layer_adaptive': ['lamb', 'lars'],
        'second_order': ['sophia', 'shampoo'],
        'low_rank': ['galore', 'flora'],
        'modern': ['adan', 'prodigy', 'came', 'muon', 'schedule_free_adamw'],
    }


# ========================================
# Phase 7: Operator-Level Backward Analysis
# ========================================

def get_backward_model_df(
    forward_model_df,
    system,
    trainable_layers: Optional[List[str]] = None,
    method: str = 'full',
    unit=None,
    lora_target_modules: Optional[List[str]] = None,
    use_flash_attention: bool = True,
) -> 'pd.DataFrame':
    """
    Compute per-operator roofline analysis for backward pass.

    This is the backward-pass equivalent of get_model_df() for forward pass.
    It creates backward operators from the forward model definition and computes
    roofline metrics for each backward operator.

    Args:
        forward_model_df: DataFrame from get_model_df() with forward pass analysis
        system: GenZ System configuration
        trainable_layers: List of layer names that are trainable (default: all except Sync/Repeat)
        method: Training method ('full', 'lora', 'qlora', etc.) - affects which layers are trainable
        unit: Unit conversion helper (defaults to Unit())
        lora_target_modules: List of module names to target for LoRA (e.g., ['q_proj', 'v_proj'])
            If None, defaults to standard attention projections for LoRA methods.
        use_flash_attention: Whether to apply FlashAttention-2 backward overhead (default: True).
            FlashAttention-2 backward requires recomputing the attention matrix, adding ~25%
            overhead (2.5x instead of 2.0x for attention backward). This is the standard for
            modern LLM training.

    Returns:
        DataFrame with per-operator backward pass metrics:
        - Layer Name: Name of backward operator
        - Op Type: Operator type (BackwardGEMM, BackwardLogit, etc.)
        - Trainable: Whether this layer is trainable
        - Backward FLOPs: FLOPs for backward computation
        - Backward Latency: Time for backward pass
        - Bound: Whether compute or memory bound
    """
    import pandas as pd
    from .unit import Unit as UnitClass

    if unit is None:
        unit = UnitClass()

    backward_rows = []
    multiplier = 1  # For handling Repeat/EndRepeat blocks

    # Default LoRA targets (common attention projections)
    default_lora_targets = [
        'QKV', 'Out Proj', 'O_proj', 'out_proj',
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'query', 'key', 'value', 'dense',
        'self_attn', 'attention',
    ]

    # Determine which layers are trainable based on method
    if trainable_layers is None:
        if method == 'full':
            # All layers are trainable
            trainable_layers = None  # Means all
        elif method in ('lora', 'qlora', 'dora', 'pissa'):
            # Use provided LoRA targets or defaults
            trainable_layers = lora_target_modules if lora_target_modules else default_lora_targets
        elif method == 'freeze':
            # Only later layers are trainable (first half frozen)
            trainable_layers = None  # Handle separately
        else:
            trainable_layers = None

    for idx, row in forward_model_df.iterrows():
        # Pandas Series uses [] instead of .get()
        op_type = row['Op Type'] if 'Op Type' in row.index else ''
        layer_name = row['Layer Name'] if 'Layer Name' in row.index else f'layer_{idx}'

        # Handle Repeat/EndRepeat multipliers
        if op_type == 'Repeat':
            dimension = row['Dimension'] if 'Dimension' in row.index else 1
            # Handle tuple dimensions
            if isinstance(dimension, (list, tuple)):
                dimension = dimension[0] if len(dimension) > 0 else 1
            multiplier *= dimension
            continue
        elif op_type == 'EndRepeat':
            dimension = row['Dimension'] if 'Dimension' in row.index else 1
            if isinstance(dimension, (list, tuple)):
                dimension = dimension[0] if len(dimension) > 0 else 1
            multiplier /= dimension
            continue

        # Skip operators that don't have backward passes
        # Note: Special_Func (activations) DO have backward passes, so we include them
        if op_type in ('Sync', 'Repeat', 'EndRepeat'):
            continue

        # Determine if this layer is trainable
        if trainable_layers is None:
            is_trainable = True
        else:
            is_trainable = any(target in layer_name for target in trainable_layers)

        # Create backward operator based on forward operator type
        backward_op = _create_backward_op_from_row(row, op_type, is_trainable)
        if backward_op is None:
            continue

        # Compute roofline for backward operator
        # IMPORTANT: Use get_roofline() NOT get_training_roofline()
        # Backward operators already compute correct backward FLOPs in get_num_ops()
        # Using get_training_roofline() would double-count (it calls get_backward_num_ops() = 2 * get_num_ops())
        backward_roofline = backward_op.get_roofline(system, unit)

        # Scale by repeat multiplier
        # For backward operators, the latency comes from get_roofline() directly
        latency_key = f'Latency ({unit.unit_time})'
        flops_key = f'Num ops ({unit.unit_flop})'
        backward_latency = backward_roofline.get(latency_key, 0) * multiplier
        backward_flops = backward_roofline.get(flops_key, 0) * multiplier

        # Apply FlashAttention-2 backward overhead for attention operators
        # FlashAttention-2 requires recomputing attention during backward, adding 25% overhead
        # This gives 2.5x instead of 2.0x backward/forward ratio for attention
        if use_flash_attention and op_type in ('Logit', 'Attend'):
            FLASH_ATTENTION_BACKWARD_OVERHEAD = 1.25  # 2.5 / 2.0
            backward_latency *= FLASH_ATTENTION_BACKWARD_OVERHEAD
            backward_flops *= FLASH_ATTENTION_BACKWARD_OVERHEAD

        backward_rows.append({
            'Layer Name': f"{layer_name}_backward",
            'Op Type': f"Backward{op_type}",
            'Forward Layer': layer_name,
            'Trainable': is_trainable,
            f'Backward FLOPs ({unit.unit_flop})': backward_flops,
            f'Backward Latency ({unit.unit_time})': backward_latency,
            'Bound': backward_roofline.get('Bound', 'Unknown'),
            'Op Intensity': backward_roofline.get('Op Intensity', 0),
            'Repeat Multiplier': multiplier,
        })

    return pd.DataFrame(backward_rows)


def _create_backward_op_from_row(row, op_type: str, trainable: bool) -> Optional[TrainingOperator]:
    """
    Create a backward operator from a forward model DataFrame row.

    Args:
        row: Row from forward_model_df (pandas Series)
        op_type: Operator type string
        trainable: Whether this layer is trainable

    Returns:
        Backward operator instance, or None if not applicable
    """
    # Handle pandas Series
    layer_name = row['Layer Name'] if 'Layer Name' in row.index else 'unknown'
    dimensions = row['Dimension'] if 'Dimension' in row.index else ()

    if dimensions is None:
        return None

    # Parse dimensions which can be in multiple formats:
    # 1. GEMM: [((B, M, N), (K, L), (O, P))] -> Extract from nested tuples
    # 2. Logit/Attend: ((B, H, M, D), (B, Hkv, M, D), (B, H, M, M)) -> Extract Q, K shapes
    # 3. Simple: (B, M, N, K) -> Use directly
    # 4. Int (Repeat/EndRepeat): 32 -> Skip

    if isinstance(dimensions, int):
        return None

    # Store original dimensions for multi-tensor operators (Logit/Attend)
    original_dimensions = dimensions

    if isinstance(dimensions, list) and len(dimensions) > 0:
        # Handle nested list format: [((B, M, N), (K, L), (O, P))]
        first_elem = dimensions[0]
        if isinstance(first_elem, tuple) and len(first_elem) > 0:
            # Keep track of all tensor shapes for multi-tensor operators
            # This is needed to extract N from weight tensor for GEMM
            original_dimensions = first_elem
            # Get the first tuple (input dimensions)
            dimensions = first_elem[0] if isinstance(first_elem[0], tuple) else first_elem
    elif isinstance(dimensions, tuple) and len(dimensions) > 0:
        # Check if it's nested tuples (tensor shapes)
        if isinstance(dimensions[0], tuple):
            # Keep track of all tensor shapes for GQA handling
            original_dimensions = dimensions
            dimensions = dimensions[0]  # Use first tensor for primary dimensions

    # Now dimensions should be a flat tuple like (B, M, N, K, ...)
    if not isinstance(dimensions, tuple) or len(dimensions) == 0:
        return None

    if op_type == 'GEMM':
        # GEMM: Y = X @ W^T where X is (B, M, K) and W is (N, K), Y is (B, M, N)
        # Forward model stores dimensions as ((B, M, K), (N, K), (B, M, N))
        # We need B, M, N, K for backward FLOPs: dW=(K,M)@(M,N)=KN, dX=(M,N)@(N,K)=MK
        if len(dimensions) >= 3:
            B = dimensions[0]
            M = dimensions[1]
            K = dimensions[2] if len(dimensions) > 2 else dimensions[1]

            # Try to get N from the second tensor if available
            N = K  # Default: assume square
            if isinstance(original_dimensions, tuple) and len(original_dimensions) > 1:
                second_tensor = original_dimensions[1]
                if isinstance(second_tensor, tuple) and len(second_tensor) >= 1:
                    N = second_tensor[0]  # Weight shape is (N, K)
            elif len(dimensions) > 3:
                N = dimensions[3]

            return BackwardGEMM(
                [f"{layer_name}_backward", B, M, N, K],
                trainable=trainable
            )
    elif op_type == 'FC':
        # FC: (B, O, I)
        if len(dimensions) >= 3:
            B, O, I = dimensions[:3]
            return BackwardFC(
                [f"{layer_name}_backward", B, O, I],
                trainable=trainable
            )
    elif op_type == 'Logit':
        # Logit dimensions: ((B, H, M, D), (B, Hkv, N, D), (B, H, M, N))
        # For Q @ K^T, we need Q shape (B, H, M, D) and K shape (B, Hkv, N, D)
        if len(dimensions) >= 4:
            B, H, M, D = dimensions[:4]
            N = M  # For self-attention, N = M

            # Try to extract Hkv from K tensor shape for GQA support
            Hkv = H  # Default to MHA
            if isinstance(original_dimensions, tuple) and len(original_dimensions) > 1:
                k_tensor = original_dimensions[1]
                if isinstance(k_tensor, tuple) and len(k_tensor) >= 2:
                    Hkv = k_tensor[1]  # K shape is (B, Hkv, N, D)
                    if len(k_tensor) >= 3:
                        N = k_tensor[2]  # Get N from K tensor

            return BackwardLogit(
                [f"{layer_name}_backward", B, H, M, N, D, Hkv],
                trainable=trainable
            )
    elif op_type == 'Attend':
        # Attend dimensions: ((B, H, M, N), (B, Hkv, N, D), (B, H, M, D))
        # For Attn @ V, we need Attn shape (B, H, M, N) and V shape (B, Hkv, N, D)
        if len(dimensions) >= 4:
            B, H, M, D_or_N = dimensions[:4]

            # For Attend, first tensor is attention weights (B, H, M, N)
            # D comes from V tensor, N from attention weights
            N = D_or_N  # Fourth dim is N for attention weights
            D = D_or_N  # Default

            # Try to extract Hkv and D from V tensor shape for GQA support
            Hkv = H  # Default to MHA
            if isinstance(original_dimensions, tuple) and len(original_dimensions) > 1:
                v_tensor = original_dimensions[1]
                if isinstance(v_tensor, tuple) and len(v_tensor) >= 4:
                    Hkv = v_tensor[1]  # V shape is (B, Hkv, N, D)
                    D = v_tensor[3]  # Get D from V tensor
                elif isinstance(v_tensor, tuple) and len(v_tensor) >= 2:
                    Hkv = v_tensor[1]

            return BackwardAttend(
                [f"{layer_name}_backward", B, H, M, N, D, Hkv],
                trainable=trainable
            )
    elif op_type == 'Norm':
        # Norm: (B, S, D) or similar
        if len(dimensions) >= 3:
            B, S, D = dimensions[:3]
            return BackwardNorm(
                [f"{layer_name}_backward", B, S, D],
                trainable=trainable
            )
    elif op_type == 'Special_Func':
        # Special functions (activations): (B, S, D) or similar
        # Try to infer function type from layer name
        func_type = 'gelu'  # Default
        name_lower = layer_name.lower()
        if 'silu' in name_lower or 'swish' in name_lower:
            func_type = 'silu'
        elif 'relu' in name_lower:
            func_type = 'relu'
        elif 'tanh' in name_lower:
            func_type = 'tanh'
        elif 'softmax' in name_lower:
            func_type = 'softmax'
        elif 'gegelu' in name_lower:
            func_type = 'gegelu'

        if len(dimensions) >= 3:
            B, S, D = dimensions[:3]
            return BackwardSpecialFunc(
                [f"{layer_name}_backward", B, S, D],
                trainable=trainable,
                func_type=func_type,
            )

    return None


def get_optimizer_model_df(
    num_params: int,
    optimizer: str,
    system,
    batch_size: int = 1,
    precision_bytes: int = 2,
    zero_stage: int = 0,
    data_parallel: int = 1,
    unit=None,
) -> 'pd.DataFrame':
    """
    Compute operator-level roofline analysis for optimizer update.

    The optimizer kernel is typically memory-bound:
    - Read: parameters, gradients, momentum (m), variance (v)
    - Write: updated parameters, momentum, variance
    - FLOPs: 8-12 per parameter for Adam

    This function models the optimizer as a single "mega-operator" and computes
    its roofline characteristics (boundedness, operational intensity, latency).

    Args:
        num_params: Number of trainable parameters
        optimizer: Optimizer type (adamw, adam, sgd, lion, etc.)
        system: GenZ System configuration
        batch_size: Not used directly, but kept for API consistency
        precision_bytes: Bytes per weight element (2 for bf16)
        zero_stage: ZeRO optimization stage (affects sharding)
        data_parallel: Data parallel degree (affects sharding with ZeRO)
        unit: Unit conversion helper

    Returns:
        DataFrame with optimizer roofline metrics:
        - Layer Name: Optimizer_{type}
        - Op Type: OptimizerUpdate
        - Num Ops: Total FLOPs
        - Latency: Execution time
        - Bound: Compute or Memory bound
        - Op Intensity: FLOPs per byte
    """
    import pandas as pd
    from .unit import Unit as UnitClass

    if unit is None:
        unit = UnitClass()

    # Get optimizer profile
    profile = OptimizerUpdate.OPTIMIZER_PROFILES.get(
        optimizer.lower(),
        OptimizerUpdate.OPTIMIZER_PROFILES['adamw']
    )

    # Calculate bytes moved
    state_bytes_per_param = profile['memory_bytes']
    flops_per_param = profile['flops']

    # ZeRO sharding reduces local params
    local_params = num_params
    if zero_stage >= 1:
        local_params = num_params // max(1, data_parallel)

    # Total bytes: read weights, gradients, states; write weights, states
    # Read: params (precision_bytes) + grads (4 bytes FP32) + states (state_bytes)
    # Write: params (precision_bytes) + states (state_bytes)
    read_bytes = local_params * (precision_bytes + 4 + state_bytes_per_param)
    write_bytes = local_params * (precision_bytes + state_bytes_per_param)
    total_bytes = read_bytes + write_bytes

    # Total FLOPs
    total_flops = local_params * flops_per_param

    # Roofline analysis
    op_intensity = total_flops / total_bytes if total_bytes > 0 else 0

    # Compute time: FLOPs / FLOPS_per_second
    # Note: Optimizer ops are simple scalar operations, not tensor core accelerated
    # Use reduced throughput (1/4 of peak) for scalar operations
    scalar_throughput = system.op_per_sec * system.compute_efficiency * 0.25
    compute_time_s = total_flops / scalar_throughput if scalar_throughput > 0 else 0

    # Memory time: bytes / bandwidth
    memory_time_s = total_bytes / (system.offchip_mem_bw * system.memory_efficiency)

    # Execution time is max of compute and memory
    exec_time_s = max(compute_time_s, memory_time_s)
    boundedness = 'Compute' if compute_time_s > memory_time_s else 'Memory'

    # Convert to ms
    exec_time_ms = exec_time_s * 1000

    # Create result DataFrame
    result = pd.DataFrame([{
        'Layer Name': f'Optimizer_{optimizer}',
        'Op Type': 'OptimizerUpdate',
        'Optimizer': optimizer,
        'Local Params': local_params,
        'Total Params': num_params,
        f'Num Ops ({unit.unit_flop})': unit.raw_to_unit(total_flops, type='O'),
        f'Total Data ({unit.unit_mem})': unit.raw_to_unit(total_bytes, type='M'),
        'Op Intensity': op_intensity,
        f'Latency ({unit.unit_time})': exec_time_ms,
        'Bound': boundedness,
        f'Compute Time ({unit.unit_time})': compute_time_s * 1000,
        f'Memory Time ({unit.unit_time})': memory_time_s * 1000,
        'ZeRO Stage': zero_stage,
        'Data Parallel': data_parallel,
    }])

    return result


def compute_backward_timing_from_forward(
    forward_time_ms: float,
    forward_model_df,
    system,
    trainable_layers: Optional[List[str]] = None,
    method: str = 'full',
    num_heads: int = 32,
    num_kv_heads: int = 32,
    use_operator_roofline: bool = True,
) -> Dict[str, float]:
    """
    Compute backward timing using operator-level roofline analysis.

    This is the main entry point for Phase 7 operator-level backward analysis.
    It can use either:
    1. Operator-level roofline (accurate per-layer analysis)
    2. Multiplier-based estimation (fallback)

    Args:
        forward_time_ms: Forward pass time in milliseconds
        forward_model_df: DataFrame from get_model_df() with forward pass analysis
        system: GenZ System configuration
        trainable_layers: List of trainable layer names
        method: Training method ('full', 'lora', etc.)
        num_heads: Number of attention heads (for GQA overhead calculation)
        num_kv_heads: Number of KV heads (for GQA overhead calculation)
        use_operator_roofline: Whether to use operator-level analysis (True) or multipliers (False)

    Returns:
        Dictionary with:
        - backward_time_ms: Total backward time
        - backward_breakdown: Per-layer breakdown (if use_operator_roofline=True)
        - backward_method: 'operator_roofline' or 'multiplier'
    """
    from .unit import Unit

    if use_operator_roofline and forward_model_df is not None:
        try:
            # Use operator-level backward analysis
            unit = Unit()
            backward_df = get_backward_model_df(
                forward_model_df,
                system,
                trainable_layers=trainable_layers,
                method=method,
                unit=unit,
            )

            # Sum backward latencies
            # The column is named "Backward Latency (msec)" from get_backward_model_df()
            backward_lat_col = f'Backward Latency ({unit.unit_time})'
            if backward_lat_col in backward_df.columns:
                backward_time_ms = backward_df[backward_lat_col].sum()
            else:
                # Try alternate column names
                lat_col = [c for c in backward_df.columns if 'Latency' in c]
                if lat_col:
                    backward_time_ms = backward_df[lat_col[0]].sum()
                else:
                    backward_time_ms = 0

            # Create breakdown by operator type
            breakdown = {}
            if 'Op Type' in backward_df.columns:
                for op_type in backward_df['Op Type'].unique():
                    op_df = backward_df[backward_df['Op Type'] == op_type]
                    lat_col = [c for c in op_df.columns if 'Latency' in c]
                    if lat_col:
                        breakdown[op_type] = op_df[lat_col[0]].sum()

            return {
                'backward_time_ms': backward_time_ms,
                'backward_breakdown': breakdown,
                'backward_method': 'operator_roofline',
                'backward_df': backward_df,
            }
        except Exception as e:
            # Fall back to multiplier method on error
            import warnings
            warnings.warn(f"Operator-level backward failed: {e}, falling back to multiplier method")

    # Fallback: Use multiplier-based estimation
    # Weighted backward multiplier based on layer composition
    # Attention ~40% of FLOPs, FFN ~60%
    from .LLM_training.training_modeling import calculate_backward_multiplier, calculate_gqa_backward_overhead

    attn_mult = calculate_backward_multiplier('attention', num_heads, num_kv_heads)
    ffn_mult = calculate_backward_multiplier('ffn')
    backward_multiplier = 0.4 * attn_mult + 0.6 * ffn_mult

    # For partial training, reduce based on trainable ratio
    if method in ('lora', 'qlora', 'dora', 'pissa'):
        # LoRA: ~2% of params trainable, but backward still flows through frozen layers
        backward_multiplier = 2.0 * 0.02 + (1.0 - 0.02) * 1.0  # dL/dX for frozen layers

    backward_time_ms = forward_time_ms * backward_multiplier

    return {
        'backward_time_ms': backward_time_ms,
        'backward_breakdown': {
            'attention': forward_time_ms * 0.4 * attn_mult,
            'ffn': forward_time_ms * 0.6 * ffn_mult,
        },
        'backward_method': 'multiplier',
        'backward_multiplier': backward_multiplier,
    }
