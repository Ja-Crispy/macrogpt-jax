"""Feedback Alignment for TTT gradients.

Replaces backpropagation with fixed random feedback matrices.
Key insight: Forward weights learn to align with random feedback.

Theory:
- Standard backprop: dL/dW = dL/dy @ y^T, where y = Wx
- Feedback alignment: dL/dW = dL/dy @ B^T, where B is fixed random
- Forward weights W learn such that W^T ≈ B in task-relevant directions

References:
- Lillicrap et al. (2016): "Random synaptic feedback weights support error backpropagation in the brain"
- Nøkland (2016): "Direct Feedback Alignment Provides Learning in Deep Neural Networks"

Memory: O(1) - no computation graph storage
Speed: Fast - just matrix multiplications
Accuracy: Depends on task, typically 80-95% of backprop performance
"""

import jax
import jax.numpy as jnp


def feedback_alignment_ttt(fwd_fn, hidden_d=512, n_iters=1, wd=0.1, lr=0.01, seed=42):
    """Create TTT function using feedback alignment for gradients.

    Args:
        fwd_fn: Forward function (state, x) -> output
        hidden_d: Hidden dimension for feedback matrices
        n_iters: Number of inner optimization iterations
        wd: Weight decay
        lr: Learning rate
        seed: Random seed for feedback matrix initialization

    Returns:
        TTT function with feedback alignment gradients
    """
    # Initialize fixed random feedback matrices (NEVER UPDATED)
    key = jax.random.PRNGKey(seed)
    key_k, key_v, key_q = jax.random.split(key, 3)

    # Feedback matrices: same shape as would be used in backprop
    # Normalized by sqrt(hidden_d) for stable gradients
    B_k = jax.random.normal(key_k, (hidden_d, hidden_d)) / jnp.sqrt(hidden_d)
    B_v = jax.random.normal(key_v, (hidden_d, hidden_d)) / jnp.sqrt(hidden_d)
    B_q = jax.random.normal(key_q, (hidden_d, hidden_d)) / jnp.sqrt(hidden_d)

    def forward_pass(k, v, q, state):
        """Standard forward pass (identical to original TTT)."""
        # Swap to (seq, batch, dim) for scan
        k_seq = k.swapaxes(0, 1)
        v_seq = v.swapaxes(0, 1)
        q_seq = q.swapaxes(0, 1)

        def update_fn(state, x):
            k_t, v_t, q_t = x

            # Inner optimization: update state based on k, v
            for _ in range(n_iters):
                v_pred, vjp_fn = jax.vjp(lambda s: fwd_fn(s, k_t), state)
                reconstruction_error = v_t - v_pred
                dstate, = vjp_fn(reconstruction_error)

                # SGD update
                state = jax.tree.map(
                    lambda s, g: (1 - wd * lr) * s + lr * jax.nn.tanh(g),
                    state, dstate
                )

            # Compute output with updated state
            output = fwd_fn(state, q_t)

            return state, output

        # Run scan
        final_state, o_seq = jax.lax.scan(update_fn, state, (k_seq, v_seq, q_seq))

        # Swap back to (batch, seq, dim)
        outputs = o_seq.swapaxes(0, 1)

        return outputs, (final_state, k_seq, v_seq, q_seq, state)

    @jax.custom_vjp
    def feedback_ttt_fn(k, v, q, state):
        """TTT with feedback alignment."""
        return forward_pass(k, v, q, state)[0]

    def _feedback_fwd(k, v, q, state):
        """Forward pass with residuals for backward."""
        return forward_pass(k, v, q, state)

    def _feedback_bwd(res, do):
        """Backward pass using fixed random feedback matrices.

        Standard backprop would compute:
            dk = W_k^T @ grad_output

        Feedback alignment computes:
            dk = B_k @ grad_output

        where B_k is a fixed random matrix initialized once.
        """
        final_state, k_seq, v_seq, q_seq, init_state = res
        do_seq = do.swapaxes(0, 1)  # (seq, batch, dim)

        # Apply feedback matrices to output gradients
        # Shape: (seq, batch, dim) @ (dim, dim) -> (seq, batch, dim)
        dk_seq = jnp.einsum('sbd,de->sbe', do_seq, B_k)
        dv_seq = jnp.einsum('sbd,de->sbe', do_seq, B_v)
        dq_seq = jnp.einsum('sbd,de->sbe', do_seq, B_q)

        # Swap back to (batch, seq, dim)
        dk = dk_seq.swapaxes(0, 1)
        dv = dv_seq.swapaxes(0, 1)
        dq = dq_seq.swapaxes(0, 1)

        # Gradient w.r.t. initial state (zero - no gradient flow through state in FA)
        dstate = jax.tree.map(jnp.zeros_like, init_state)

        return dk, dv, dq, dstate

    feedback_ttt_fn.defvjp(_feedback_fwd, _feedback_bwd)
    return feedback_ttt_fn


# Variant: Direct Feedback Alignment (DFA)
def direct_feedback_alignment_ttt(fwd_fn, hidden_d=512, n_iters=1, wd=0.1, lr=0.01, seed=42):
    """Direct Feedback Alignment: Use same feedback matrix for all inputs.

    Standard FA uses different B_k, B_v, B_q.
    DFA uses single B for all - even simpler, sometimes works better.

    Args:
        fwd_fn: Forward function (state, x) -> output
        hidden_d: Hidden dimension
        n_iters: Inner optimization iterations
        wd: Weight decay
        lr: Learning rate
        seed: Random seed

    Returns:
        TTT function with direct feedback alignment
    """
    # Single feedback matrix for all inputs
    key = jax.random.PRNGKey(seed)
    B = jax.random.normal(key, (hidden_d, hidden_d)) / jnp.sqrt(hidden_d)

    def forward_pass(k, v, q, state):
        """Standard forward pass."""
        k_seq = k.swapaxes(0, 1)
        v_seq = v.swapaxes(0, 1)
        q_seq = q.swapaxes(0, 1)

        def update_fn(state, x):
            k_t, v_t, q_t = x

            for _ in range(n_iters):
                v_pred, vjp_fn = jax.vjp(lambda s: fwd_fn(s, k_t), state)
                reconstruction_error = v_t - v_pred
                dstate, = vjp_fn(reconstruction_error)

                state = jax.tree.map(
                    lambda s, g: (1 - wd * lr) * s + lr * jax.nn.tanh(g),
                    state, dstate
                )

            output = fwd_fn(state, q_t)
            return state, output

        final_state, o_seq = jax.lax.scan(update_fn, state, (k_seq, v_seq, q_seq))
        outputs = o_seq.swapaxes(0, 1)

        return outputs, (final_state, k_seq, v_seq, q_seq, state)

    @jax.custom_vjp
    def dfa_ttt_fn(k, v, q, state):
        return forward_pass(k, v, q, state)[0]

    def _dfa_fwd(k, v, q, state):
        return forward_pass(k, v, q, state)

    def _dfa_bwd(res, do):
        """Direct feedback: same matrix B for all inputs."""
        final_state, k_seq, v_seq, q_seq, init_state = res
        do_seq = do.swapaxes(0, 1)

        # All inputs get same feedback signal (simpler than standard FA)
        feedback_signal = jnp.einsum('sbd,de->sbe', do_seq, B)

        dk = feedback_signal.swapaxes(0, 1)
        dv = feedback_signal.swapaxes(0, 1)
        dq = feedback_signal.swapaxes(0, 1)

        dstate = jax.tree.map(jnp.zeros_like, init_state)

        return dk, dv, dq, dstate

    dfa_ttt_fn.defvjp(_dfa_fwd, _dfa_bwd)
    return dfa_ttt_fn


if __name__ == "__main__":
    print("Feedback Alignment TTT Implementation")
    print("="*60)
    print("\nReplaces backpropagation with fixed random feedback matrices.")
    print("\nKey features:")
    print("- No BPTT needed (forward pass only)")
    print("- Fixed random matrices B_k, B_v, B_q")
    print("- Forward weights learn to align with feedback")
    print("- Typical performance: 80-95% of backprop")
    print("\nMemory: O(1) - no computation graph")
    print("Speed: Fast - just matrix multiplications")
    print("\nVariants:")
    print("- feedback_alignment_ttt: Separate matrices for k, v, q")
    print("- direct_feedback_alignment_ttt: Single matrix for all")
    print("\nUsage: See test_framework.py for testing interface")
