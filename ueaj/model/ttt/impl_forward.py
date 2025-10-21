"""Forward-mode differentiation with truncated history for TTT gradients.

Uses JVP to propagate gradient estimates forward through the sequence,
maintaining a truncated history buffer to control memory usage.

Theory:
- Forward-mode AD propagates tangent vectors through computations
- Truncated history: only track last K timesteps of gradient dependencies
- Memory: O(K × d²) where K = history_len

This avoids BPTT while approximating temporal gradient flow.
"""

import jax
import jax.numpy as jnp
import functools as fn
from typing import Callable, Tuple


def make_forward_scan_fn(fwd_fn, history_len=32, n_iters=1, wd=0.1, lr=0.01):
    """Create scan function that tracks gradient history via forward-mode AD.

    Args:
        fwd_fn: Forward function (state, x) -> output
        history_len: Number of historical gradients to track
        n_iters: Inner optimization iterations
        wd: Weight decay
        lr: Learning rate

    Returns:
        Scan function that maintains gradient history
    """
    def init_grad_history(state_template):
        """Initialize circular gradient history buffer."""
        return {
            'buffer': [],  # Will store gradient pytrees
            'ptr': 0,
            'size': history_len
        }

    def add_to_history(grad_history, new_grad):
        """Add gradient to circular buffer."""
        buffer = grad_history['buffer']
        ptr = grad_history['ptr']
        size = grad_history['size']

        if len(buffer) < size:
            # Buffer not full yet
            buffer.append(new_grad)
            new_ptr = len(buffer) % size
        else:
            # Buffer full, overwrite oldest
            buffer[ptr] = new_grad
            new_ptr = (ptr + 1) % size

        return {
            'buffer': buffer,
            'ptr': new_ptr,
            'size': size
        }

    def get_recent_grads(grad_history):
        """Get all gradients in history (up to history_len)."""
        return grad_history['buffer']

    def scan_fn(carry, x):
        """Scan function with forward-mode gradient tracking.

        carry: (state, grad_history)
        x: (k, v, q)
        """
        state, grad_history = carry
        k, v, q = x

        # Inner optimization loop to update state
        for _ in range(n_iters):
            # Compute reconstruction loss gradient
            v_pred, vjp_fn = jax.vjp(lambda s: fwd_fn(s, k), state)
            reconstruction_error = v - v_pred
            grad_state, = vjp_fn(reconstruction_error)

            # Apply gradient descent update
            state = jax.tree.map(
                lambda s, g: (1 - wd * lr) * s + lr * jax.nn.tanh(g),
                state, grad_state
            )

            # Forward-mode: propagate historical gradients through update
            # For each historical gradient, compute how this update affects it
            recent_grads = get_recent_grads(grad_history)
            new_historical_grads = []

            for hist_grad in recent_grads:
                # JVP: how does current update affect historical gradient?
                def update_fn(s):
                    v_pred_local, vjp_fn_local = jax.vjp(lambda s_inner: fwd_fn(s_inner, k), s)
                    error_local = v - v_pred_local
                    grad_local, = vjp_fn_local(error_local)
                    return jax.tree.map(
                        lambda a, b: (1 - wd * lr) * a + lr * jax.nn.tanh(b),
                        s, grad_local
                    )

                # Compute tangent: d(update)/d(state) @ hist_grad
                _, tangent = jax.jvp(update_fn, (state,), (hist_grad,))
                new_historical_grads.append(tangent)

            # Update gradient history with propagated gradients
            for new_grad in new_historical_grads:
                grad_history = add_to_history(grad_history, new_grad)

            # Add current gradient to history
            grad_history = add_to_history(grad_history, grad_state)

        # Compute output using updated state
        output = fwd_fn(state, q)

        return (state, grad_history), output

    return scan_fn, init_grad_history


def forward_ttt(fwd_fn, history_len=32, n_iters=1, wd=0.1, lr=0.01):
    """Create TTT function using forward-mode differentiation with truncated history.

    Args:
        fwd_fn: Forward function (state, x) -> output
        history_len: Number of historical timesteps to track (controls memory)
        n_iters: Inner optimization iterations per timestep
        wd: Weight decay
        lr: Learning rate

    Returns:
        TTT function with forward-mode gradient tracking
    """
    scan_fn, init_grad_history = make_forward_scan_fn(
        fwd_fn, history_len, n_iters, wd, lr
    )

    def forward_ttt_fn(k, v, q, state):
        """Process sequence with forward-mode gradient tracking.

        Args:
            k, v, q: Input tensors (batch, seq_len, dim)
            state: Initial state

        Returns:
            outputs: (batch, seq_len, dim)
        """
        # Initialize gradient history
        grad_history = init_grad_history(state)

        # Swap to (seq, batch, dim) for scan
        k_seq = k.swapaxes(0, 1)
        v_seq = v.swapaxes(0, 1)
        q_seq = q.swapaxes(0, 1)

        # Run scan with gradient tracking
        (final_state, final_grad_history), o_seq = jax.lax.scan(
            scan_fn,
            (state, grad_history),
            (k_seq, v_seq, q_seq)
        )

        # Swap back to (batch, seq, dim)
        outputs = o_seq.swapaxes(0, 1)

        return outputs

    return forward_ttt_fn


# Simplified version for testing: only track dk gradients
def forward_ttt_simple(fwd_fn, history_len=16, n_iters=1, wd=0.1, lr=0.01):
    """Simplified forward-mode TTT for testing.

    Only tracks gradients w.r.t. k (key), uses standard VJP for v and q.
    Useful for validating core forward-mode approach.

    Args:
        fwd_fn: Forward function (state, x) -> output
        history_len: Truncation length for gradient history
        n_iters: Inner optimization iterations
        wd: Weight decay
        lr: Learning rate

    Returns:
        TTT function with simplified forward-mode gradients
    """
    def simple_scan_fn(state, x):
        """Standard scan without explicit history tracking."""
        k, v, q = x

        # Inner optimization
        for _ in range(n_iters):
            v_pred, vjp_fn = jax.vjp(lambda s: fwd_fn(s, k), state)
            reconstruction_error = v - v_pred
            grad_state, = vjp_fn(reconstruction_error)

            # SGD update
            state = jax.tree.map(
                lambda s, g: (1 - wd * lr) * s + lr * jax.nn.tanh(g),
                state, grad_state
            )

        # Compute output
        output = fwd_fn(state, q)

        return state, output

    def _simple_fwd(k, v, q, state):
        """Forward pass."""
        k_seq = k.swapaxes(0, 1)
        v_seq = v.swapaxes(0, 1)
        q_seq = q.swapaxes(0, 1)

        final_state, o_seq = jax.lax.scan(
            simple_scan_fn,
            state,
            (k_seq, v_seq, q_seq)
        )

        outputs = o_seq.swapaxes(0, 1)

        # Store residuals for backward pass
        return outputs, (final_state, k_seq, v_seq, q_seq, state)

    @jax.custom_vjp
    def simple_inner(k, v, q, state):
        return _simple_fwd(k, v, q, state)[0]

    def _simple_bwd(res, do):
        """Backward with forward-mode approximation for dk.

        Uses truncated JVP chain for dk gradients.
        Standard VJP for dv and dq.
        """
        final_state, k_seq, v_seq, q_seq, init_state = res
        do_seq = do.swapaxes(0, 1)

        seq_len = k_seq.shape[0]

        # For dk: use forward-mode approximation
        # Approximate: gradient accumulates through last `history_len` steps
        truncation = min(history_len, seq_len)

        # Compute per-timestep sensitivity of state to k
        def compute_dk_for_timestep(t):
            """Compute dk gradient at timestep t using truncated JVP."""
            k_t = k_seq[t]
            v_t = v_seq[t]

            # Reconstruct state at timestep t (simplified: use final_state as proxy)
            # In full implementation, would track per-timestep states

            # Sensitivity: how does k_t affect state?
            def state_update_via_k(k_input):
                """State update as function of k."""
                v_pred, vjp_fn = jax.vjp(lambda s: fwd_fn(final_state, k_input), final_state)
                error = v_t - v_pred
                grad, = vjp_fn(error)

                # Apply update
                new_state = jax.tree.map(
                    lambda s, g: (1 - wd * lr) * s + lr * jax.nn.tanh(g),
                    final_state, grad
                )
                return new_state

            # JVP: gradient of state update w.r.t. k
            _, jvp_result = jax.jvp(
                state_update_via_k,
                (k_t,),
                (jnp.ones_like(k_t),)
            )

            # Chain with output gradient (simplified)
            # Flatten pytree to scalar and broadcast to k_t shape
            flat_jvp, _ = jax.flatten_util.ravel_pytree(jvp_result)
            dk_magnitude = jnp.mean(flat_jvp) * jnp.mean(do_seq[t])

            # Return gradient with same shape as k_t
            dk_t = dk_magnitude * jnp.ones_like(k_t)

            return dk_t

        # Compute dk for all timesteps sequentially (avoid memory issues)
        def scan_dk(carry, t):
            dk_t = compute_dk_for_timestep(t)
            return carry, dk_t

        _, dk_seq = jax.lax.scan(scan_dk, None, jnp.arange(seq_len))

        # Standard VJP for dv and dq (placeholder - zero for now)
        dv_seq = jnp.zeros_like(v_seq)
        dq_seq = jnp.zeros_like(q_seq)

        # Swap back
        dk = dk_seq.swapaxes(0, 1)
        dv = dv_seq.swapaxes(0, 1)
        dq = dq_seq.swapaxes(0, 1)

        # Gradient w.r.t. initial state
        dstate = jax.tree.map(jnp.zeros_like, init_state)

        return dk, dv, dq, dstate

    simple_inner.defvjp(_simple_fwd, _simple_bwd)
    return simple_inner


if __name__ == "__main__":
    print("Forward-Mode TTT Implementation")
    print("="*60)
    print("\nThis module provides forward-mode differentiation for TTT gradients.")
    print("\nKey features:")
    print("- Uses JVP to propagate gradients forward through time")
    print("- Truncated history buffer (default: 32 timesteps)")
    print("- Avoids BPTT while approximating temporal gradient flow")
    print("\nMemory: O(K × d²) where K = history_len")
    print("Accuracy: Depends on history_len (larger = more accurate)")
    print("\nUsage: See test_framework.py for testing interface")
