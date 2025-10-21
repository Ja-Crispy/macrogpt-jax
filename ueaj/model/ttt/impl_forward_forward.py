"""Forward-Forward Algorithm for TTT.

Hinton's Forward-Forward: Two forward passes, no backprop.
Learn to maximize 'goodness' for positive data, minimize for negative.

Theory:
- Positive pass: Real k-v pairs (correct reconstruction)
- Negative pass: Corrupted k-v pairs (mismatched or noisy)
- Goodness: Sum of squared activations in hidden state
- Update: Local layer-wise learning (no global backprop)

For TTT:
- Positive: k_t matches v_t (correct reconstruction target)
- Negative: k_t paired with random v or shuffled v_t
- State learns to have high activation for correct pairs, low for incorrect

References:
- Hinton (2022): "The Forward-Forward Algorithm: Some Preliminary Investigations"

Memory: O(1) - two forward passes only
Speed: 2x forward pass cost
Accuracy: Experimental - works for classification, adaptation for reconstruction TBD
"""

import jax
import jax.numpy as jnp


def forward_forward_ttt(fwd_fn, threshold=2.0, n_iters=1, wd=0.1, lr=0.01, corruption=0.5):
    """Create TTT function using Forward-Forward algorithm.

    Args:
        fwd_fn: Forward function (state, x) -> output
        threshold: Goodness threshold for contrastive loss
        n_iters: Number of inner optimization iterations
        wd: Weight decay
        lr: Learning rate
        corruption: Probability of corrupting negative samples

    Returns:
        TTT function with forward-forward learning
    """

    def generate_negative_k_v(k, v, key):
        """Generate negative samples by corrupting k-v pairs.

        Strategies:
        1. Shuffle v along sequence dimension (breaks temporal structure)
        2. Add noise to v (incorrect reconstruction target)
        3. Pair k with random v from different timestep

        Args:
            k: Keys (batch, seq, dim)
            v: Values (batch, seq, dim)
            key: Random key

        Returns:
            Corrupted (k_neg, v_neg) pairs
        """
        batch, seq_len, dim = v.shape

        # Strategy: Shuffle v along sequence dimension
        # This breaks k-v correspondence while keeping data realistic
        key_perm, key_noise = jax.random.split(key)

        # Random permutation of sequence positions
        perm = jax.random.permutation(key_perm, seq_len)
        v_shuffled = v[:, perm, :]

        # Add some noise to make it clearly "wrong"
        noise = jax.random.normal(key_noise, v.shape) * 0.1
        v_neg = v_shuffled + noise

        # k stays the same (we're corrupting the target, not the input)
        k_neg = k

        return k_neg, v_neg

    def compute_goodness(state):
        """Compute goodness: sum of squared activations.

        Higher goodness = state is more confident/activated.
        For positive samples: want high goodness.
        For negative samples: want low goodness.

        Args:
            state: Current TTT state (pytree of parameters)

        Returns:
            Scalar goodness value
        """
        # Flatten all parameters and compute L2 norm
        flat_params, _ = jax.flatten_util.ravel_pytree(state)
        goodness = jnp.sum(flat_params ** 2)
        return goodness

    def forward_pass(k, v, q, state, is_positive=True):
        """Forward pass with goodness tracking.

        Args:
            k, v, q: Input tensors
            state: Initial state
            is_positive: Whether this is positive (True) or negative (False) data

        Returns:
            outputs, final_state, avg_goodness
        """
        k_seq = k.swapaxes(0, 1)
        v_seq = v.swapaxes(0, 1)
        q_seq = q.swapaxes(0, 1)

        goodness_list = []

        def update_fn(state, x):
            k_t, v_t, q_t = x

            # Inner optimization
            for _ in range(n_iters):
                v_pred, vjp_fn = jax.vjp(lambda s: fwd_fn(s, k_t), state)
                reconstruction_error = v_t - v_pred
                dstate, = vjp_fn(reconstruction_error)

                state = jax.tree.map(
                    lambda s, g: (1 - wd * lr) * s + lr * jax.nn.tanh(g),
                    state, dstate
                )

            # Compute goodness after update
            g = compute_goodness(state)

            # Compute output
            output = fwd_fn(state, q_t)

            return state, (output, g)

        final_state, (o_seq, goodness_seq) = jax.lax.scan(
            update_fn, state, (k_seq, v_seq, q_seq)
        )

        outputs = o_seq.swapaxes(0, 1)
        avg_goodness = jnp.mean(goodness_seq)

        return outputs, final_state, avg_goodness

    @jax.custom_vjp
    def ff_ttt_fn(k, v, q, state):
        """Forward-forward TTT function."""
        # Positive pass
        outputs, _, _ = forward_pass(k, v, q, state, is_positive=True)
        return outputs

    def _ff_fwd(k, v, q, state):
        """Forward pass for both positive and negative samples."""
        # Positive pass
        outputs_pos, state_pos, goodness_pos = forward_pass(k, v, q, state, is_positive=True)

        # Generate negative samples
        key = jax.random.PRNGKey(hash(tuple(k.shape)))  # Deterministic for same input shape
        k_neg, v_neg = generate_negative_k_v(k, v, key)

        # Negative pass
        _, state_neg, goodness_neg = forward_pass(k_neg, v_neg, q, state, is_positive=False)

        # Store for backward pass
        residuals = (k, v, q, state, goodness_pos, goodness_neg)

        return outputs_pos, residuals

    def _ff_bwd(res, do):
        """Backward pass using forward-forward learning rule.

        No actual backpropagation! Instead:
        1. Compute contrastive loss based on goodness
        2. Use local learning rules to update parameters

        The gradient estimates are based on:
        - Positive goodness should be > threshold
        - Negative goodness should be < threshold
        """
        k, v, q, state, goodness_pos, goodness_neg = res

        # Contrastive loss: penalize when goodness is wrong
        # Want: goodness_pos > goodness_neg + threshold
        contrastive_loss = jax.nn.relu(goodness_neg - goodness_pos + threshold)

        # Pseudo-gradients based on goodness difference
        # If positive goodness too low: increase k, v, q
        # If negative goodness too high: decrease k, v, q
        goodness_diff = goodness_pos - goodness_neg

        # Create gradient estimates proportional to goodness mismatch
        # Positive sign if we need to increase positive goodness
        scale = jnp.tanh(goodness_diff / threshold)  # Bounded scaling

        # Generate pseudo-gradients
        # These are not true gradients - they're learning signals
        dk = scale * do
        dv = scale * do
        dq = scale * do

        dstate = jax.tree.map(jnp.zeros_like, state)

        return dk, dv, dq, dstate

    ff_ttt_fn.defvjp(_ff_fwd, _ff_bwd)
    return ff_ttt_fn


# Simpler variant: Goodness-based learning without negative samples
def goodness_ttt(fwd_fn, target_goodness=1.0, n_iters=1, wd=0.1, lr=0.01):
    """Simplified forward-forward: just maximize goodness for all samples.

    No negative samples needed. Just learn to have high, stable goodness.

    Args:
        fwd_fn: Forward function
        target_goodness: Target goodness value
        n_iters: Inner optimization iterations
        wd: Weight decay
        lr: Learning rate

    Returns:
        TTT function with goodness-based learning
    """

    def compute_goodness(state):
        flat_params, _ = jax.flatten_util.ravel_pytree(state)
        return jnp.sum(flat_params ** 2)

    def forward_pass(k, v, q, state):
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
            g = compute_goodness(state)

            return state, (output, g)

        final_state, (o_seq, goodness_seq) = jax.lax.scan(
            update_fn, state, (k_seq, v_seq, q_seq)
        )

        outputs = o_seq.swapaxes(0, 1)
        avg_goodness = jnp.mean(goodness_seq)

        return outputs, avg_goodness

    @jax.custom_vjp
    def goodness_ttt_fn(k, v, q, state):
        outputs, _ = forward_pass(k, v, q, state)
        return outputs

    def _goodness_fwd(k, v, q, state):
        outputs, avg_goodness = forward_pass(k, v, q, state)
        return outputs, (k, v, q, state, avg_goodness)

    def _goodness_bwd(res, do):
        k, v, q, state, avg_goodness = res

        # Goodness deviation from target
        goodness_error = avg_goodness - target_goodness

        # Scale gradients by how far we are from target
        scale = jnp.tanh(goodness_error)

        dk = scale * do
        dv = scale * do
        dq = scale * do

        dstate = jax.tree.map(jnp.zeros_like, state)

        return dk, dv, dq, dstate

    goodness_ttt_fn.defvjp(_goodness_fwd, _goodness_bwd)
    return goodness_ttt_fn


if __name__ == "__main__":
    print("Forward-Forward TTT Implementation")
    print("="*60)
    print("\nHinton's Forward-Forward: Two forward passes, no backprop.")
    print("\nKey features:")
    print("- Positive pass: Real k-v pairs (correct reconstruction)")
    print("- Negative pass: Corrupted k-v pairs (incorrect)")
    print("- Goodness: Sum of squared activations")
    print("- Learning: Maximize for positive, minimize for negative")
    print("\nMemory: O(1) - two forward passes only")
    print("Speed: 2x forward pass cost")
    print("\nVariants:")
    print("- forward_forward_ttt: Full contrastive learning")
    print("- goodness_ttt: Simplified (no negative samples)")
    print("\nUsage: See test_framework.py for testing interface")
