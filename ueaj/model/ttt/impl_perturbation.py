"""Activity Perturbation for TTT gradients.

Estimate gradients via finite differences with noise injection.
No backprop needed - just forward passes with perturbed states.

Theory:
- Add small noise ε to state: s' = s + ε
- Measure output change: ΔL = L(s') - L(s)
- Gradient estimate: ∇L ≈ (ΔL / ||ε||) * ε
- Average over multiple samples for stability

For TTT:
- Perturb state before each inner optimization step
- Measure reconstruction loss change
- Estimate state gradients from perturbations
- Propagate to inputs via chain rule approximation

References:
- Williams (1992): "Simple statistical gradient-following algorithms for connectionist reinforcement learning"
- Seung (2003): "Learning in spiking neural networks by reinforcement of stochastic synaptic transmission"

Memory: O(1) - no computation graph
Speed: SLOW - O(n_samples) forward passes per update
Accuracy: Depends on n_samples and epsilon; high variance
"""

import jax
import jax.numpy as jnp


def perturbation_ttt(fwd_fn, n_samples=10, epsilon=0.01, n_iters=1, wd=0.1, lr=0.01):
    """Create TTT function using activity perturbation for gradients.

    WARNING: Very slow! Requires n_samples forward passes per gradient estimate.
    For hidden_d=512, expect ~10-100x slower than backprop.

    Args:
        fwd_fn: Forward function (state, x) -> output
        n_samples: Number of perturbation samples (more = less variance, slower)
        epsilon: Perturbation magnitude (smaller = more accurate, noisier)
        n_iters: Inner optimization iterations
        wd: Weight decay
        lr: Learning rate

    Returns:
        TTT function with perturbation-based gradients
    """

    def forward_pass(k, v, q, state):
        """Standard forward pass."""
        k_seq = k.swapaxes(0, 1)
        v_seq = v.swapaxes(0, 1)
        q_seq = q.swapaxes(0, 1)

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

            output = fwd_fn(state, q_t)
            return state, output

        final_state, o_seq = jax.lax.scan(update_fn, state, (k_seq, v_seq, q_seq))
        outputs = o_seq.swapaxes(0, 1)

        return outputs, (final_state, k_seq, v_seq, q_seq, state)

    @jax.custom_vjp
    def perturb_ttt_fn(k, v, q, state):
        return forward_pass(k, v, q, state)[0]

    def _perturb_fwd(k, v, q, state):
        return forward_pass(k, v, q, state)

    def _perturb_bwd(res, do):
        """Backward pass using activity perturbation.

        For each input (k, v, q):
        1. Compute baseline loss
        2. Add noise, recompute loss
        3. Estimate gradient from loss difference
        4. Average over n_samples
        """
        final_state, k_seq, v_seq, q_seq, init_state = res
        do_seq = do.swapaxes(0, 1)  # (seq, batch, dim)

        seq_len = k_seq.shape[0]

        # Baseline: compute loss with current state
        def compute_loss_for_sequence(k_in, v_in, q_in):
            """Compute total reconstruction loss for a sequence."""
            outputs, _ = forward_pass(
                k_in.swapaxes(0, 1),
                v_in.swapaxes(0, 1),
                q_in.swapaxes(0, 1),
                init_state
            )
            # Reconstruction loss (simplified: MSE with do as target)
            loss = jnp.sum((outputs.swapaxes(0, 1) - do_seq) ** 2)
            return loss

        baseline_loss = compute_loss_for_sequence(k_seq, v_seq, q_seq)

        # Estimate gradients via perturbation
        def estimate_gradient_for_input(input_seq, input_name):
            """Estimate gradient for k, v, or q via perturbation.

            Args:
                input_seq: Input tensor (seq, batch, dim)
                input_name: 'k', 'v', or 'q'

            Returns:
                Gradient estimate (seq, batch, dim)
            """
            key = jax.random.PRNGKey(hash(input_name))
            grad_estimate = jnp.zeros_like(input_seq)

            for i in range(n_samples):
                # Generate perturbation
                key, subkey = jax.random.split(key)
                perturbation = epsilon * jax.random.normal(subkey, input_seq.shape)

                # Perturbed input
                input_perturbed = input_seq + perturbation

                # Compute loss with perturbed input
                if input_name == 'k':
                    perturbed_loss = compute_loss_for_sequence(input_perturbed, v_seq, q_seq)
                elif input_name == 'v':
                    perturbed_loss = compute_loss_for_sequence(k_seq, input_perturbed, q_seq)
                else:  # 'q'
                    perturbed_loss = compute_loss_for_sequence(k_seq, v_seq, input_perturbed)

                # Gradient estimate: (ΔL / ε) * perturbation
                loss_diff = perturbed_loss - baseline_loss
                grad_contribution = (loss_diff / epsilon) * perturbation

                # Accumulate
                grad_estimate += grad_contribution / n_samples

            return grad_estimate

        # Estimate gradients for k, v, q
        dk_seq = estimate_gradient_for_input(k_seq, 'k')
        dv_seq = estimate_gradient_for_input(v_seq, 'v')
        dq_seq = estimate_gradient_for_input(q_seq, 'q')

        # Swap back to (batch, seq, dim)
        dk = dk_seq.swapaxes(0, 1)
        dv = dv_seq.swapaxes(0, 1)
        dq = dq_seq.swapaxes(0, 1)

        # Gradient w.r.t. initial state
        dstate = jax.tree.map(jnp.zeros_like, init_state)

        return dk, dv, dq, dstate

    perturb_ttt_fn.defvjp(_perturb_fwd, _perturb_bwd)
    return perturb_ttt_fn


# Faster variant: Simultaneous perturbation
def simultaneous_perturbation_ttt(fwd_fn, n_samples=5, epsilon=0.01, n_iters=1, wd=0.1, lr=0.01):
    """Simultaneous Perturbation Stochastic Approximation (SPSA).

    Faster than standard perturbation: perturb ALL parameters at once.
    Fewer samples needed, but noisier gradient estimates.

    Args:
        fwd_fn: Forward function
        n_samples: Number of perturbation samples (can be smaller than standard)
        epsilon: Perturbation magnitude
        n_iters: Inner optimization iterations
        wd: Weight decay
        lr: Learning rate

    Returns:
        TTT function with SPSA gradients
    """

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
            return state, output

        final_state, o_seq = jax.lax.scan(update_fn, state, (k_seq, v_seq, q_seq))
        outputs = o_seq.swapaxes(0, 1)

        return outputs, (final_state, k_seq, v_seq, q_seq, state)

    @jax.custom_vjp
    def spsa_ttt_fn(k, v, q, state):
        return forward_pass(k, v, q, state)[0]

    def _spsa_fwd(k, v, q, state):
        return forward_pass(k, v, q, state)

    def _spsa_bwd(res, do):
        """SPSA: Perturb all inputs simultaneously."""
        final_state, k_seq, v_seq, q_seq, init_state = res
        do_seq = do.swapaxes(0, 1)

        # Baseline loss
        def compute_loss(k_in, v_in, q_in):
            outputs, _ = forward_pass(
                k_in.swapaxes(0, 1),
                v_in.swapaxes(0, 1),
                q_in.swapaxes(0, 1),
                init_state
            )
            return jnp.sum((outputs.swapaxes(0, 1) - do_seq) ** 2)

        baseline_loss = compute_loss(k_seq, v_seq, q_seq)

        # Simultaneous perturbation
        key = jax.random.PRNGKey(42)
        grad_k = jnp.zeros_like(k_seq)
        grad_v = jnp.zeros_like(v_seq)
        grad_q = jnp.zeros_like(q_seq)

        for i in range(n_samples):
            # Generate random perturbation directions (Rademacher: ±1)
            key, *subkeys = jax.random.split(key, 4)

            delta_k = 2 * jax.random.randint(subkeys[0], k_seq.shape, 0, 2) - 1
            delta_v = 2 * jax.random.randint(subkeys[1], v_seq.shape, 0, 2) - 1
            delta_q = 2 * jax.random.randint(subkeys[2], q_seq.shape, 0, 2) - 1

            # Perturb all inputs simultaneously
            k_plus = k_seq + epsilon * delta_k
            v_plus = v_seq + epsilon * delta_v
            q_plus = q_seq + epsilon * delta_q

            # Measure loss change
            perturbed_loss = compute_loss(k_plus, v_plus, q_plus)
            loss_diff = perturbed_loss - baseline_loss

            # SPSA gradient estimate
            grad_k += (loss_diff / epsilon) * delta_k / n_samples
            grad_v += (loss_diff / epsilon) * delta_v / n_samples
            grad_q += (loss_diff / epsilon) * delta_q / n_samples

        dk = grad_k.swapaxes(0, 1)
        dv = grad_v.swapaxes(0, 1)
        dq = grad_q.swapaxes(0, 1)

        dstate = jax.tree.map(jnp.zeros_like, init_state)

        return dk, dv, dq, dstate

    spsa_ttt_fn.defvjp(_spsa_fwd, _spsa_bwd)
    return spsa_ttt_fn


if __name__ == "__main__":
    print("Activity Perturbation TTT Implementation")
    print("="*60)
    print("\nEstimate gradients via finite differences with noise injection.")
    print("\nKey features:")
    print("- Add noise to inputs, measure loss change")
    print("- Gradient estimate: (ΔL / ε) * noise")
    print("- No backprop needed")
    print("\nMemory: O(1) - no computation graph")
    print("Speed: VERY SLOW - O(n_samples) forward passes")
    print("Accuracy: High variance, needs many samples")
    print("\nVariants:")
    print("- perturbation_ttt: Per-input perturbation (accurate but slow)")
    print("- simultaneous_perturbation_ttt: SPSA (faster but noisier)")
    print("\nWARNING: Expect 10-100x slowdown vs backprop!")
    print("\nUsage: See test_framework.py for testing interface")
