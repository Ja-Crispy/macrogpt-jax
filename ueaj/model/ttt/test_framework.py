"""Test framework for TTT gradient methods.

Provides consistent testing interface for comparing different gradient computation methods:
- Surrogate (original)
- Implicit differentiation
- Hybrid chunking
- Forward-mode
- Full BPTT (ground truth)
"""

import jax
import jax.numpy as jnp
from flax import nnx
import os
import numpy as np


class TTTGradientTester:
    """Unified testing for all TTT gradient implementations."""

    def __init__(self, seq_len=500, hidden_d=512, batch_size=4, model_d=128, seed=42):
        """Initialize tester with configuration matching ueaj's setup.

        Args:
            seq_len: Sequence length for testing
            hidden_d: Hidden dimension (TTT state size)
            batch_size: Batch size
            model_d: Model dimension (input/output)
            seed: Random seed for reproducibility
        """
        self.seq_len = seq_len
        self.hidden_d = hidden_d
        self.batch_size = batch_size
        self.model_d = model_d
        self.seed = seed

    def generate_synthetic_data(self, seed=None):
        """Generate random k, v, q for testing.

        Args:
            seed: Override default seed if provided

        Returns:
            Tuple of (k, v, q) arrays
        """
        if seed is None:
            seed = self.seed

        key = jax.random.PRNGKey(seed)
        k_key, v_key, q_key = jax.random.split(key, 3)

        k = jax.random.normal(k_key, (self.batch_size, self.seq_len, self.hidden_d))
        v = jax.random.normal(v_key, (self.batch_size, self.seq_len, self.hidden_d))
        q = jax.random.normal(q_key, (self.batch_size, self.seq_len, self.hidden_d))

        return k, v, q

    def compute_gradients(self, model, k, v, q, grad_output):
        """Compute gradients w.r.t. k, v, q using VJP.

        Args:
            model: TTTModel instance
            k, v, q: Input tensors
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Tuple of (grad_k, grad_v, grad_q)
        """
        def forward(k, v, q):
            return model.apply_ttt(k, v, q)

        output, vjp_fn = jax.vjp(forward, k, v, q)
        grad_k, grad_v, grad_q = vjp_fn(grad_output)

        return grad_k, grad_v, grad_q

    def cosine_similarity_per_token(self, grad_true, grad_method):
        """Compute cosine similarity per token position (averaged over batch).

        Args:
            grad_true: Ground truth gradient (batch, seq_len, dim)
            grad_method: Method gradient (batch, seq_len, dim)

        Returns:
            Cosine similarity per token (seq_len,)
        """
        # Compute per batch element
        dot_product = jnp.sum(grad_true * grad_method, axis=2)  # (batch, seq)
        norm_true = jnp.linalg.norm(grad_true, axis=2)
        norm_method = jnp.linalg.norm(grad_method, axis=2)
        cosine_sim = dot_product / (norm_true * norm_method + 1e-8)
        return cosine_sim.mean(axis=0)  # (seq,)

    def rms_error_per_token(self, grad_true, grad_method):
        """Compute RMS error per token position.

        Args:
            grad_true: Ground truth gradient
            grad_method: Method gradient

        Returns:
            RMS error per token (seq_len,)
        """
        diff = grad_true - grad_method
        rms_per_batch = jnp.sqrt(jnp.mean(diff**2, axis=2))
        return rms_per_batch.mean(axis=0)

    def check_descent_direction(self, grad_true, grad_method):
        """Check if method gradient is valid descent direction (d·g > 0).

        Args:
            grad_true: Ground truth gradient
            grad_method: Method gradient

        Returns:
            Boolean: True if valid descent direction
        """
        dot_product = jnp.sum(grad_true * grad_method)
        return bool(dot_product > 0)

    def check_numerical_stability(self, *grads):
        """Check for NaN/Inf in gradients.

        Args:
            *grads: Variable number of gradient tensors

        Returns:
            Dict with NaN/Inf flags per gradient
        """
        results = {}
        for i, grad in enumerate(grads):
            results[f'grad_{i}_has_nan'] = bool(jnp.isnan(grad).any())
            results[f'grad_{i}_has_inf'] = bool(jnp.isinf(grad).any())
        return results

    def test_method(self, method_model, true_model, name="method", verbose=True):
        """Test a single gradient method against ground truth BPTT.

        Args:
            method_model: TTTModel with the method to test
            true_model: TTTModel with surrogate=False (ground truth BPTT)
            name: Method name for logging
            verbose: Print progress messages

        Returns:
            Tuple of (metrics_dict, summary_dict)
        """
        if verbose:
            print(f"\nTesting method: {name}")
            print("="*60)

        # Generate data
        k, v, q = self.generate_synthetic_data()
        grad_output = jax.random.normal(
            jax.random.PRNGKey(456),
            (self.batch_size, self.seq_len, self.hidden_d)
        )

        if verbose:
            print("Computing ground truth gradients (BPTT)...")

        # Ground truth gradients
        true_dk, true_dv, true_dq = self.compute_gradients(true_model, k, v, q, grad_output)

        if verbose:
            print(f"Computing {name} gradients...")

        # Method gradients
        try:
            method_dk, method_dv, method_dq = self.compute_gradients(
                method_model, k, v, q, grad_output
            )
        except Exception as e:
            print(f"✗ Error computing gradients: {e}")
            return None, None

        if verbose:
            print("Computing metrics...")

        # Check numerical stability
        stability = self.check_numerical_stability(method_dk, method_dv, method_dq)
        if any(stability.values()):
            print(f"⚠ Warning: Numerical instability detected!")
            for key, val in stability.items():
                if val:
                    print(f"  {key}: {val}")

        # Compute metrics per token
        metrics = {
            'dk_cosine_sim': self.cosine_similarity_per_token(true_dk, method_dk),
            'dv_cosine_sim': self.cosine_similarity_per_token(true_dv, method_dv),
            'dq_cosine_sim': self.cosine_similarity_per_token(true_dq, method_dq),

            'dk_rms': self.rms_error_per_token(true_dk, method_dk),
            'dv_rms': self.rms_error_per_token(true_dv, method_dv),
            'dq_rms': self.rms_error_per_token(true_dq, method_dq),

            'dk_descent': self.check_descent_direction(true_dk, method_dk),
            'dv_descent': self.check_descent_direction(true_dv, method_dv),
            'dq_descent': self.check_descent_direction(true_dq, method_dq),
        }
        metrics.update(stability)

        # Summary statistics
        summary = {
            'name': name,
            'dk_cosine_mean': float(metrics['dk_cosine_sim'].mean()),
            'dk_cosine_min': float(metrics['dk_cosine_sim'].min()),
            'dk_cosine_max': float(metrics['dk_cosine_sim'].max()),
            'dk_cosine_std': float(metrics['dk_cosine_sim'].std()),

            'dv_cosine_mean': float(metrics['dv_cosine_sim'].mean()),
            'dv_cosine_min': float(metrics['dv_cosine_sim'].min()),
            'dv_cosine_max': float(metrics['dv_cosine_sim'].max()),

            'dq_rms_mean': float(metrics['dq_rms'].mean()),
            'dq_rms_max': float(metrics['dq_rms'].max()),

            'all_descent_valid': (
                metrics['dk_descent'] and
                metrics['dv_descent'] and
                metrics['dq_descent']
            ),
            'numerically_stable': not any(stability.values()),
        }

        if verbose:
            print("\nResults:")
            print(f"  dk cosine sim: [{summary['dk_cosine_min']:.4f}, {summary['dk_cosine_max']:.4f}], mean={summary['dk_cosine_mean']:.4f}")
            print(f"  dv cosine sim: [{summary['dv_cosine_min']:.4f}, {summary['dv_cosine_max']:.4f}], mean={summary['dv_cosine_mean']:.4f}")
            print(f"  dq RMS error:  mean={summary['dq_rms_mean']:.6f}, max={summary['dq_rms_max']:.6f}")
            print(f"  Valid descent: {summary['all_descent_valid']}")
            print(f"  Stable: {summary['numerically_stable']}")

            # Success assessment
            if summary['dk_cosine_mean'] > 0.5:
                print(f"\n✓ SUCCESS: {name} achieves target dk cosine_sim > 0.5")
            elif summary['dk_cosine_mean'] > 0.2:
                print(f"\n⚠ PARTIAL: {name} shows improvement but below target")
            else:
                print(f"\n✗ FAILURE: {name} does not improve dk gradients")

        return metrics, summary

    def compare_methods(self, methods_dict, save_dir='./ttt_results'):
        """Compare multiple gradient methods and generate visualizations.

        Args:
            methods_dict: Dict of {'method_name': (method_model, true_model), ...}
            save_dir: Directory to save results

        Returns:
            Tuple of (all_results, all_summaries)
        """
        import matplotlib.pyplot as plt

        os.makedirs(save_dir, exist_ok=True)

        all_results = {}
        all_summaries = []

        print("\n" + "="*60)
        print("COMPARING GRADIENT METHODS")
        print("="*60)

        # Test each method
        for name, (method_model, true_model) in methods_dict.items():
            metrics, summary = self.test_method(method_model, true_model, name)
            if metrics is not None:
                all_results[name] = metrics
                all_summaries.append(summary)

        if not all_results:
            print("\n✗ No methods succeeded")
            return None, None

        # Generate comparison plots
        print("\nGenerating comparison plots...")
        self.plot_comparison(all_results, save_dir)

        # Save summary table
        self.save_summary_table(all_summaries, save_dir)

        print(f"\n✓ Results saved to {save_dir}/")

        return all_results, all_summaries

    def plot_comparison(self, all_results, save_dir):
        """Generate comparison plots for all methods."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('TTT Gradient Methods Comparison', fontsize=14, fontweight='bold')

        tokens = np.arange(self.seq_len)

        # Plot dk cosine similarity
        for name, metrics in all_results.items():
            axes[0].plot(tokens, metrics['dk_cosine_sim'], label=name, linewidth=2, alpha=0.8)
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2, label='Random', alpha=0.5)
        axes[0].axhline(y=0.5, color='g', linestyle=':', linewidth=2, label='Target', alpha=0.5)
        axes[0].set_xlabel('Token Position')
        axes[0].set_ylabel('Cosine Similarity')
        axes[0].set_title('Gradient dk')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-1.05, 1.05)

        # Plot dv cosine similarity
        for name, metrics in all_results.items():
            axes[1].plot(tokens, metrics['dv_cosine_sim'], label=name, linewidth=2, alpha=0.8)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2, label='Random', alpha=0.5)
        axes[1].set_xlabel('Token Position')
        axes[1].set_ylabel('Cosine Similarity')
        axes[1].set_title('Gradient dv')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(-1.05, 1.05)

        # Plot dq RMS error
        for name, metrics in all_results.items():
            axes[2].plot(tokens, metrics['dq_rms'], label=name, linewidth=2, alpha=0.8)
        axes[2].set_xlabel('Token Position')
        axes[2].set_ylabel('RMS Error')
        axes[2].set_title('Gradient dq')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_yscale('log')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {save_dir}/comparison.png")

    def save_summary_table(self, summaries, save_dir):
        """Save summary statistics as CSV and text."""
        import pandas as pd

        df = pd.DataFrame(summaries)
        df.to_csv(f'{save_dir}/summary.csv', index=False)

        # Human-readable text
        with open(f'{save_dir}/summary.txt', 'w') as f:
            f.write("TTT Gradient Methods Summary\n")
            f.write("="*60 + "\n\n")
            for summary in summaries:
                f.write(f"Method: {summary['name']}\n")
                f.write(f"  dk cosine sim: {summary['dk_cosine_mean']:.4f} "
                       f"[{summary['dk_cosine_min']:.4f}, {summary['dk_cosine_max']:.4f}] "
                       f"(std={summary['dk_cosine_std']:.4f})\n")
                f.write(f"  dv cosine sim: {summary['dv_cosine_mean']:.4f}\n")
                f.write(f"  dq RMS error:  {summary['dq_rms_mean']:.6f} (max={summary['dq_rms_max']:.6f})\n")
                f.write(f"  Valid descent: {summary['all_descent_valid']}\n")
                f.write(f"  Stable: {summary['numerically_stable']}\n")

                # Assessment
                if summary['dk_cosine_mean'] > 0.5:
                    f.write(f"  → SUCCESS ✓\n")
                elif summary['dk_cosine_mean'] > 0.2:
                    f.write(f"  → PARTIAL ⚠\n")
                else:
                    f.write(f"  → FAILURE ✗\n")
                f.write("\n")

        print(f"  Saved: {save_dir}/summary.txt")
        print(f"  Saved: {save_dir}/summary.csv")


# Usage example
if __name__ == "__main__":
    print("TTT Gradient Test Framework")
    print("="*60)
    print("\nUsage:")
    print("""
    from ueaj.model.ttt.test_framework import TTTGradientTester
    from ueaj.model.ttt import TTTModel
    from ueaj.model import GMLP
    from flax.nnx import rnglib as rng

    # Create tester
    tester = TTTGradientTester(seq_len=200, hidden_d=512)

    # Create models
    model_surrogate = TTTModel(128, 512, GMLP, surrogate=True, rngs=rng.Rngs(42))
    model_bptt = TTTModel(128, 512, GMLP, surrogate=False, rngs=rng.Rngs(42))

    # Test single method
    metrics, summary = tester.test_method(model_surrogate, model_bptt, "Surrogate")

    # Or compare multiple methods
    methods = {
        'Surrogate': (model_surrogate, model_bptt),
        # 'Implicit': (model_implicit, model_bptt),
        # 'Hybrid': (model_hybrid, model_bptt),
    }
    results, summaries = tester.compare_methods(methods)
    """)
