"""Test-Time Training (TTT) module implementation.

This module implements the TTT algorithm which learns to adapt a hidden state
during inference using gradient descent on a self-supervised objective.

Available gradient methods:

**Backprop-based** (failed - see ALTERNATIVE_METHODS_FAILURE_ANALYSIS.md):
- surrogate: Original surrogate gradient (dk fails)
- bptt: Full BPTT (ground truth, expensive)
- implicit: Implicit differentiation (ill-conditioned Hessian → divergence)
- hybrid: Hybrid chunking (partial success but gradient interference)
- forward: Forward-mode with truncated history (returns zeros)

**Alternative Learning Mechanisms** (no backprop):
- feedback: Feedback alignment (fixed random feedback matrices)
- dfa: Direct feedback alignment (single matrix for all inputs)
- forward_forward: Forward-Forward algorithm (contrastive learning)
- goodness: Goodness-based learning (simplified forward-forward)
- perturbation: Activity perturbation (SLOW - finite differences)
- spsa: Simultaneous perturbation (faster but still slow)
"""

from .module import TTTModel
from .impl import ttt, make_scan_fn, make_reverse_fn
from .impl_implicit import implicit_ttt
from .impl_hybrid import hybrid_ttt
from .impl_forward import forward_ttt
from .impl_feedback import feedback_alignment_ttt, direct_feedback_alignment_ttt
from .impl_forward_forward import forward_forward_ttt, goodness_ttt
from .impl_perturbation import perturbation_ttt, simultaneous_perturbation_ttt
from .test_framework import TTTGradientTester

__all__ = [
	'TTTModel',
	'ttt',
	'make_scan_fn',
	'make_reverse_fn',
	'implicit_ttt',
	'hybrid_ttt',
	'forward_ttt',
	'feedback_alignment_ttt',
	'direct_feedback_alignment_ttt',
	'forward_forward_ttt',
	'goodness_ttt',
	'perturbation_ttt',
	'simultaneous_perturbation_ttt',
	'TTTGradientTester',
]