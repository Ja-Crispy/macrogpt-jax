"""Test-Time Training (TTT) module implementation.

This module implements the TTT algorithm which learns to adapt a hidden state
during inference using gradient descent on a self-supervised objective.

Available gradient methods:
- surrogate: Original surrogate gradient (dk fails)
- bptt: Full BPTT (ground truth, expensive)
- implicit: Implicit differentiation (avoids BPTT)
- hybrid: Hybrid chunking (BPTT within blocks)
"""

from .module import TTTModel
from .impl import ttt, make_scan_fn, make_reverse_fn
from .impl_implicit import implicit_ttt
from .impl_hybrid import hybrid_ttt
from .test_framework import TTTGradientTester

__all__ = [
	'TTTModel',
	'ttt',
	'make_scan_fn',
	'make_reverse_fn',
	'implicit_ttt',
	'hybrid_ttt',
	'TTTGradientTester',
]