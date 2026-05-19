"""
Regime-aware density profiling for AFES.

The package sits under ``signal/regime_memory`` to match the project layout,
but it is intended to be imported by adding ``AFES/signal`` to ``sys.path``
and then importing ``regime_memory``. This avoids collisions with Python's
stdlib ``signal`` module while preserving the requested directory structure.
"""

from .config import RegimeMemoryConfig
from .decayed_density_estimator import DensityProfilerBase, LegacyExpandingWindowProfiler
from .profile_manager import RegimeAwareProfileManager


def build_profiler(
    config: RegimeMemoryConfig,
    use_regime_memory: bool = True,
) -> DensityProfilerBase:
    """
    Build a density profiler behind a feature flag.

    This factory preserves one integration point for A/B testing. Downstream
    code can ask for a profiler once and switch between legacy expanding memory
    and regime-aware memory without branching inside the execution logic.
    """

    if use_regime_memory:
        return RegimeAwareProfileManager(config)
    return LegacyExpandingWindowProfiler(config)


__all__ = [
    "RegimeMemoryConfig",
    "DensityProfilerBase",
    "LegacyExpandingWindowProfiler",
    "RegimeAwareProfileManager",
    "build_profiler",
]
