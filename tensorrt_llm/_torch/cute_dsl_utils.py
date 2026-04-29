import platform

from ..logger import logger

IS_CUTLASS_DSL_AVAILABLE = False

if platform.system() != "Windows":
    try:
        import cutlass  # noqa
        import cutlass.cute as cute  # noqa

        # Probe a deep cute_dsl_kernels submodule so that cutlass._mlir.* paths
        # (which custom_ops/cute_dsl_custom_ops.py eagerly imports at module load)
        # are validated up front. Catches half-installed cutlass-dsl where the
        # split-package RECORD overlap clobbers .pth/.so before trtllm crashes.
        from .cute_dsl_kernels.blackwell.utils import make_ptr  # noqa: F401
        logger.info(f"cutlass dsl is available")
        IS_CUTLASS_DSL_AVAILABLE = True
    except ImportError:
        pass
