from typing import Any

TRITON_ERROR = (
    "'triton' required. Please install the missing dependency via "
    '`pip install -U triton`.'
)


def load_triton() -> tuple[Any, Any]:
    try:
        import triton  # ty: ignore[unresolved-import]
        import triton.language as tl  # ty: ignore[unresolved-import]
    except ImportError as e:
        raise ImportError(TRITON_ERROR) from e

    return triton, tl


def has_triton() -> bool:
    try:
        load_triton()
    except ImportError:
        return False
    return True
