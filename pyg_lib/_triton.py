from typing import Any

triton: Any
tl: Any

try:
    import triton as _triton
    import triton.language as _tl

    major_triton_version = int(_triton.__version__.split('.')[0])
    if major_triton_version < 2:
        raise ImportError("'triton>=2.0.0' required")

except ImportError:

    class TritonJit:
        def __init__(self, func_name: str):
            self.func_name = func_name

        def report_error(self):
            raise ValueError(
                f"Could not compile function '{self.func_name}' "
                f"since 'triton>=2.0.0' dependency was not "
                f'found. Please install the missing dependency '
                f'via `pip install -U -pre triton`.',
            )

        def __call__(self, *args, **kwargs):
            self.report_error()

        def __getitem__(self, *args, **kwargs):
            self.report_error()

    class Triton:
        @staticmethod
        def jit(func):
            return TritonJit(func.__name__)

        @staticmethod
        def cdiv(*args, **kwargs):
            raise ValueError("'triton>=2.0.0' required")

    class TL:
        constexpr = Any

    triton = Triton()
    tl = TL()

else:
    triton = _triton
    tl = _tl
