from typing import Any

try:
    import triton
    import triton.language as tl

    major_triton_version = int(triton.__version__.split('.')[0])
    if major_triton_version < 2:
        raise ImportError("'triton>=2.0.0' required")

except ImportError:

    class TritonJit:
        def __init__(self, func_name: str):
            self.func_name = func_name

        def report_error(self):
            raise ValueError(f"Could not compile function '{self.func_name}' "
                             f"since 'triton>=2.0.0' dependency was not "
                             f"found. Please install the missing dependency "
                             f"via `pip install -U -pre triton`.")

        def __call__(self, *args, **kwargs):
            self.report_error()

        def __getitem__(self, *args, **kwargs):
            self.report_error()

    triton = type('triton', (object, ), {})()
    triton.jit = lambda func: TritonJit(func.__name__)

    tl = type('tl', (object, ), {})()
    tl.constexpr = Any
