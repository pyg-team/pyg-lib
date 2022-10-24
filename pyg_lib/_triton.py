try:
    import triton
    import triton.language as tl
except ImportError:

    class TritonJit:
        def __init__(self, func_name: str):
            self.func_name = func_name

        def report_error(self):
            raise ValueError(f"Could not compile function '{self.func_name}' "
                             f"since 'triton' dependency was not found. "
                             f"Please install the missing dependency via "
                             f"`pip install triton`.")

        def __call__(self, *args, **kwargs):
            self.report_error()

        def __getitem__(self, *args, **kwargs):
            self.report_error()

    triton = type('triton', (object, ), {})()
    triton.jit = lambda func: TritonJit(func.__name__)
    tl = None
