import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

extra_compile_args = {
    'cxx': [
        '-O2',
        '-DPy_LIMITED_API=0x03090000',
        '-DTORCH_TARGET_VERSION=0x020a000000000000',
    ],
}

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='pyg_stable_abi_example',
    version='0.0.1',
    ext_modules=[
        CppExtension(
            '_C',
            sources=[os.path.join(this_dir, 'csrc', 'ops.cpp')],
            extra_compile_args=extra_compile_args,
            py_limited_api=True,
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    options={'bdist_wheel': {'py_limited_api': 'cp39'}},
)
