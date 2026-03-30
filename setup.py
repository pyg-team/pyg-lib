# Environment flags to control different options
#
# - USE_MKL_BLAS=1:
#   Enables use of MKL BLAS (requires PyTorch to be built with MKL support)

import importlib
import os
import os.path as osp
import re
import subprocess
import warnings

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

__version__ = '0.7.0'
URL = 'https://github.com/pyg-team/pyg-lib'


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = osp.abspath(sourcedir)


class CMakeBuild(build_ext):
    @staticmethod
    def check_env_flag(name: str, default: str = '') -> bool:
        value = os.getenv(name, default).upper()
        return value in ['1', 'ON', 'YES', 'TRUE', 'Y']

    def get_ext_filename(self, ext_name):
        # Remove Python ABI suffix:
        ext_filename = super().get_ext_filename(ext_name)
        ext_filename_parts = ext_filename.split('.')
        ext_filename_parts = ext_filename_parts[:-2] + ext_filename_parts[-1:]
        return '.'.join(ext_filename_parts)

    def build_extension(self, ext):
        import sysconfig

        import torch

        extdir = osp.abspath(osp.dirname(self.get_ext_fullpath(ext.name)))
        self.build_type = 'DEBUG' if self.debug else 'RELEASE'
        if self.debug is None:
            if CMakeBuild.check_env_flag('DEBUG'):
                self.build_type = 'DEBUG'
            elif CMakeBuild.check_env_flag('REL_WITH_DEB_INFO'):
                self.build_type = 'RELWITHDEBINFO'

        if not osp.exists(self.build_temp):
            os.makedirs(self.build_temp)

        WITH_CUDA = torch.cuda.is_available()
        WITH_CUDA = bool(int(os.getenv('FORCE_CUDA', WITH_CUDA)))

        cmake_args = [
            '-DBUILD_TEST=OFF',
            '-DBUILD_BENCHMARK=OFF',
            f'-DWITH_CUDA={"ON" if WITH_CUDA else "OFF"}',
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={extdir}',
            f'-DCMAKE_BUILD_TYPE={self.build_type}',
            f'-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}',
        ]

        if WITH_CUDA and not os.getenv('TORCH_CUDA_ARCH_LIST'):
            # Set TORCH_CUDA_ARCH_LIST from PyTorch's built architectures
            # so that torch's cmake uses the correct gencode flags.
            # Note: torch.cuda.get_arch_list() returns [] without a GPU,
            # so we call the underlying C function directly.
            arch_flags = torch._C._cuda_getArchFlags()
            if arch_flags:
                arch_list = [
                    x for x in arch_flags.split() if x.startswith('sm_')
                ]
                if arch_list:
                    # Convert ['sm_75', 'sm_100'] to "7.5;10.0"
                    os.environ['TORCH_CUDA_ARCH_LIST'] = ';'.join(
                        f'{d[:-1]}.{d[-1]}' for x in arch_list for d in [x[3:]]
                    )

        if CMakeBuild.check_env_flag('USE_MKL_BLAS'):
            include_dir = f'{sysconfig.get_path("data")}{os.sep}include'
            cmake_args.append(f'-DBLAS_INCLUDE_DIR={include_dir}')
            cmake_args.append('-DUSE_MKL_BLAS=ON')

        with_ninja = importlib.util.find_spec('ninja') is not None
        with_ninja |= os.environ.get('FORCE_NINJA') is not None
        if with_ninja:
            cmake_args += ['-GNinja']
        else:
            warnings.warn(
                "Building times of 'pyg-lib' can be heavily improved"
                " by installing 'ninja': `pip install ninja`",
            )

        build_args = []

        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args,
            cwd=self.build_temp,
        )
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=self.build_temp,
        )


def mkl_dependencies():
    if not CMakeBuild.check_env_flag('USE_MKL_BLAS'):
        return []

    import torch

    dependencies = []
    torch_config = torch.__config__.show()
    with_mkl_blas = 'BLAS_INFO=mkl' in torch_config
    if torch.backends.mkl.is_available() and with_mkl_blas:
        product_version = '2023.1.0'
        pattern = r'oneAPI Math Kernel Library Version [0-9]{4}\.[0-9]+'
        match = re.search(pattern, torch_config)
        if match:
            product_version = match.group(0).split(' ')[-1]
        dependencies.append(f'mkl-include=={product_version}')
        dependencies.append(f'mkl-static=={product_version}')

    return dependencies


install_requires = [] + mkl_dependencies()

triton_requires = [
    'triton',
]

test_requires = [
    'pytest',
    'pytest-cov',
]

dev_requires = [
    'pre-commit',
]

if not bool(os.getenv('BUILD_DOCS', 0)):
    ext_modules = [CMakeExtension('libpyg')]
    cmdclass = {'build_ext': CMakeBuild}
else:
    ext_modules = None
    cmdclass = {}

setup(
    name='pyg_lib',
    version=__version__,
    install_requires=install_requires,
    extras_require={
        'triton': triton_requires,
        'test': test_requires,
        'dev': dev_requires,
    },
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
