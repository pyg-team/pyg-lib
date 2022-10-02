import importlib
import os
import os.path as osp
import subprocess
import warnings

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

__version__ = '0.1.0'
URL = 'https://github.com/pyg-team/pyg-lib'


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def get_ext_filename(self, ext_name):
        # Remove Python ABI suffix:
        ext_filename = super().get_ext_filename(ext_name)
        ext_filename_parts = ext_filename.split('.')
        ext_filename_parts = ext_filename_parts[:-2] + ext_filename_parts[-1:]
        return '.'.join(ext_filename_parts)

    def check_env_flag(self, name: str, default: str = "") -> bool:
        value = os.getenv(name, default).upper()
        return value in ["1", "ON", "YES", "TRUE", "Y"]

    def build_extension(self, ext):
        import torch

        extdir = os.path.abspath(osp.dirname(self.get_ext_fullpath(ext.name)))
        self.build_type = "DEBUG" if self.debug else "RELEASE"
        if self.debug is None:
            if self.check_env_flag("DEBUG"):
                self.build_type = "DEBUG"
            elif self.check_env_flag("REL_WITH_DEB_INFO"):
                self.build_type = "RELWITHDEBINFO"
        print(f"-- Build type: {self.build_type}")

        if not osp.exists(self.build_temp):
            os.makedirs(self.build_temp)

        WITH_CUDA = torch.cuda.is_available()
        WITH_CUDA = bool(int(os.getenv('FORCE_CUDA', WITH_CUDA)))

        cmake_args = [
            '-DBUILD_TEST=OFF',
            '-DBUILD_BENCHMARK=OFF',
            '-DUSE_PYTHON=ON',
            f'-DWITH_CUDA={"ON" if WITH_CUDA else "OFF"}',
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DCMAKE_BUILD_TYPE={self.build_type}',
            f'-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}',
        ]

        if importlib.util.find_spec('ninja') is not None:
            cmake_args += ['-GNinja']
        else:
            warnings.warn("Building times of 'pyg-lib' can be heavily improved"
                          " by installing 'ninja': `pip install ninja`")

        build_args = []

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)


install_requires = []

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
    description='Low-Level Graph Neural Network Operators for PyG',
    author='PyG Team',
    author_email='team@pyg.org',
    url=URL,
    download_url=f'{URL}/archive/{__version__}.tar.gz',
    keywords=[
        'deep-learning',
        'pytorch',
        'geometric-deep-learning',
        'graph-neural-networks',
        'graph-convolutional-networks',
    ],
    python_requires='>=3.7',
    install_requires=install_requires,
    extras_require={
        'dev': dev_requires,
    },
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
