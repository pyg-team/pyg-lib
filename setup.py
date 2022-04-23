import importlib
import os
import os.path as osp
import subprocess

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

__version__ = '0.0.0'
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

    def build_extension(self, ext):
        import torch

        extdir = os.path.abspath(osp.dirname(self.get_ext_fullpath(ext.name)))

        if self.debug is None:
            self.debug = bool(int(os.environ.get('DEBUG', 0)))

        if not osp.exists(self.build_temp):
            os.makedirs(self.build_temp)

        cmake_args = [
            '-DUSE_PYTHON=ON',
            '-DBUILD_TEST=OFF',
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DCMAKE_BUILD_TYPE={"DEBUG" if self.debug else "RELEASE"}',
            f'-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}',
            f'-DWITH_CUDA={"ON" if torch.cuda.is_available() else "OFF"}',
        ]

        if importlib.util.find_spec('ninja') is not None:
            cmake_args += ['-GNinja']

        build_args = []

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)


install_requires = []

test_requires = [
    'pytest',
    'pytest-cov',
]

dev_requires = test_requires + [
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
        'deep-learning'
        'pytorch',
        'geometric-deep-learning',
        'graph-neural-networks',
        'graph-convolutional-networks',
    ],
    python_requires='>=3.7',
    install_requires=install_requires,
    extras_require={
        'test': test_requires,
        'dev': dev_requires,
    },
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    include_package_data=True,
)
