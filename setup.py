import os
import os.path as osp
import subprocess

import torch
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

__version__ = '0.0.0'
URL = 'https://github.com/pyg-team/pyg-lib'


class CMakeBuild(build_ext):
    def build_extension(self, ext):

        if self.debug is None:
            self.debug = bool(int(os.environ.get('DEBUG', 0)))

        if not osp.exists(self.build_temp):
            os.makedirs(self.build_temp)

        cmake_args = [
            '-DUSE_PYTHON=ON',
            f'-DCMAKE_BUILD_TYPE={"DEBUG" if self.debug else "RELEASE"}',
            f'-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}',
            f'-DWITH_CUDA={"ON" if torch.cuda.is_available() else "OFF"}',
        ]
        subprocess.check_call(['cmake', osp.abspath('.')] + cmake_args,
                              cwd=self.build_temp)

        build_args = []
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
    ext_modules=[Extension('pyg_lib', sources=[])],
    cmdclass={'build_ext': CMakeBuild},
    include_package_data=True,
)
