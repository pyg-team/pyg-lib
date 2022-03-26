from setuptools import find_packages, setup

__version__ = '0.0.0'
URL = 'https://github.com/pyg-team/pyg-lib'

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
    author='Matthias Fey',
    author_email='matthias.fey@tu-dortmund.de',
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
    include_package_data=True,
)
