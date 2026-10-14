import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import setuptools
from setuptools import Distribution


@pytest.fixture
def build_cuda_extension(monkeypatch, tmp_path):
    monkeypatch.setenv('FORCE_CUDA', '1')
    monkeypatch.setenv('FORCE_NINJA', '1')
    monkeypatch.setenv('USE_MKL_BLAS', '0')
    monkeypatch.setenv('TORCH_CUDA_ARCH_LIST', '')
    monkeypatch.delenv('BUILD_DOCS', raising=False)
    monkeypatch.setattr(setuptools, 'setup', Mock())
    setup_path = Path(__file__).resolve().parents[1] / 'setup.py'
    setup = runpy.run_path(str(setup_path))

    def build(arch_flags, override=None):
        get_arch_flags = Mock(return_value=arch_flags)
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            _C=SimpleNamespace(_cuda_getArchFlags=get_arch_flags),
            utils=SimpleNamespace(cmake_prefix_path=str(tmp_path / 'torch')),
        )
        monkeypatch.setitem(sys.modules, 'torch', torch)
        if override is not None:
            monkeypatch.setenv('TORCH_CUDA_ARCH_LIST', override)

        calls = []

        def check_call(args, cwd):
            calls.append((args, cwd, os.environ['TORCH_CUDA_ARCH_LIST']))

        monkeypatch.setattr(subprocess, 'check_call', check_call)
        extension = setup['CMakeExtension'](
            'pyg_lib.libpyg',
            sourcedir=str(setup_path.parent),
        )
        command = setup['CMakeBuild'](Distribution())
        command.build_lib = str(tmp_path / 'lib')
        command.build_temp = str(tmp_path / 'build')
        command.debug = False
        command.build_extension(extension)

        assert len(calls) == 2
        assert '-DWITH_CUDA=ON' in calls[0][0]
        assert '-DCMAKE_CUDA_ARCHITECTURES=OFF' in calls[0][0]
        assert calls[1][0] == ['cmake', '--build', '.']
        assert calls[0][1] == calls[1][1] == command.build_temp
        if override is not None:
            get_arch_flags.assert_not_called()
        else:
            get_arch_flags.assert_called_once_with()
        return calls[0][2]

    return build


@pytest.mark.parametrize(
    ('arch_flags', 'expected'),
    [
        ('sm_80 sm_90 compute_120', '8.0;9.0;12.0+PTX'),
        ('sm_80 sm_90 compute_100', '8.0;9.0;10.0+PTX'),
        ('sm_80 sm_100 sm_120', '8.0;10.0+PTX;12.0'),
        ('sm_100 sm_120 compute_120', '10.0+PTX;12.0;12.0+PTX'),
        ('sm_100 compute_100', '10.0;10.0+PTX'),
        ('sm_107', '10.7'),
        ('sm_90a', '9.0a'),
        ('compute_90a', '9.0a+PTX'),
        ('sm_100a', '10.0a'),
        ('compute_100a', '10.0a+PTX'),
        ('sm_100f', '10.0f'),
        ('compute_100f', '10.0f+PTX'),
        ('sm_50 compute_50 sm_60 sm_75', '6.0;7.5'),
    ],
)
def test_cuda_architectures(build_cuda_extension, arch_flags, expected):
    assert build_cuda_extension(arch_flags) == expected


def test_cuda_architectures_override(build_cuda_extension):
    arch_list = '8.0;9.0+PTX;10.0;12.0+PTX'
    assert build_cuda_extension(None, override=arch_list) == arch_list
