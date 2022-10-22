import pyg_lib


def test_version():
    assert len(pyg_lib.__version__.split('.')) == 3
