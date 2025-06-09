import pyg_lib


def test_version():
    if 'dev' in pyg_lib.__version__:
        assert len(pyg_lib.__version__.split('.')) == 4
    else:
        assert len(pyg_lib.__version__.split('.')) == 3
