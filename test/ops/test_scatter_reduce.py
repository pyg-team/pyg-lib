# from pyg_lib import fused_scatter_reduce
from pyg_lib.testing import onlyCUDA, onlyTriton


@onlyCUDA
@onlyTriton
def test_fused_scatter_reduce():
    pass
