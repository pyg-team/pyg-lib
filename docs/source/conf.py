import datetime

import pyg_lib

author = 'PyG Team'
project = 'pyg_lib'
version = pyg_lib.__version__
copyright = f'{datetime.datetime.now().year}, {author}'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
]

html_theme = 'sphinx_rtd_theme'
autodoc_member_order = 'bysource'

intersphinx_mapping = {
    'python': ('http://docs.python.org', None),
    'torch': ('https://pytorch.org/docs/master', None),
}
