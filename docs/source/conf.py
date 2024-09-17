import datetime
import os.path as osp
import sys

import pyg_sphinx_theme
from sphinx.application import Sphinx

import pyg_lib

author = 'PyG Team'
project = 'pyg_lib'
version = pyg_lib.__version__
copyright = f'{datetime.datetime.now().year}, {author}'

sys.path.append(osp.join(osp.dirname(pyg_sphinx_theme.__file__), 'extension'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'sphinx_autodoc_typehints',
    'pyg',
]

html_theme = 'pyg_sphinx_theme'
html_logo = ('https://raw.githubusercontent.com/pyg-team/pyg_sphinx_theme/'
             'master/pyg_sphinx_theme/static/img/pyg_logo.png')
html_favicon = ('https://raw.githubusercontent.com/pyg-team/pyg_sphinx_theme/'
                'master/pyg_sphinx_theme/static/img/favicon.png')

add_module_names = False
autodoc_member_order = 'bysource'

intersphinx_mapping = {
    'python': ('http://docs.python.org', None),
    'torch': ('https://pytorch.org/docs/stable', None),
}

typehints_use_rtype = False
typehints_defaults = 'comma'


def setup(app: Sphinx) -> None:
    r"""Setup sphinx application."""
    # Do not drop type hints in signatures:
    del app.events.listeners['autodoc-process-signature']
