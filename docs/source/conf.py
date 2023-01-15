import datetime
import os.path as osp
import sys

import pyg_sphinx_theme

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
    'pyg',
]

html_theme = 'pyg_sphinx_theme'
html_logo = ('https://raw.githubusercontent.com/pyg-team/pyg_sphinx_theme/'
             'master/pyg_sphinx_theme/static/img/pyg_logo.png')
html_favicon = ('https://raw.githubusercontent.com/pyg-team/pyg_sphinx_theme/'
                'master/pyg_sphinx_theme/static/img/favicon.png')
html_static_path = ['_static']

add_module_names = False
autodoc_member_order = 'bysource'

intersphinx_mapping = {
    'python': ('http://docs.python.org', None),
    'torch': ('https://pytorch.org/docs/master', None),
}
