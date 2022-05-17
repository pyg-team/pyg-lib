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

html_theme = 'pyg_sphinx_theme'
autodoc_member_order = 'bysource'
html_logo = ('https://raw.githubusercontent.com/pyg-team/pyg_sphinx_theme/'
             'master/pyg_sphinx_theme/static/img/pyg_logo.png')
html_theme_options = {
    'logo_only': True,
}

intersphinx_mapping = {
    'python': ('http://docs.python.org', None),
    'torch': ('https://pytorch.org/docs/master', None),
}
