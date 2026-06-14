import os
import sys
from datetime import datetime

# Ensure project root is on sys.path so autodoc can import the package
sys.path.insert(0, os.path.abspath('..'))

project = 'soliton_solver'
author = 'Paul Leask'
copyright = f"{datetime.now().year}, {author}"

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

mathjax3_config = {
  "tex": {
    "inlineMath": [['\\(', '\\)']],
    "displayMath": [["\\[", "\\]"]],
  }
}

autosummary_generate = True
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_book_theme'
# html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    "logo": {
        "image_light": "soliton_solver_logo.png",
        "image_dark": "soliton_solver_logo.png",
    },
    "repository_url": "https://github.com/Paulnleask/soliton_solver",
    "use_repository_button": True,
}

# MyST config
myst_enable_extensions = [
    'deflist',
    'html_admonition',
    'html_image',
    'dollarmath',
]

myst_dmath_double_inline = True
