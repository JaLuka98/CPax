# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
try:
    from cpax import __version__
except (ImportError, AttributeError) as e:
    raise RuntimeError(
        "Failed to import `__version__` from the `cpax` package. Ensure that the `cpax` package is installed "
        "and contains a `__version__` attribute."
    ) from e
sys.path.insert(0, os.path.abspath('../../src'))

project = 'CPax'
copyright = '2025, Jan Lukas Späh'
author = 'Jan Lukas Späh'
release = __version__
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # For Google/NumPy-style docstrings
    'sphinx.ext.viewcode',
    'autoapi.extension',  # let us test this
    'sphinx_autodoc_typehints',
    'myst_parser',  # Optional for Markdown
]

autoapi_type = 'python'
autoapi_dirs = ['../../src']
autoapi_ignore = ["*/__main__.py", "*/tests/*"]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
