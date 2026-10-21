"""Sphinx configuration for CPax."""

from importlib.metadata import version as distribution_version

project = "CPax"
author = "Jan Lukas Späh"
copyright = "2026, Jan Lukas Späh"
release = distribution_version("CPax")
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "autoapi.extension",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

autoapi_type = "python"
autoapi_dirs = ["../../src/cpax"]
autoapi_ignore = ["*/tests/*", "*/_backends/*"]
autoapi_add_toctree_entry = False
autoapi_options = ["members", "undoc-members", "show-inheritance", "show-module-summary"]
templates_path = []
exclude_patterns = []
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
suppress_warnings = ["autoapi.python_import_resolution"]
