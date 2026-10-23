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
autoapi_options = ["members", "undoc-members", "show-inheritance", "show-module-summary", "imported_members"]
templates_path = []
exclude_patterns = []
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/JaLuka98/CPax",
    "show_nav_level": 2,
    "navigation_depth": 3,
}

html_static_path = ["_static"]
suppress_warnings = ["autoapi.python_import_resolution"]
