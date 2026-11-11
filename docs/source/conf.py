"""Sphinx configuration for the tscp documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

import tscp  # noqa: E402

project = "tscp"
author = "Roberto Neglia"
copyright = "2026, Roberto Neglia"
release = tscp.__version__
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["build", "generated"]

autodoc_default_options = {"members": True, "imported-members": True, "member-order": "bysource"}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
myst_heading_anchors = 3

# Set TSCP_DOCS_OFFLINE=1 to build without fetching the inventories of other projects.
intersphinx_mapping = (
    {}
    if os.environ.get("TSCP_DOCS_OFFLINE")
    else {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable", None),
        "torch": ("https://docs.pytorch.org/docs/stable", None),
    }
)

html_theme = "furo"
html_title = f"tscp {release}"
