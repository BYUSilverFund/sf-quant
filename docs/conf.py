import sys
from pathlib import Path

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Add project source to path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Silver Fund Quant"
copyright = "2025, Andrew Hall"
author = "Andrew Hall"
release = "0.1.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autosummary_generate = True

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = "Silver Fund Quant"
html_theme = "pydata_sphinx_theme"
html_show_sourcelink = False
templates_path = ["_templates"]
