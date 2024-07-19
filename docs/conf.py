# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from __future__ import annotations

# import os
from dotenv import load_dotenv

load_dotenv()

project = "MLCompare"
copyright = "2024, Mitchell Medeiros"
author = "Mitchell Medeiros"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinxawesome_theme.deprecated",
    "sphinx_sitemap",
    # "dater",
    "sphinx_copybutton",
    # "sphinx_inline_tabs",
    "sphinx_design",
    "sphinx_docsearch",
    "sphinxext.opengraph",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.viewcode",
]

needs_sphinx = "7.3"
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
source_suffix = ".rst"
language = "en"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
nitpicky = True

# docsearch_app_id = os.getenv("DOCSEARCH_APP_ID")
# docsearch_api_key = os.getenv("DOCSEARCH_API_KEY")
# docsearch_index_name = os.getenv("DOCSEARCH_INDEX_NAME")

# ogp_image = (
#     "https://raw.githubusercontent.com/python-pillow/pillow-logo/main/"
#     "pillow-logo-dark-text-1280x640.png"
# )
# ogp_image_alt = "Pillow"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = project
html_theme = "sphinxawesome_theme"
html_static_path = ["_static"]
html_baseurl = "http://127.0.0.1:3000/docs/_build/html/index.html"
html_extra_path = ["robots.txt"]
html_title = "MLCompare Docs"

# html_theme_options = {}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = "resources/pillow-logo.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = "resources/favicon.ico"
