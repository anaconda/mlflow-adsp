# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from os.path import dirname, join, pardir

sys.path.insert(0, join(dirname(__file__), pardir))
sys.path.insert(0, os.path.abspath("../../"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MLFlow Plugin For The Anaconda Data Science Platform"
copyright = "2023, Anaconda, Inc"
author = "Joshua C. Burt"

# The full version, including alpha/beta/rc tags
from mlflow_adsp.__init__ import __version__

release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_parser", "sphinx.ext.napoleon", "sphinx_click"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
