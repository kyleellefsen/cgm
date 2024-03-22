"""Configuration file for the Sphinx documentation builder."""
import sys
import pathlib
import sphinx_rtd_theme

cgm_path = pathlib.Path('../src').resolve()
assert cgm_path.exists(), f"Path does not exist: {cgm_path}"
sys.path.insert(0, str(cgm_path))

project = 'Causal Graphical Models'
copyright = '2024, Kyle Ellefsen'
author = 'Kyle Ellefsen'
version = "0.0.9"
release = version

extensions = [
    "myst_parser",
    "autodoc2",
    "sphinx_math_dollar",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    # "sphinx.ext.todo",
    # "sphinx.ext.mathjax",
    # "sphinx.ext.ifconfig",
    # "sphinx.ext.viewcode",
    # "sphinx.ext.githubpages",
    # "sphinx.ext.graphviz",
    # "sphinx.ext.doctest",
]

# Autodoc2 settings
autodoc2_packages = [
    {
        "path": "../src/cgm",
        "auto_mode": False,
        "exclude_dirs": [],
        "exclude_files": ["example_graphs.py"],
    }
]

# Napoleon settings
napoleon_google_docstring = True

# autodoc_inherit_docstrings = False
source_suffix = [".md", ".rst"]
master_doc = "index"
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = "sphinx"
language = 'en'

# Theme and styling
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]