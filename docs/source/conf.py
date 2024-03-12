# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = "GridKit"
copyright = "2023, Timo Millenaar"
author = "Timo Millenaar"

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import shutil
import sys
import warnings

sys.path.insert(0, os.path.abspath("."))
import generate_api_rst

sys.path.insert(0, os.path.abspath("../../gridkit"))
sys.path.insert(0, os.path.abspath("../source"))
sys.path.insert(
    0,
    os.path.abspath(
        "/home/timo/Documents/projects/venv_gridding_bare/lib/python3.10/site-packages/"
    ),
)
import gridkit

# The full version, including alpha/beta/rc tags
release = version = gridkit.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.plantuml",
]


warnings.filterwarnings(
    "ignore", category=RuntimeWarning
)  # Do not report warnings in gallery
sphinx_gallery_conf = {
    "examples_dirs": ["../../examples"],  # path to your example scripts
    "gallery_dirs": "example_gallery",  # path to where to save gallery generated output
    "filename_pattern": "^((?!sgskip).)*$",
    "remove_config_comments": True,  # remove comments like: # sphinx_gallery_thumbnail_number = -1
    "nested_sections": False,
    "matplotlib_animations": True,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []


# -- Generate api doc stubs -----------------------------------------------

if os.path.exists("api"):
    shutil.rmtree("api")
os.mkdir("api")
generate_api_rst.gen_api_reference("../../gridkit", "api")

# Warn about broken references.
nitpicky = True

# Populate nitpick_ignore form separate file https://stackoverflow.com/a/30624034
nitpick_ignore = []
for line in open("nitpick-exceptions.txt"):
    if line.strip() == "" or line.startswith("#"):
        continue
    dtype, target = line.split(None, 1)
    target = target.strip()
    nitpick_ignore.append((dtype, target))

autodoc_default_options = {
    "member-order": "bysource",  # Options: alphabetical, groupwise, bysource
    "private-members": False,  # Display private objects (eg. _foo)
    "special-members": False,  # Display special objects (eg. __foo__)
}


import gridkit.bounded_grid
import gridkit.hex_grid
import gridkit.rect_grid
import gridkit.tri_grid

# need to assign the names here, otherwise autodoc won't document these classes,
# and will instead just say 'alias of ...'
# after https://github.com/slundberg/shap/blob/6af9e1008702fb0fab939bf2154bbf93dfe84a16/docs/conf.py#L380-L394

gridkit.tri_grid.BoundedTriGrid.__name__ = "BoundedTriGrid"
gridkit.rect_grid.BoundedRectGrid.__name__ = "BoundedRectGrid"
gridkit.hex_grid.BoundedHexGrid.__name__ = "BoundedHexGrid"
gridkit.bounded_grid.BoundedGrid.__name__ = "BoundedGrid"
