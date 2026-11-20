import os
import sys


sys.path.insert(0, os.path.abspath("../colony_differentiation"))


project = "Colony Differentiation"
language = "en"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",]


source_suffix = ".rst"
root_doc = "index"


html_theme = "furo"
html_title = "Colony Differentiation"

html_static_path = ["static"]
html_css_files = ["custom.css"]


autodoc_member_order = "bysource"
numpydoc_show_class_members = False


exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/.ipynb_checkpoints"]