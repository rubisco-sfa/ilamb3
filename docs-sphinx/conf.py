# See https://myst-parser.readthedocs.io/en/latest/configuration.html
# See https://myst-nb.readthedocs.io/en/latest/configuration.html#config-intro
project = "ilamb3"

extensions = ["myst_nb", "sphinx_design", "autodoc2", "sphinx_copybutton"]

myst_enable_extensions = [
    "dollarmath",  # mystmd has this on by default; myst-nb does not
    "colon_fence",
]

nb_execution_mode = "cache"  # only re-execute when the cell source changes
html_theme = "sphinx_book_theme"

# autodoc2 things
autodoc2_packages = [
    {"path": "../ilamb3"},
]
autodoc2_render_plugin = "myst"
autodoc2_docstring_parser_regexes = [
    (r".*", "myst"),
]
