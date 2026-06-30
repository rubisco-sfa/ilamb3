# Notes on creating autodocs and switching from mystmd to myst-nb + sphinx

## 1. Added these dependencies to the docs group:
```bash
uv add --group docs sphinx myst-nb sphinx-autodoc2 sphinx-design pydata-sphinx-theme sphinx-copybutton
```
- `sphinx`: the documentation build engine (replace current `mystmd`); runs sphinx-build
- `myst-nb`: sphinx extension that lets sphinx parse myst .md but is built on top of `myst-parser` to also execute notebook cells (`myst-parser` alone cannot execute cells)
- `sphinx-autodoc2`: generates API docs from docstrings (recommended by `myst-parser` on their doc pages); static analysis, emits Markdown
- `sphinx-design`: enables us to use grids and cards and such
- `pydata-sphinx-theme`: html theme that is geared toward Jupyter-style outputs
- `sphinx-copy-button`: I just wanted a copy button on the cells, so I added this

Note: mystmd (the Node CLI, `myst start`/`myst build`) cannot do autodoc. The autodoc capability lives in the Sphinx versions of myst.


## 2. Created a conf.py in new docs-sphinx directory (didn't want to mess with existing docs):
Conf options are for [myst-parser](# See https://myst-parser.readthedocs.io/en/latest/configuration.html) and [myst-nb](https://myst-nb.readthedocs.io/en/latest/configuration.html#config-intro) can be used in the `conf.py`.

```python
project = "ilamb3"

extensions = [
    "myst_nb",
    "sphinx_design",
]

myst_enable_extensions = [
    "dollarmath",  # mystmd has this on by default; myst-nb does not
    "colon_fence",
]

nb_execution_mode = "cache"  # only re-execute when the cell source changes
html_theme = "sphinx_book_theme"
```


## 3. Copy and paste the existing index.md with some small tweaks to get it working
The tweaks in question:
- `{card}` -> `{grid-item-card}`
- `:link: getting_started` -> `:link: getting_started` and `:link-type: doc`


## 4. Set up the docs and see if they actually render fine and without error
```bash
uv run sphinx-build -b html docs-sphinx docs-sphinx/_build/html
```
This creates a _build directory.

```bash
uv run python -m http.server -d docs-sphinx/_build/html 8000
```
This allows me to look at the rendered docs.


## 5. Add more complexity; see if I can get intake.md to execute code cells
First, I need to run setup_doc_assets.py to create the _generated directory.
```bash
python setup_doc_assets.py
```

I found that we have to add this to the top of any markdowns that run code:
```markdown
---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  name: python3
  display_name: 'Python 3'
---
```

I also found that we have to be sure to add the file to the index toctree for it to work. I think we should do nesting (both toctrees and file organization) to make it less confusing and allow drop-downs to work.


## 6. Attempt to add API Docs (it's really easy actually)
Added autodocs2 to extensions in `conf.py`
```python
extensions = ["myst_nb", "sphinx_design", "autodoc2", "sphinx_copybutton"]
```

Added autodocs2-specific configurations
```python
# autodoc2 things
autodoc2_packages = [
    {"path": "../ilamb3"},
]
autodoc2_render_plugin = "myst"
autodoc2_docstring_parser_regexes = [
    (r".*", "myst"),
]
```