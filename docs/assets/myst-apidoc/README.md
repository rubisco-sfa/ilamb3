# myst-apidoc

Proof of concept API documentation generator for [mystmd](https://mystmd.org).

Recursively walk a module, and export numpydoc-format docstrings to JSON.

Refer to the [plugin README](myst-api-plugin/README.md) for usage.

For an example rendering, see: https://github.com/stefanv/myst-apidoc-example

## Included here

- `./fleece`: script to extract numpydoc-style docstrings from a Python package
- `./myst-apidoc-plugin`: mystmd plugin to render the JSON output by fleece
