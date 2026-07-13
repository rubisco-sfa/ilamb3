Cloned the repository and installed dependencies:

```bash
git clone https://github.com/stefanv/myst-apidoc.git myst-apidoc
cd myst-apidoc/myst-apidoc-plugin
npm install
npm run build
```

Added this to myst.yml:

```yaml
  plugins:
    - assets/myst-apidoc/myst-apidoc-plugin/dist/index.mjs  # relative to the myst.yml file
```

Added dependencies:

```bash
uv pip install numpydoc
uv pip install rst-to-myst
```