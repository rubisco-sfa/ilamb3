## 1. Cloned the repository and built the plugin:

```bash
git clone https://github.com/stefanv/myst-apidoc.git myst-apidoc
cd myst-apidoc/myst-apidoc-plugin
npm install
npm run build
```

## 2. Added this to myst.yml:

```yaml
  plugins:
    - assets/myst-apidoc/myst-apidoc-plugin/dist/index.mjs  # relative to the myst.yml file
```

## 3. Added dependencies:

```bash
uv pip install numpydoc
uv pip install rst-to-myst
```

## 4. Removed the .git repo inside myst-apidoc/myst-apidoc-plugin to avoid confusion:

```bash
cd ../../../  # back into the docs folder from the plugin folder
rm -rf assets/myst-apidoc/.git
```

## 5. Added some of their stuff to .gitignore:

```gitignore
# Myst-APIDoc plugin dependencies
docs/assets/myst-apidoc/myst-apidoc-plugin/node_modules/
docs/assets/myst-apidoc/myst-apidoc-plugin/.github/
docs/assets/myst-apidoc/myst-apidoc-plugin/thumbnail.png
```

## 6. Create the API JSON file:

```bash
cd .. # back into the root of the repo
python docs/assets/myst-apidoc/fleece ilamb3 > docs/_generated/ilamb3-api.json
```
