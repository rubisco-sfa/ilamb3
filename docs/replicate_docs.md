# How to stand up a copy of the ILAMB documentation locally
ILAMB documentation is built using [mystmd](https://mystmd.org/). To build the documentation locally and modify it, you can follow these steps:

1. After cloning the repository, create a new branch for your documentation changes:

```bash
git checkout -b my-docs-changes
```

2. Sync the docs environment and activate it:

```bash
uv sync --group docs
source .venv/bin/activate
```

3. Navigate to the docs directory and set up the docs assets

```bash
cd docs
python setup_doc_assets.py
```
This will generate the API reference pages and the assets used in the documentation.

4. Start the local documentation server:

```bash
myst start
```
This will start a local server where you can view the documentation. The server will automatically reload as you make changes to the documentation files.