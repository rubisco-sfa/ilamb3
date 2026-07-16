# ILAMB Docs

If you find that the docs are out-of-date, or you want to create and share a documentation page that is missing, this page provides instructions for replicating the `ilamb` documentation locally. This will allow you to create/edit `markdown` files that are located in `ilamb3/docs/`. The `ilamb` documentation is built using [mystmd](https://mystmd.org/). To build the documentation locally and modify it, you can follow the steps below.

## Create a new branch for your changes
After cloning the repository (see [develop.md](develop.md)), it is best practice to create a new `git` branch for your changes. See [develop.md](develop.md) for more details about `git` best practices. You should choose a branch name that reflects the changes you are making. For example, if you are making changes to the documentation, you could name your branch `docs-updates`. First ensure you are on the `main` branch (sometimes also called `master`), then you can create a new branch using the following command:
First, switch to the `main` branch:

```bash
git switch main
```

Then create a new branch for your changes:
```bash
git switch -c docs-updates
```

### Sync the docs environment
To work on the documentation, your working environment will need to include `mystmd`-related packages that don't come with the default `ilamb` installation. To set up the environment and add `docs` dependencies, ensure you are in the `ilamb` root directory where your `uv` virtual environment is stored, then run the following commands:

```bash
uv sync --group docs
source .venv/bin/activate  # activate if you haven't already
```

### Set up the docs assets
Most of what you need to build the documentation is already included in the repository. However, some assets are generated from the code and need to be created before you can build the docs. To generate these assets, run the following command from the `docs` directory:

```bash
cd docs
python setup_doc_assets.py
```
This will generate the API reference pages as well as other assets used in the documentation.

### Start the local documentation server

```bash
myst start  --execute
```
This will start a local server where you can view the documentation. The server will automatically reload as you make changes to the documentation files.

```{tip}
If your docs aren't rendering as expected, close the server (Ctrl+C), clear the cache using `myst clean --site --cache --yes` and `myst clean --execute`, then restart the server.
```