# Start Developing ILAMB

We recommend using [`uv`](https://docs.astral.sh/uv/) to manage Python environments. Below, we provide best practices for contributing to `ilamb` and instructions for setting up a development environment.

### Fork and clone the repository

The repository you clone should be a repository that you can `git push` to. The default is that only maintainers have `push` rights to the main repository, while outside contributors should fork the repository then submit pull requests to contribute to `ilamb`. There are two major `git`-based web platforms for hosting code repositories: [GitHub](https://github.com/) and [GitLab](https://gitlab.com/). Because `ilamb` is hosted on GitHub, the instructions below assume you are using GitHub as well. If you are using GitLab, the instructions are similar, but you will need to adjust the URLs accordingly.

First, [fork `rubisco-sfa/ilamb3`](https://github.com/rubisco-sfa/ilamb3/fork) on GitHub. Then clone your fork, replacing `YOUR-USERNAME` with your GitHub username:

```bash
git clone git@github.com:YOUR-USERNAME/ilamb3.git
cd ilamb3
git remote add upstream git@github.com:rubisco-sfa/ilamb3.git
```

In this setup, `origin` is your fork and `upstream` is the main repository. Confirm the configuration with:

```bash
git remote -v
```

```{tip}
These examples use SSH. If you have not configured an SSH key for GitHub, see [Connecting to GitHub with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).
Alternatively, you can use an HTTPS clone URL, but you will need to consult [authentication methods for command-line Git](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github#authenticating-with-the-command-line) to push to your fork without error.
```

## Install the development dependencies

From the repository root, create the environment and install the development and ESGF dependency groups:

```bash
uv sync --group dev --group esgf
source .venv/bin/activate
```

The `dev` group includes tools such as `pytest`. The optional `esgf` group installs `intake-esgf`, which is useful when testing with model data from ESGF. See `pyproject.toml` for all available dependency groups.

```{tip}
For parallel execution on a cluster, add `--group parallel` to `uv sync`. This group requires an MPI implementation such as Open MPI. On many clusters you can make it available with `module load openmpi`; ask your cluster administrator if you are unsure about what is available.
```

## Verify the installation

Check that the command-line interface is available:

```bash
ilamb --version
ilamb --help
```

## Make and submit changes

Every time you contribute to `ilamb`, you should follow the best practices below. If you are contributing to the documentation, see [replicate-docs.md](replicate-docs.md) for additional instructions.

### 1. Update your local `main` branch

If you cloned a fork, update from `upstream`, and then update your fork:

```bash
git fetch upstream
git switch main
git merge --ff-only upstream/main
git push origin main
```
Every time you work on your changes, you should ensure that your local `main` branch is up to date with the main repository. This will help avoid merge conflicts when you submit your changes later.

### 2. Create a topic branch

Create a branch with a short, descriptive name. Do not work directly on `main`. For example, if you are fixing a bug in the transform code, you could name your branch `transform-bugfix`:

```bash
git switch -c transform-bugfix
```

### 3. Develop and test

Make focused changes and run relevant tests. You can create and add your own code tests to `ilamb3/tests/`. Any time you add new functionality to `ilamb`, it is best to create corresponding tests. To run the full test suite:

```bash
pytest
```

Be sure to resolve any failing tests before submitting a Pull Request (PR). You can also run a subset of tests by specifying the path to the test file or directory:

```bash
pytest tests/test_transform.py  # all tests in the .py
pytest tests/test_transform.py::test_transform  # a single test in the .py
```

Review and commit your work with a descriptive message:

```bash
git status
git add path/to/changed-file
git commit -m "bugfix: fix <bug description> in <module>"
```

### 4. Keep your branch current

Before submitting the change, incorporate recent changes from the main repository:

```bash
git fetch upstream
git rebase upstream/main
```

Resolve any conflicts, rerun the relevant tests, and continue the rebase with `git rebase --continue` if needed.

### 5. Push and open a pull request

Push the topic branch to `origin`:

```bash
git push -u origin transform-bugfix
```

Then open a [pull request](https://github.com/rubisco-sfa/ilamb3/compare) against the `main` branch of `rubisco-sfa/ilamb3`. For a fork, GitHub will use the branch in your fork as the pull request's source.

If you rebase after pushing, update the remote branch safely with:

```bash
git push --force-with-lease
```
