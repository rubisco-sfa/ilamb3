# `ilamb3` Documentation

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card}
**[🐑 Get Started with ILAMB](getting_started)**

Learn how ILAMB works and how to add your own model or reference data.
- [Installation](installation)
- [Quickstart](quickstart)
- [Datasets](datasets)
- Reading the ILAMB results (dataset page, unified dashboard)
:::

:::{grid-item-card}
**[🔍 How do I...](how_do_i)**

After you have learned the basics, examples of more advanced usage.
- [Obtaining CMIP model data](intake)
- [Write a configure file](configure_yaml)
- [Organize a benchmark configure file](organize)
- Run ilamb in parallel
- Call ilamb analysis routines from my own notebook
- Run over and encode my own regions
- `ilamb run` options
- [Set global options](global_options)
- Access reference data programmatically
:::

:::{grid-item-card}
**[🤿 Deep dive](deep_dive)**

Expanded descriptions of how key functionality works in the `ilamb3` internals.
- Datasets package
- Compare package
- [Transforms](transforms)
- [Analyses](analysis)
- Meta analysis
:::

:::{grid-item-card}
**[📖 Reference](datasets)**

Documentation for all `ilamb3` internals.
- [Glossary](glossary)
- Auto-generated API documentation
- A list of every markdown file in the doc for easy search
- Links/descriptions to our supported runs
- Links/descriptions to community use
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

getting_started
how_do_i
deep_dive
datasets
intake
```