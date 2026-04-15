---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# Creating Model CSV Files with intake-esgf

This tutorial will teach you how to download CMIP data and create CSV files which `ilamb3` requires for execution using [intake-esgf](https://intake-esgf.readthedocs.io/). In order to stay as general as possible, `ilamb3` does not depend directly on `intake-esgf` but it is a useful tool for easily accessing data hosted on the Earth System Grid Federation (ESGF). For a more thorough explanation of that package and all its options, please consult the intake-esgf documentation.

`intake-esgf` catalogs initialize empty and are populated by writing a faceted `search`.

```{code-cell} python
from intake_esgf import ESGFCatalog

cat = ESGFCatalog().search(
    experiment_id="historical",
    source_id="CanESM5",
    variable_id=["gpp", "areacella", "sftlf"],
    table_id=["Lmon", "fx"],
    file_start="1980-01",
    file_end="2016-01",
)
cat
```

From the catalog summary, we see that there are many ensemble members and we only want a single member for this run. The catalog has a function that can be used to remove all ensembles except the smallest.

```{code-cell} python
cat.remove_ensembles()
cat
```

Once the catalog represents the data that you wish to download and use in your benchmarking study, we ask the catalog for a dictionary of paths.

```{code-cell} python
dpd = cat.to_path_dict(minimal_keys=False)
dpd
```

We have used the keyword argument `minimal_keys=False` so that the keys of the dictionary contain all the facets which define a unique dataset. We can use these keys and the known order they occur in to create a pandas DataFrame with the required columns for `ilamb3`.

```{code-cell} python
import pandas as pd

# The order in which facets appear in the keys
KEY_PATTERN = [
    "mip_era",
    "activity_id",
    "institution_id",
    "source_id",
    "experiment_id",
    "member_id",
    "table_id",
    "variable_id",
    "grid_label",
]
# Create each row of the dataframe
df = []
for key, paths in dpd.items():
    row = {col: value for col, value in zip(KEY_PATTERN, key.split("."))}
    for path in paths:
        row["path"] = str(path)
        df.append(row)
df = pd.DataFrame(df)
# Export as a CSV
df.to_csv("CanESM5.csv")
```

This produces a CSV file which looks like this:

```{code-cell} python
:tags: [remove-input]
from pathlib import Path
Path("CanESM5.csv").unlink()
import pandas as pd
df = pd.read_csv("_generated/CanESM5.csv").drop(columns=["Unnamed: 0"])
df
```

While you need not store each model's output as a separate CSV file, this is a useful convention so that including/excluding any model from a study is simple.
