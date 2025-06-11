# Run an Analysis

We are actively moving `ilamb3` to become the main ILAMB version, but the transition is taking time. If you are familiar with our system, you will find that many convenience functions (like an `ilamb-run`) are missing and you must run custom driver scripts.

## Reference Data Configuration

ILAMB has moved to using [YAML](https://yaml.org/) files to setup the study comparisons. The configure language in ILAMB2x was of my own invention and required me to maintain a custom parser. By using a standard language we leverage other tools and a language the community uses.

Also, you may now arbitrarily nest your reference data blocks (the `WECANN` block in the following example created to resemble the ILAMB2x organization).

```yaml
Ecosystem and Carbon Cycle:
  Gross Primary Productivity:
    WECANN:
      sources:
        gpp: WECANN/gpp.nc
```

However, `ilamb3` could handle a deeper nesting or even none (as shown in the example below). The ILAMB configure parser is looking for the `sources` keyword, which defines the reference data sources to be used in the comparison. In this case, this tells the system to look for the `gpp` variable in the `WECANN/gpp.nc` dataset. At minimum, you always must supply at least one data source.

```yaml
WECANN:
  sources:
    gpp: WECANN/gpp.nc
```

### Alternate Variables

As in previous versions of ILAMB, you may specify a keyword `alternate_vars` to give synonyms that the analysis should accept. In the following example, if a model does have `gpp` it will also accept `GPP`.

```yaml
WECANN:
  sources:
    gpp: WECANN/gpp.nc
  alternate_vars:
  - GPP
```

### Transforms

Newly introduced in `ilamb3`, you may also now provide *transforms* to preprocess both reference and model data before applying analyses. For example, in the following block we intend on configuring a surface comparison of the ocean temperature, which is a function of `time, depth, lat, lon`.

```yaml
thetao-WOA2023-surface:
  sources:
    thetao: WOA/thetao_mon_WOA_A5B4_gn_200501-201412.nc
  transforms:
  - select_depth:
      value: 0
  alternate_vars:
  - tos
```

The transforms are applied to both the reference and comparison data. The `WOA` data is 4 dimensional, and so the transform will select the layer that contains the given value. If a model has `thetao`, then its surface value will be selected but we would also accept the variable `tos` because it is given in the `alternate_vars`.

The library of transforms is growing as we have need. If you can think of something you need you can try to [implement](transform) it or raise an [issue](https://github.com/rubisco-sfa/ilamb3/issues) to make a request.

### Analyses

The configure examples thus far have not specified any analyses to run. We will run the standard set (`bias`, `RMSE`, `Annual Cycle`, `Spatial Distribution`) implicitly unless the user specifies otherwise. For example, if only the `bias` analysis is wanted:

```yaml
WECANN:
  sources:
    gpp: WECANN/gpp.nc
  analyses:
  - bias
```

Like transforms, we are expanding the analyses implementations to capture everything we have in ILAMB2x. While we encourage you to think about making your analysis routines abstract with respect to a given variable, we can also accomodate specific methods. See this [tutorial](analysis) for adding an analysis.

At the moment, relationship analysis is available but follows special setup rules. We look for a `relationships` block where each independent variable is listed with a source.

```yaml
WECANN:
  sources:
    gpp: WECANN/gpp.nc
  relationships:
    pr: GPCPv2.3/pr.nc
    tas: CRU4.02/tas.nc
```

This syntax means that ILAMB will compare `gpp` vs. `pr` and vs. `tas` using these sources, and the model variables.
