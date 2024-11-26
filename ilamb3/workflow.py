import glob
import itertools
import logging
import os
import re
import time
import warnings
from inspect import isclass
from pathlib import Path
from traceback import format_exc

import pandas as pd
import xarray as xr
import yaml
from jinja2 import Template
from mpi4py.futures import MPIPoolExecutor
from tqdm import tqdm

import ilamb3
import ilamb3.analysis as anl
import ilamb3.dataset as dset
import ilamb3.models as ilamb_models
from ilamb3.analysis.base import ILAMBAnalysis
from ilamb3.exceptions import AnalysisFailure
from ilamb3.models.base import Model
from ilamb3.regions import Regions

DEFAULT_ANALYSES = {"bias": anl.bias_analysis}


def open_reference_data(key_or_path: str | Path) -> xr.Dataset:
    """
    Load the reference data.

    Parameters
    ----------
    key_or_path: str or Path
        The key or path to the reference data. First we check if it is a key in the
        `ilamb3.ilamb_catalog()` and then if it is a path to a file. That path may be
        absolute or relative to an environment variable `ILAMB_ROOT`.

    Returns
    -------
    xr.Dataset
        The reference data.
    """
    key_or_path = Path(key_or_path)
    if key_or_path.is_file():
        return xr.open_dataset(key_or_path)
    if "ILAMB_ROOT" in os.environ:
        root = Path(os.environ["ILAMB_ROOT"])
        if (root / key_or_path).is_file():
            return xr.open_dataset(root / key_or_path)
    cat = ilamb3.ilamb_catalog()
    key_or_path = str(key_or_path)
    if key_or_path in cat:
        return cat[key_or_path].read()
    raise FileNotFoundError(
        f"'{key_or_path}' is not a key in the ilamb3 catalog, nor is it a valid file as an absolute path or relative to ILAMB_ROOT={root}"
    )


def _get_logger(logfile: str, reset: bool = False) -> logging.Logger:

    # Where will the log be written?
    logfile = Path(logfile).expanduser()
    if reset and logfile.exists():
        logfile.unlink()
    logfile.touch()

    # We need a named logger to avoid other packages that use the root logger
    logger = logging.getLogger(str(logfile))
    if not logger.handlers:
        # Now setup the file into which we log relevant information
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(
            logging.Formatter(
                "[\x1b[36;20m%(asctime)s\033[0m][\x1b[36;20m%(levelname)s\033[0m] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    # This is probably wrong, but when I log from my logger it logs from parent also
    logger.parent.handlers = []
    return logger


def work_model_data(work):

    # Unpack the work
    model, analysis_setup = work

    # Setup logging
    root = Path(ilamb3.conf["build_dir"]) / (analysis_setup["path"])
    logfile = root / f"{model.name}.log"
    logger = _get_logger(logfile, reset=True)

    logger.info(f"Begin {analysis_setup['path']} {model.name}")
    with warnings.catch_warnings(record=True) as recorded_warnings:
        # If a specialized analysis will be required, then it should be specified in the
        # analysis_setup.
        try:
            if "analysis" in analysis_setup:
                raise NotImplementedError()
            else:
                analysis_time = time.time()
                df, ds_ref, ds_com = work_default_analysis(model, **analysis_setup)

                # serialize, the reference is tricky as many threads may be trying
                df.to_csv(root / f"{model.name}.csv", index=False)
                ds_com.to_netcdf(root / f"{model.name}.nc")
                ref_file = root / "Reference.nc"
                if not ref_file.is_file():
                    try:
                        ds_ref.to_netcdf(ref_file)
                        logger.info(f"Saved reference file: {ref_file}")
                    except Exception:
                        logger.info("Did not save reference file.")

                analysis_time = time.time() - analysis_time
                logger.info(f"Analysis completed in {analysis_time:.1f} [s]")
        except Exception:
            logger.error(format_exc())

    # Dump the warnings to the logfile
    for w in recorded_warnings:
        logger.warning(str(w.message))

    return


def load_local_assets(root: str | Path) -> tuple[pd.DataFrame, dict[str, xr.Dataset]]:
    root = Path(root)
    df = pd.concat(
        [
            pd.read_csv(f, keep_default_na=False, na_values=["NaN"])
            for f in glob.glob(str(root / "*.csv"))
        ]
    ).drop_duplicates(subset=["source", "region", "analysis", "name"])
    df["name"] = df["name"] + " [" + df["units"] + "]"
    dsd = {
        Path(key).stem: xr.open_dataset(key) for key in glob.glob(str(root / "*.nc"))
    }
    # consistency checks
    assert set(df["source"]) == set(dsd.keys())
    return df, dsd


def post_model_data(analysis_setup):
    """."""
    # Setup logging
    root = Path(ilamb3.conf["build_dir"]) / (analysis_setup["path"])
    logfile = root / "post.log"
    logger = _get_logger(logfile, reset=True)

    analyses = setup_analyses(**analysis_setup)
    ilamb_regions = Regions()
    logger.info(f"Begin post {analysis_setup['path']}")
    with warnings.catch_warnings(record=True) as recorded_warnings:
        post_time = time.time()

        # Load what we have in the local directory
        try:
            df, com = load_local_assets(root)
            ref = com.pop("Reference") if "Reference" in com else xr.Dataset()
        except Exception:
            logger.error("An exception was encountered loading local assets.")
            logger.error(format_exc())
            return

        # Make plots and write plots
        try:
            df_plots = []
            for _, analysis in analyses.items():
                if "plots" in dir(analysis):
                    df_plots += [analysis.plots(df, ref, com)]
            df_plots = pd.concat(df_plots, ignore_index=True)
            for _, row in df_plots.iterrows():
                row["axis"].get_figure().savefig(
                    root / f"{row['source']}_{row['region']}_{row['name']}.png"
                )
        except Exception:
            logger.error("An exception was encountered creating plots")
            logger.error(format_exc())
            return

        # Write out html
        try:
            df = df.reset_index(drop=True)
            df["id"] = df.index
            data = {
                "page_header": ref.attrs["header"] if "header" in ref.attrs else "",
                "model_names": [m for m in df["source"].unique() if m != "Reference"],
                "regions": {
                    (None if key == "None" else key): (
                        "All Data" if key == "None" else ilamb_regions.get_name(key)
                    )
                    for key in df["region"].unique()
                },
                "analyses": list(df["analysis"].unique()),
                "data_information": {
                    key.capitalize(): ref.attrs[key]
                    for key in ["title", "institutions", "version"]
                    if key in ref.attrs
                },
                "table_data": str(
                    [row.to_dict() for _, row in df.drop(columns="units").iterrows()]
                ),
            }
            template = open(
                Path(ilamb3.__path__[0]) / "templates/dataset_page.html"
            ).read()
            open(root / "index.html", mode="w").write(Template(template).render(data))
        except Exception:
            logger.error("An exception was encountered creating the html page.")
            logger.error(format_exc())
            return

        post_time = time.time() - post_time
        logger.info(f"Post-processing completed in {post_time:.1f} [s]")

    # Dump the warnings to the logfile
    for w in recorded_warnings:
        logger.warning(str(w.message))

    return


def setup_analyses(**analysis_setup) -> dict[str, ILAMBAnalysis]:

    # Check on inputs
    sources = analysis_setup.get("sources", {})
    relationships = analysis_setup.get("relationships", {})
    if len(sources) != 1:
        raise ValueError(
            f"The default ILAMB analysis requires a single variable and source, but I found: {sources}"
        )
    variable = list(sources.keys())[0]

    # Setup the default analysis
    cmap = (
        analysis_setup.pop("variable_cmap")
        if "variable_cmap" in analysis_setup
        else "viridis"
    )
    analyses = {
        name: a(variable, variable_cmap=cmap)
        for name, a in DEFAULT_ANALYSES.items()
        if analysis_setup.get(f"skip_{name}", False) is False
    }
    analyses.update(
        {
            f"rel_{ind_variable}": anl.relationship_analysis(variable, ind_variable)
            for ind_variable in relationships
        }
    )
    return analyses


def work_default_analysis(model: Model, **analysis_setup):

    # Check on inputs
    sources = analysis_setup.get("sources", {})
    relationships = analysis_setup.get("relationships", {})
    if len(sources) != 1:
        raise ValueError(
            f"The default ILAMB analysis requires a single variable and source, but I found: {sources}"
        )
    variable = list(sources.keys())[0]

    # Setup the default analysis
    cmap = (
        analysis_setup.pop("variable_cmap")
        if "variable_cmap" in analysis_setup
        else "viridis"
    )
    analyses = {
        name: a(variable, variable_cmap=cmap)
        for name, a in DEFAULT_ANALYSES.items()
        if analysis_setup.get(f"skip_{name}", False) is False
    }
    analyses.update(
        {
            f"rel_{ind_variable}": anl.relationship_analysis(variable, ind_variable)
            for ind_variable in relationships
        }
    )

    # Get reference data
    ref = {v: open_reference_data(s) for v, s in (sources | relationships).items()}
    header = f"{variable} | {Path(analysis_setup['path']).name}"
    if dset.is_temporal(ref[variable]):
        header += f" | {ref[variable]['time'].dt.year.min():d}-{ref[variable]['time'].dt.year.max():d}"
    attrs = ref[variable].attrs
    if relationships:
        # Interpolate relationships to the reference grid
        lat = ref[variable][dset.get_dim_name(ref[variable], "lat")]
        lon = ref[variable][dset.get_dim_name(ref[variable], "lon")]
        for v in relationships:
            ref[v] = ref[v].interp(
                {
                    dset.get_dim_name(ref[v], "lat"): lat,
                    dset.get_dim_name(ref[v], "lon"): lon,
                },
                method="nearest",
            )
    ref = xr.merge([v for _, v in ref.items()])

    # Get the model data, only use the measures from the primary variables
    com = [
        model.get_variable(v).drop_vars("cell_measures", errors="ignore")
        for v in set([v for _, a in analyses.items() for v in a.required_variables()])
    ]
    com = xr.merge(com)

    # Run the analyses
    dfs = []
    ds_refs = []
    ds_coms = []
    for name, a in analyses.items():
        try:
            df, ds_ref, ds_com = a(ref, com)
        except Exception:
            raise AnalysisFailure(name, variable, "?", model.name)
        dfs.append(df)
        ds_refs.append(ds_ref)
        ds_coms.append(ds_com)

    # Merge results
    dfs = pd.concat(dfs)
    dfs["source"] = dfs["source"].str.replace("Comparison", model.name)
    ds_ref = xr.merge(ds_refs).pint.dequantify()
    ds_com = xr.merge(ds_coms).pint.dequantify()
    ds_ref.attrs = attrs
    ds_ref.attrs["header"] = header

    return dfs, ds_ref, ds_com


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict) and "sources" not in v:
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def parse_model_setup(yaml_file: str | Path) -> list[ilamb_models.Model]:
    """Parse the model setup file."""
    # parse yaml file
    yaml_file = Path(yaml_file)
    with open(yaml_file) as fin:
        setups = yaml.safe_load(fin)
    assert isinstance(setups, dict)

    # does this bit belong elsewhere?
    abstract_models = {
        name: model
        for name, model in ilamb_models.__dict__.items()
        if (isclass(model) and issubclass(model, ilamb_models.Model))
    }

    # setup models, needs error checking
    models = []
    for name, setup in tqdm(
        setups.items(),
        bar_format="{desc:>28}: {percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt} [{rate_fmt}{postfix}]",
        desc="Initializing Models",
        unit="model",
    ):
        if "type" in setup:
            model_type = setup.pop("type")
            mod = abstract_models[model_type](**setup)
            models.append(mod)
        else:
            paths = setup.pop("paths")
            mod = abstract_models["Model"](name=name, **setup).find_files(paths)
            models.append(mod)
    return models


def parse_benchmark_setup(yaml_file: str | Path) -> dict:
    """Parse the file which is analagous to the old configure file."""
    yaml_file = Path(yaml_file)
    with open(yaml_file) as fin:
        analyses = yaml.safe_load(fin)
    assert isinstance(analyses, dict)
    return analyses


def _clean_pathname(filename: str) -> str:
    """Removes characters we do not want in our paths."""
    invalid_chars = r'[\\/:*?"<>|\s]'
    cleaned_filename = re.sub(invalid_chars, "", filename)
    return cleaned_filename


def _is_leaf(current: dict) -> bool:
    """Is the current item in the nested dictionary a leaf?"""
    if not isinstance(current, dict):
        return False
    if "sources" in current:
        return True
    return False


def _add_path(current: dict, path: Path | None = None) -> dict:
    """Recursively add the nested dictionary headings as a `path` in the leaves."""
    path = Path() if path is None else path
    for key, val in current.items():
        if not isinstance(val, dict):
            continue
        key_path = path / Path(_clean_pathname(key))
        if _is_leaf(val):
            val["path"] = str(key_path)
        else:
            current[key] = _add_path(val, key_path)
    return current


def _to_leaf_list(current: dict, leaf_list: list | None = None) -> list:
    """Recursively flatten the nested dictionary only returning the leaves."""
    leaf_list = [] if leaf_list is None else leaf_list
    for _, val in current.items():
        if not isinstance(val, dict):
            continue
        if _is_leaf(val):
            leaf_list.append(val)
        else:
            _to_leaf_list(val, leaf_list)
    return leaf_list


def _create_paths(current: dict, root: Path = Path("_build")):
    """Recursively ensure paths in the leaves are created."""
    for _, val in current.items():
        if not isinstance(val, dict):
            continue
        if _is_leaf(val):
            if "path" in val:
                (root / Path(val["path"])).mkdir(parents=True, exist_ok=True)
        else:
            _create_paths(val, root)


def _is_complete(work, root) -> bool:
    """Have we already performed this work?"""
    model, setup = work
    scalar_file = Path(root) / setup["path"] / f"{model.name}.csv"
    if scalar_file.is_file():
        return True
    return False


def run_study(study_setup: str, model_setup: str):

    # Define the models
    models = parse_model_setup(model_setup)

    # Some yaml text that would get parsed like a dictionary.
    analyses = parse_benchmark_setup(study_setup)

    # The yaml analysis setup can be as structured as the user needs. We are no longer
    # limited to the `h1` and `h2` headers from ILAMB 2.x. We will detect leaf nodes by
    # the presence of a `sources` dictionary.
    analyses = _add_path(analyses)

    # Various traversal actions
    _create_paths(analyses)

    # Create a list of just the leaves to use in creation all work combinations
    analyses_list = _to_leaf_list(analyses)

    # Create a work list but remove things that are already 'done'
    work_list = [
        w
        for w in itertools.product(models, analyses_list)
        if not _is_complete(w, ilamb3.conf["build_dir"])
    ]

    # Phase I
    if work_list:
        with MPIPoolExecutor() as executor:
            results = tqdm(
                executor.map(work_model_data, work_list),
                bar_format="{desc:>28}: {percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt} [{rate_fmt:>15s}{postfix}]",
                desc="Analysis of Model-Data Pairs",
                unit="pair",
                total=len(work_list),
            )
            results = list(results)

    # Phase 2: plotting
    with MPIPoolExecutor() as executor:
        results = tqdm(
            executor.map(post_model_data, analyses_list),
            bar_format="{desc:>28}: {percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt} [{rate_fmt:>15s}{postfix}]",
            desc="Post-processing",
            unit="analysis",
            total=len(analyses_list),
        )
        results = list(results)
