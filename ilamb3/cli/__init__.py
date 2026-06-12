from collections.abc import MutableMapping
from pathlib import Path
from typing import Annotated

import pandas as pd
import pooch
import typer

import ilamb3
import ilamb3.load as ill
import ilamb3.meta as meta
import ilamb3.regions as ilr
from ilamb3.run import parse_benchmark_setup, run_study

try:
    from ilamb3.parallel import run_study_parallel

    HAS_MPI4PY = True
except ImportError:
    HAS_MPI4PY = False

try:
    import intake_esgf  # noqa

    HAS_INTAKE = True
except ImportError:
    HAS_INTAKE = False

app = typer.Typer(name="ilamb", no_args_is_help=True)


def _flatten_dict_gen(d, parent_key, sep):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "."):
    """
    Flatten a nested dictionary by composing keys with a separator.
    """
    return dict(_flatten_dict_gen(d, parent_key, sep))


def parse_registry_keys(config: Path) -> list[str]:
    """
    Return all keys from the configure file that are part of a ilamb registry.
    """
    setup = parse_benchmark_setup(config)
    keys = list(
        set(
            [val for _, val in flatten_dict(setup).items() if isinstance(val, str)]
        ).intersection(
            set(
                ilamb3.ilamb_catalog().registry_files
                + ilamb3.iomb_catalog().registry_files
                + ilamb3.ilamb3_catalog().registry_files
            )
        )
    )
    return keys


def get_local_path(key: str, catalogs: list[pooch.Pooch]) -> Path | None:
    """
    Return the local path of a file in our registry, if the file exists.
    """
    for cat in catalogs:
        if key in cat.registry_files:
            local = cat.abspath / key
            if local.is_file():
                return local
            else:
                return None
    return None


def fetch_key(key: str, catalogs: list[pooch.Pooch]) -> None:
    """
    Fetch from a list of pooch catalogs.
    """
    for cat in catalogs:
        if key in cat.registry_files:
            cat.fetch(key)


def form_reference_dataframe(keys: list[str]) -> pd.DataFrame:
    """
    Create a dataframe for the keys found in ilamb catalogs.
    """
    catalogs = [ilamb3.ilamb_catalog(), ilamb3.iomb_catalog(), ilamb3.ilamb3_catalog()]
    df = pd.DataFrame(
        [{"key": key, "path": get_local_path(key, catalogs)} for key in keys]
    )
    df = df.set_index("key")
    return df


@app.command(help="Run a benchmarking analysis.")
def run(
    config: Annotated[Path, typer.Argument(help="The benchmark study yaml file.")],
    model_db: Annotated[
        list[Path],
        typer.Option(
            help="The model database file(s) in CSV format. Use the option multiple times to specify multiple files."
        ),
    ],
    region: Annotated[
        list[str] | None,
        typer.Option(
            help="The region label over which the analysis is run. Use the option multiple times to specify multiple regions."
        ),
    ] = None,
    region_source: Annotated[
        list[Path] | None,
        typer.Option(
            help="The file or key which provides more regions over which the analysis may be run. Use the option multiple times to specify multiple files."
        ),
    ] = None,
    output_path: Annotated[
        Path,
        typer.Option(
            help="The path in which ilamb3 will write the benchmark study output."
        ),
    ] = Path("_build"),
    cache: Annotated[
        bool,
        typer.Option(help="Enable to use cached intermediate files in the analysis."),
    ] = True,
    central_longitude: Annotated[
        float,
        typer.Option(
            help="The longitude around which the global map plots will be centered."
        ),
    ] = 0.0,
    title: Annotated[
        str,
        typer.Option(
            help="A title to be displayed on the benchmarking study results page."
        ),
    ] = "Benchmarking Results",
    main_region: Annotated[
        str | None,
        typer.Option(help="A region label that will be displayed first in the output."),
    ] = None,
):
    ilamb3.conf.reset()
    region_sources = (
        list() if region_source is None else [str(r) for r in region_source]
    )
    ilamb3.conf["region_sources"] = region_sources
    for source in region_sources:
        ilr.Regions().add_netcdf(ill.load_key_or_filename(str(source)))

    # set options
    ilamb3.conf.set(
        regions=region,
        use_cached_results=cache,
        use_uncertainty=True,
        plot_central_longitude=central_longitude,
        comparison_groupby=["source_id", "grid_label"],
        model_name_facets=["source_id"],
        global_region=main_region,
    )

    # load local databases
    df_ref = form_reference_dataframe(parse_registry_keys(config))
    if df_ref["path"].isnull().any():
        sconfig = str(config)
        raise ValueError(
            f"Some of the reference data keys you specify in {sconfig=} is not locally available: {list(df_ref[df_ref['path'].isnull()].index)}.\nRun `ilamb fetch {sconfig}` to download files locally."
        )
    df_com = pd.concat([pd.read_csv(f) for f in model_db])
    df_com = ill.add_frequency_column(df_com)

    # execute
    if HAS_MPI4PY:
        run_study_parallel(
            str(config),
            df_com,
            df_ref,
            output_path=output_path,
        )
    else:
        run_study(
            str(config),
            df_com,
            ref_datasets=df_ref,
            output_path=output_path,
        )
    try:
        meta.generate_dashboard_page(output_path, page_title=title)
    except Exception:
        pass


@app.command(help="Fetch reference data if part of an ILAMB catalog.")
def fetch(
    config: Annotated[
        Path | None, typer.Argument(help="The benchmark study yaml file.")
    ] = None,
    key: Annotated[
        list[str] | None,
        typer.Option(
            help="Additional keys to fetch from the catalog. Especially useful to pre-download region files prior to running the study. Use the option multiple times to specify multiple files."
        ),
    ] = None,
):
    catalogs = [ilamb3.ilamb_catalog(), ilamb3.iomb_catalog(), ilamb3.ilamb3_catalog()]
    keys = []
    if config is not None:
        df = form_reference_dataframe(parse_registry_keys(config))
        df = df[df["path"].isnull()]  # just the datasets we don't have
        keys += [str(key) for key in df.index]
    keys += key or []
    for download_key in keys:
        fetch_key(download_key, catalogs)


@app.command(help="What went wrong in the run?")
def debug(
    output_path: Annotated[
        Path,
        typer.Argument(
            help="The path in which ilamb3 will write the benchmark study output."
        ),
    ],
):
    for root, _, files in output_path.walk():
        logs_no_csvs = [
            f
            for f in files
            if f.endswith(".log")
            and f"{(root / f).stem}.csv" not in files
            and (root / f).stat().st_size > 0
        ]
        if logs_no_csvs:
            print(f"\n{root}")
            for log in logs_no_csvs:
                with open(root / log) as fin:
                    print("  -", Path(log).stem, fin.readlines()[-1].strip())


@app.command(help="Search/download model data from ESGF.")
def esgf(
    config: Annotated[Path, typer.Argument(help="The benchmark study yaml file.")],
    source_id: Annotated[
        list[str] | None,
        typer.Option(
            help="The source_id for which you would like to limit the search. Use the option multiple times to specify multiple source_id's."
        ),
    ] = None,
    variables: Annotated[
        bool,
        typer.Option(
            help="Enable to see only a list of variables used in the configure."
        ),
    ] = False,
    counts: Annotated[
        bool,
        typer.Option(help="Enable to see only counts of datasets found in the search."),
    ] = False,
) -> None:
    if not HAS_INTAKE:
        raise ImportError(
            "The 'ilamb esgf' command requires that you install the 'esgf' extras."
        )
    import intake_esgf

    import ilamb3.esgf as ile

    intake_esgf.conf.set(print_log_on_error=True)

    df_info = ile.get_configure_variables(config)
    if variables:
        print(df_info["variable_id"].to_list())
        return
    cat = ile.get_esgf_catalog(df_info, source_id)
    if counts:
        print(cat.model_groups().to_string())
        return
    path_dict = ile.download_esgf_catalog(df_info, cat)

    # output CSVs
    def _path_dict_to_pandas(
        facets: list[str], path_dict: dict[str, list[str]]
    ) -> pd.DataFrame:
        df = []
        for key, paths in path_dict.items():
            for path in paths:
                row = {col: value for col, value in zip(facets, key.split("."))}
                row["path"] = str(path)
                df.append(row)
        df = pd.DataFrame(df)
        return df

    if source_id is None:
        out = _path_dict_to_pandas(cat.project.master_id_facets(), path_dict)
        out.to_csv("model_data.csv", index=False)
        return
    for s in source_id:
        out = _path_dict_to_pandas(
            cat.project.master_id_facets(),
            {key: val for key, val in path_dict.items() if s in key},
        )
        out.to_csv(f"{s}.csv", index=False)


if __name__ == "__main__":
    app()
