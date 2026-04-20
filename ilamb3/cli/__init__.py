from collections.abc import MutableMapping
from pathlib import Path
from typing import Annotated

import pandas as pd
import pooch
import typer

import ilamb3
import ilamb3.meta as meta
import ilamb3.regions as ilr
from ilamb3.run import parse_benchmark_setup, run_study

try:
    from ilamb3.parallel import run_study_parallel

    HAS_MPI4PY = True
except ImportError:
    HAS_MPI4PY = False


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
        set([val for _, val in flatten_dict(setup).items()]).intersection(
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
    regions: Annotated[
        str | None,
        typer.Option(
            help="The region label over which the analysis is run. Use the option multiple times to specify multiple regions."
        ),
    ] = None,
    region_source: Annotated[
        list[Path] | None,
        typer.Option(
            help="The file (text or netCDF) which provides more regions over which the analysis may be run. Use the option multiple times to specify multiple files."
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
    if region_source is not None:
        cat = ilamb3.ilamb_catalog()
        for source in region_source:
            ilr.Regions().add_netcdf(cat.fetch(source))
        ilamb3.conf["region_sources"] = region_source

    # set options
    ilamb3.conf.set(
        regions=regions,
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
        config = str(config)
        raise ValueError(
            f"Some of the reference data keys you specify in {config=} is not locally available: {list(df_ref[df_ref['path'].isnull()].index)}.\nRun `ilamb fetch {config}` to download files locally."
        )
    df_com = pd.concat([pd.read_csv(f) for f in model_db])

    # execute
    if HAS_MPI4PY:
        run_study_parallel(
            config,
            df_com,
            df_ref,
            output_path=output_path,
        )
    else:
        run_study(
            config,
            df_com,
            ref_datasets=df_ref,
            output_path=output_path,
        )
    try:
        meta.generate_dashboard_page(output_path, page_title=title)
    except Exception:
        pass
    try:
        meta.generate_directory_of_dashboards(output_path.parent)
    except Exception:
        pass


@app.command(help="Fetch reference data if part of an ILAMB catalog.")
def fetch(
    config: Annotated[Path, typer.Argument(help="The benchmark study yaml file.")],
):
    catalogs = [ilamb3.ilamb_catalog(), ilamb3.iomb_catalog(), ilamb3.ilamb3_catalog()]
    df = form_reference_dataframe(parse_registry_keys(config))
    df = df[df["path"].isnull()]
    for key in df.index:
        fetch_key(key, catalogs)


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


if __name__ == "__main__":
    app()
