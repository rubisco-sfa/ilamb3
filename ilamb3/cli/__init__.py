import os
from pathlib import Path

import pandas as pd
import typer

try:
    from ilamb3.parallel import run_study_parallel

    HAS_MPI4PY = True
except ImportError:
    HAS_MPI4PY = False

import ilamb3
import ilamb3.meta as meta
import ilamb3.regions as ilr
from ilamb3.run import parse_benchmark_setup, run_study

app = typer.Typer(name="ilamb", no_args_is_help=True)


def _dataframe_reference(
    root: Path = Path().home() / ".cache/ilamb3/",
    cache_file: Path = Path("df_reference.csv"),
) -> pd.DataFrame:
    if cache_file.exists():
        df = pd.read_csv(cache_file)
        df = df.set_index("key")
        return df
    if "ILAMB_ROOT" in os.environ:
        root = Path(os.environ["ILAMB_ROOT"])
    df = []
    for dirpath, _, files in root.walk():
        for fname in files:
            if not fname.endswith(".nc"):
                continue
            path = dirpath / fname
            df.append(
                {
                    "key": str(Path(*path.relative_to(root).parts[1:])),
                    "path": str(path.absolute()),
                }
            )
    df = pd.DataFrame(df)
    df.to_csv(cache_file, index=False)
    df = df.set_index("key")
    return df


def _dataframe_cmip(
    root: Path | None = None,
    cache_file: Path = Path("df_cmip.csv"),
) -> pd.DataFrame:
    if cache_file.exists():
        df = pd.read_csv(cache_file)
        return df
    if root is None:
        if "ESGF_ROOT" in os.environ:
            root = Path(os.environ["ESGF_ROOT"])
        else:
            root = Path.home() / ".esgf"
    df = []
    for dirpath, _, files in root.walk():
        for fname in files:
            if not fname.endswith(".nc"):
                continue
            path = str((dirpath / fname).absolute())
            df.append(
                {
                    "mip_era": path.split("/")[-11],
                    "activity_id": path.split("/")[-10],
                    "institution_id": path.split("/")[-9],
                    "source_id": path.split("/")[-8],
                    "experiment_id": path.split("/")[-7],
                    "member_id": path.split("/")[-6],
                    "table_id": path.split("/")[-5],
                    "variable_id": path.split("/")[-4],
                    "grid_label": path.split("/")[-3],
                    "path": path,
                }
            )
    df = pd.DataFrame(df)
    df.to_csv(cache_file, index=False)
    return df


@app.command(help="Run a benchmarking analysis")
def run(
    config: Path,
    regions: str | None = None,
    region_sources: list[str] | None = None,
    df_comparison: Path | None = None,
    output_path: Path = Path("_build"),
    cache: bool = True,
    central_longitude: float = 0.0,
    title: str = "Benchmarking Results",
):
    # by default, we run over the None region, that is, no regional reduction
    if regions is None:
        regions = [None]
    else:
        regions = [None if r.lower() == "none" else r for r in regions.split(",")]
    if region_sources is not None:
        cat = ilamb3.ilamb_catalog()
        for source in region_sources:
            ilr.Regions().add_netcdf(cat.fetch(source))
        ilamb3.conf["region_sources"] = region_sources

    # set options
    ilamb3.conf.set(
        regions=regions,
        use_cached_results=cache,
        use_uncertainty=True,
        plot_central_longitude=central_longitude,
        comparison_groupby=["source_id", "grid_label"],
        model_name_facets=["source_id"],
    )

    # load local databases, need a better way
    df_ref = _dataframe_reference()
    if df_comparison is None:
        df_com = _dataframe_cmip()
    else:
        df_com = pd.read_csv(df_comparison)

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


@app.command(help="Fetch reference data")
def fetch(config: Path):
    def _extract_sources(current: dict):
        """Recursively extract the source keys."""
        sources = []
        for _, val in current.items():
            if not isinstance(val, dict):
                continue
            if "sources" in val:
                sources += [key for _, key in val["sources"].items()]
            elif "relationships" in val:
                sources += [key for _, key in val["relationships"].items()]
            else:
                sources += _extract_sources(val)
        return sources

    setup = parse_benchmark_setup(config)
    sources = list(set(_extract_sources(setup)))
    registries = [
        ilamb3.ilamb_catalog(),
        ilamb3.iomb_catalog(),
        ilamb3.ilamb3_catalog(),
    ]
    for source in sources:
        found = False
        for reg in registries:
            if source in reg.registry_files:
                reg.fetch(source)
                found = True
        if not found:
            raise ValueError(f"Could not find '{source}' in the data registries.")


if __name__ == "__main__":
    app()
