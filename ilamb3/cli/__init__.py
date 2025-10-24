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
from ilamb3.run import parse_benchmark_setup, run_study

app = typer.Typer(name="ilamb", no_args_is_help=True)


def _dataframe_reference(
    root: Path = Path("/home/nate/.cache/ilamb3/"),
    cache_file: Path = Path("df_reference.csv"),
) -> pd.DataFrame:
    if cache_file.exists():
        df = pd.read_csv(cache_file)
        df = df.set_index("key")
        return df
    df = []
    for dirpath, _, files in root.walk():
        for fname in files:
            if not fname.endswith(".nc"):
                continue
            path = (dirpath / fname).absolute()
            df.append(
                {
                    "key": str(path.parent).split("/")[-1] + f"/{path.name}",
                    "path": str(path),
                }
            )
    df = pd.DataFrame(df)
    df.to_csv(cache_file, index=False)
    df = df.set_index("key")
    return df


def _dataframe_cmip(
    root: Path = Path("/home/nate/esgf-data/CMIP6/CMIP/"),
    cache_file: Path = Path("df_cmip.csv"),
) -> pd.DataFrame:
    if cache_file.exists():
        df = pd.read_csv(cache_file)
        return df
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

    # set options
    ilamb3.conf.set(
        regions=regions,
        use_cached_results=cache,
        use_uncertainty=True,
        plot_central_longitude=central_longitude,
    )

    # load local databases, need a better way
    df_ref = _dataframe_reference()
    df_com = _dataframe_cmip()
    df_com = df_com[df_com["source_id"] == "CanESM5"]
    df_com = df_com[df_com["member_id"] == "r1i1p1f1"]
    df_com = df_com[
        df_com["variable_id"].apply(lambda v: v not in ["areacello", "sftof"])
    ]

    # add a few CESM2 variables that CanESM5 does not have
    df = _dataframe_cmip()
    df = df[df["source_id"] == "CESM2"]
    df = df[
        df["variable_id"].apply(
            lambda v: v in ["areacella", "sftlf", "fBNF", "burntFractionAll"]
        )
    ]
    df_com = pd.concat([df_com, df])

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
        for reg in registries:
            if source in reg.registry_files:
                reg.fetch(source)


if __name__ == "__main__":
    app()
