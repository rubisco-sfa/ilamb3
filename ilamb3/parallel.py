import itertools
import shutil
from functools import partial
from pathlib import Path
from traceback import format_exc

import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor, get_comm_workers
from tqdm import tqdm

import ilamb3
import ilamb3.regions as ilr
import ilamb3.run as run


def _perform_work_phase1(work, reference_data, output_path):
    """ """
    # unpack work
    setup, grp = work
    block_name = setup["path"].split("/")[-1]
    local_path = output_path / setup["path"]

    # get parallel worker information for logging
    comm = get_comm_workers()
    rank = comm.Get_rank()
    size = comm.Get_size()
    process_name = MPI.Get_processor_name()

    # setup some model quantities, skip if we have done this
    source_name = "-".join([grp.iloc[0][f] for f in ilamb3.conf["model_name_facets"]])
    csv_file = local_path / f"{source_name}.csv"
    ref_file = local_path / f"Reference{rank}.nc"
    com_file = local_path / f"{source_name}.nc"
    log_file = local_path / f"{source_name}.log"
    log_file.unlink(missing_ok=True)
    if ilamb3.conf["use_cached_results"] and csv_file.is_file() and com_file.is_file():
        return

    # setup the analysis
    setup = run.augment_setup_with_options(setup, reference_data)
    variable = run.select_analysis_variable(setup)
    analyses = run.setup_analyses(setup, local_path)
    transforms = run.setup_transforms(setup)

    # thin out the dataframe to only contain variables we need for this block
    grp = grp[
        grp["variable_id"].isin(
            run.find_related_variables(
                analyses, transforms, setup.get("alternate_vars", [])
            )
        )
    ]
    # if we didn't find anything, just leave
    if len(grp) < 1:
        return

    # try to run the comparison
    try:
        ref = run._load_reference_data(
            reference_data,
            variable,
            setup["sources"],
            setup["relationships"] if "relationships" in setup else {},
            transforms=transforms,
        )
        com = run._load_comparison_data(
            grp,
            variable,
            alternate_vars=setup.get("alternate_vars", []),
            transforms=transforms,
        )
        dfs, ds_ref, ds_com = run.run_analyses(ref, com, analyses)
        dfs["source"] = dfs["source"].str.replace("Comparison", source_name)

        # set a group name optionally, if facets were specified
        if ilamb3.conf["group_name_facets"] is not None:
            if not set(ilamb3.conf["group_name_facets"]).issubset(grp.columns):
                raise ValueError(
                    f"Could not set model group name. You gave these facets {ilamb3.conf['group_name_facets']} but I am not finding them in the comparison dataset dataframe {grp.columns}."
                )
            group_name = grp[ilamb3.conf["group_name_facets"]].apply(
                lambda row: "-".join(row), axis=1
            )
            assert all(group_name == group_name.iloc[0])
            dfs["group"] = str(group_name.iloc[0])

        # Write out artifacts
        dfs.to_csv(csv_file, index=False)
        if not ref_file.is_file():
            ds_ref.to_netcdf(ref_file)
        ds_com.to_netcdf(com_file)

    except Exception:
        with open(log_file, "a") as log:
            log.write(
                f"ILAMB analysis '{block_name}' failed for '{source_name}' on {process_name} ({rank}/{size})\n"
            )
            log.write(format_exc())
        return

    return


def _perform_work_phase2(setup, output_path):
    # unpack work
    block_name = setup["path"].replace("/", " | ")
    local_path = output_path / setup["path"]
    log_file = local_path / "post.log"
    log_file.unlink(missing_ok=True)
    analyses = run.setup_analyses(setup, local_path)

    # get parallel worker information for logging
    comm = get_comm_workers()
    rank = comm.Get_rank()
    size = comm.Get_size()
    process_name = MPI.Get_processor_name()

    # local scalars
    try:
        df_all = [pd.read_csv(str(df)) for df in local_path.glob("*.csv")]
        if not df_all:
            return
        df = pd.concat(df_all).drop_duplicates(
            subset=["source", "region", "analysis", "name"]
        )
        df["region"] = df["region"].astype(str).str.replace("nan", "None")
        df = run.add_overall_score(df)
    except Exception:
        with open(log_file, "a") as log:
            log.write(
                f"ILAMB '{block_name}' failed in post on {process_name} ({rank}/{size})"
            )
            log.write(format_exc())
        return

    try:
        # cleanup local directory
        for f in local_path.glob("Reference*.nc"):
            f.rename(local_path / "Reference.nc")

        # load nc files
        ds_com = {f.stem: xr.load_dataset(str(f)) for f in local_path.glob("*.nc")}
        ds_ref = ds_com.pop("Reference") if "Reference" in ds_com else xr.Dataset()

        # plot
        plt.rcParams.update({"figure.max_open_warning": 0})
        df_plots = run.plot_analyses(df, ds_ref, ds_com, analyses, local_path)
    except Exception:
        with open(log_file, "a") as log:
            log.write(
                f"ILAMB '{block_name}' failed in post on {process_name} ({rank}/{size})"
            )
            log.write(format_exc())
        return

    if ilamb3.conf["debug_mode"] and (local_path / "index.html").is_file():
        return

    # generate an output page
    try:
        ds_ref.attrs["header"] = block_name
        html = run.generate_html_page(df, ds_ref, ds_com, df_plots)
        with open(local_path / "index.html", mode="w") as out:
            out.write(html)
    except Exception:
        with open(log_file, "a") as log:
            log.write(
                f"ILAMB '{block_name}' failed in post on {process_name} ({rank}/{size})"
            )
            log.write(format_exc())


def _start_worker(cfg_path: Path):
    ilamb3.conf.load(cfg_path)
    ilamb_regions = ilr.Regions()
    for source in ilamb3.conf["region_sources"]:
        cat = ilamb3.ilamb_catalog()
        ilamb_regions.add_netcdf(cat.fetch(source))


def run_study_parallel(
    study_setup: str,
    com_datasets: pd.DataFrame,
    ref_datasets: pd.DataFrame,
    output_path: str | Path = "_build",
):
    """
    mpirun -n 4 python -m mpi4py.futures parallel.py
    """
    if ref_datasets.index.name != "key":
        ref_datasets = ref_datasets.set_index("key")
    run.set_model_colors(com_datasets)
    ilamb3.conf["run_mode"] = "batch"
    output_path = Path(output_path)
    analyses = run.parse_benchmark_setup(study_setup)
    analyses = run._add_path(analyses)
    run._create_paths(analyses, output_path)
    analyses_list = run._to_leaf_list(analyses)

    # Save configure so it can be passed to other workers
    shutil.copy(Path(study_setup), output_path / "run.yaml")
    cfg_file = output_path / "cfg.yaml"
    ilamb3.conf.save(cfg_file)

    # Phase 1
    work_list = list(
        itertools.product(
            analyses_list,
            [grp for _, grp in com_datasets.groupby(ilamb3.conf["comparison_groupby"])],
        )
    )
    results = []
    perform_work_phase1 = partial(
        _perform_work_phase1, output_path=output_path, reference_data=ref_datasets
    )
    with MPIPoolExecutor(
        max_workers=len(work_list),
        initializer=_start_worker,
        initargs=(cfg_file,),
    ) as executor:
        results = tqdm(
            executor.map(perform_work_phase1, work_list),
            bar_format="{desc:>20}: {percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt} [{rate_fmt}{postfix}]",
            desc="Running model-data pairs",
            unit="pair",
            total=len(work_list),
            ncols=100,
        )
    list(results)  # trigger the generator

    # Phase 2
    results = []
    perform_work_phase2 = partial(_perform_work_phase2, output_path=output_path)
    with MPIPoolExecutor(
        max_workers=len(analyses_list), initializer=_start_worker, initargs=(cfg_file,)
    ) as executor:
        results = tqdm(
            executor.map(perform_work_phase2, analyses_list),
            bar_format="{desc:>20}: {percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt} [{rate_fmt}{postfix}]",
            desc="Post-process model-data pairs",
            unit="pair",
            total=len(analyses_list),
            ncols=100,
        )
    list(results)  # trigger the generator
