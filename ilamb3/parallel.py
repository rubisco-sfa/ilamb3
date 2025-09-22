import itertools
from functools import partial
from pathlib import Path

import pandas as pd
from loguru import logger
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor, get_comm_workers
from tqdm import tqdm

import ilamb3
import ilamb3.run as run


def _perform_work(work, reference_data, output_path):
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
    log_id = logger.add(log_file, backtrace=True, diagnose=True)
    # logger.info(
    # )
    if ilamb3.conf["use_cached_results"] and csv_file.isfile():
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
        logger.exception(
            f"ILAMB analysis '{block_name}' failed for '{source_name}' on {process_name} ({rank}/{size})"
        )
        return

    logger.remove(log_id)

    return


def run_study_parallel(
    study_setup: str,
    com_datasets: pd.DataFrame,
    ref_datasets: pd.DataFrame,
    output_path: str | Path = "_build",
):
    """
    mpirun -n 4 python -m mpi4py.futures parallel.py
    """
    ilamb3.conf["run_mode"] = "batch"
    output_path = Path(output_path)
    analyses = run.parse_benchmark_setup(study_setup)
    analyses = run._add_path(analyses)
    run._create_paths(analyses, output_path)
    analyses_list = run._to_leaf_list(analyses)
    work_list = list(
        itertools.product(
            analyses_list,
            [grp for _, grp in com_datasets.groupby(ilamb3.conf["comparison_groupby"])],
        )
    )
    if ref_datasets.index.name != "key":
        ref_datasets = ref_datasets.set_index("key")
    perform_work = partial(
        _perform_work, output_path=output_path, reference_data=ref_datasets
    )
    with MPIPoolExecutor(max_workers=len(work_list)) as executor:
        results = tqdm(
            executor.map(perform_work, work_list),
            bar_format="{desc:>20}: {percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt} [{rate_fmt}{postfix}]",
            desc="Running model-data pairs",
            unit="pair",
            total=len(work_list),
            ncols=100,
        )

    # Do something with the results, here just print
    print(list(results))
