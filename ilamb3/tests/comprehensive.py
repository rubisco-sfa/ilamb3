"""
Functions used for more comprehensive tests of ilamb3.
"""

import logging
from traceback import format_exc

import intake_esgf
import pandas as pd
import xarray as xr
from intake_esgf import ESGFCatalog
from tqdm import tqdm

from ilamb3.analysis.base import ILAMBAnalysis
from ilamb3.models.esgf import ModelESGF


def get_logger() -> logging.Logger:
    """
    Setup the location and logging for this package.

    Returns
    -------
    logginer.Logger
        The logging object configured for use in the tests.
    """

    # We need a named logger to avoid other packages that use the root logger
    logger = logging.getLogger("ilamb3")
    if not logger.handlers:
        # Now setup the file into which we log relevant information
        file_handler = logging.FileHandler("test.log")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

    # This is probably wrong, but when I log from my logger it logs from parent also
    logger.parent.handlers = []
    return logger


def get_ilamb_models() -> pd.Series:
    """
    Return the model groups present in ESGF which are applicable to ilamb.

    Returns
    -------
    pd.Series
        The series whose index consists of `source_id`, `member_id`, and `grid_label`.
    """
    variables = ["cSoil", "cVeg", "gpp", "lai", "nbp"]
    with intake_esgf.conf.set(additional_df_cols=[]):
        cat = (
            ESGFCatalog()
            .search(
                experiment_id="historical",
                frequency="mon",
                variable_id=variables,
            )
            .remove_incomplete(
                lambda df: (
                    False
                    if (set(variables) - set(df["variable_id"].unique()))
                    else True
                )
            )
            .remove_ensembles()
        )
    return cat.model_groups()


def comprehensive_test_harness(
    ref: xr.Dataset,
    model_groups: pd.Series,
    analysis: ILAMBAnalysis,
    **analysis_kwargs: dict,
) -> pd.DataFrame:

    logger = get_logger()
    dfs = []
    for model, variant, grid in tqdm(model_groups.index, desc="  Confronting models"):
        try:
            com = xr.merge(
                [
                    ModelESGF(model, variant, grid).get_variable(var)
                    for var in analysis.required_variables()
                ]
            )
        except Exception:
            logger.debug(
                f"({model}, {variant}, {grid}) model initialization failure\n{format_exc()}"
            )
            continue
        try:
            df, _, _ = analysis(ref, com, **analysis_kwargs)
        except Exception:
            logger.debug(
                f"({model}, {variant}, {grid}) analysis method failure\n{format_exc()}"
            )
            continue
        logger.info(f"({model}, {variant}, {grid}) completed")
        df["source"] = df["source"].apply(
            lambda c: model if c == "Comparison" else "Reference"
        )
        dfs.append(df)
    df = pd.concat(dfs).set_index(["source", "name"]).drop_duplicates()
    return df
