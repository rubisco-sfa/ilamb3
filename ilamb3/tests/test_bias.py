import intake
import pandas as pd
import pytest

import ilamb3.analysis as anl
from ilamb3.analysis.base import ILAMBAnalysis
from ilamb3.models.esgf import ModelESGF
from ilamb3.regions import Regions


@pytest.mark.skip(reason="expensive, run manually when needed")
def test_bias_exhaustive():
    # Reference data will come from the ILAMB intake catalog
    cat = intake.open_catalog(
        "https://raw.githubusercontent.com/rubisco-sfa/intake-ilamb/main/ilamb.yaml"
    )
    mod = ModelESGF(source_id="CanESM5", variant_label="r1i1p1f1", grid_label="gn")

    # Setup regional quantiles
    regions = Regions()
    regions.add_netcdf(cat["regions_whittaker_biomes | ILAMB"].read())
    dbase = pd.read_parquet(
        "https://github.com/rubisco-sfa/ILAMB/raw/master/src/ILAMB/data/quantiles_Whittaker_cmip5v6.parquet"
    )

    for key in [
        "lai | AVH15C1",
        "lai | AVHRR",
        "rlus | CERESed4.1",
        "hfss | CLASS",
        "pr | CMAPv1904",
        "tas | CRU4.02",
        "hfls | DOLCE",
        "rhums | ERA5",
        "biomass | ESACCI",
        "gpp | FLUXCOM",
        "evspsbl | GLEAMv3.3a",
        "pr | GPCCv2018",
        "pr | GPCPv2.3",
        "cSoil | HWSD",
        "mrro | LORA",
        "evspsbl | MODIS",
        "biomass | NBCD2000",
        "cSoil | NCSCDV22",
        "biomass | Thurner",
        "biomass | Saatchi2011",
        "biomass | USForest",
        "biomass | XuSaatchi2021",
        "biomass | GEOCARBON",
        "gpp | WECANN",
        "mrsos | WangMao",
    ]:
        variable_id = key.split("|")[0].strip()
        ref = cat[key].read()
        com = mod.get_variable(variable_id)
        assert issubclass(anl.bias_analysis, ILAMBAnalysis)
        run = anl.bias(variable_id)

        for m in ["Collier2018", "RegionalQuantiles"]:
            df, ref_out, com_out = run(
                ref,
                com,
                method=m,
                quantile_dbase=dbase,
            )
            print(f"{key} | bias | {m}")
            print(df)
