from ilamb3.transform.amoc import msftmz_to_rapid
from ilamb3.transform.expression import expression
from ilamb3.transform.gradient import depth_gradient
from ilamb3.transform.ohc import ocean_heat_content
from ilamb3.transform.runoff_sensitivity import runoff_sensitivity
from ilamb3.transform.select import select_depth, select_time
from ilamb3.transform.soilmoisture import soil_moisture_to_vol_fraction
from ilamb3.transform.stratification_index import stratification_index

ALL_TRANSFORMS = {
    "depth_gradient": depth_gradient,
    "expression": expression,
    "msftmz_to_rapid": msftmz_to_rapid,
    "ocean_heat_content": ocean_heat_content,
    "runoff_sensitivity": runoff_sensitivity,
    "select_depth": select_depth,
    "select_time": select_time,
    "soil_moisture_to_vol_fraction": soil_moisture_to_vol_fraction,
    "stratification_index": stratification_index,
}
__all__ = list(ALL_TRANSFORMS.keys())
