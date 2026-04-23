import inspect

from ilamb3.transform.amoc import msftmz_to_rapid  # noqa
from ilamb3.transform.base import ILAMBTransform
from ilamb3.transform.expression import expression  # noqa
from ilamb3.transform.gradient import depth_gradient  # noqa
from ilamb3.transform.integrate import (  # noqa
    integrate_depth,
    integrate_space,
    integrate_time,
)
from ilamb3.transform.label import apply_label  # noqa
from ilamb3.transform.mask import mask_condition  # noqa
from ilamb3.transform.ohc import ocean_heat_content  # noqa
from ilamb3.transform.permafrost import (  # noqa
    active_layer_thickness,
    permafrost_extent,
)
from ilamb3.transform.runoff_sensitivity import runoff_sensitivity  # noqa
from ilamb3.transform.select import (  # noqa
    select_depth,
    select_lat,
    select_lon,
    select_time,
)
from ilamb3.transform.soilmoisture import soil_moisture_to_vol_fraction  # noqa
from ilamb3.transform.stratification_index import stratification_index  # noqa

ALL_TRANSFORMS = {
    key: fnc
    for key, fnc in vars().items()
    if inspect.isclass(fnc) and issubclass(fnc, ILAMBTransform)
}

__all__ = list(ALL_TRANSFORMS.keys())
