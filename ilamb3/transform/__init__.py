from ilamb3.transform.amoc import msftmz_to_rapid
from ilamb3.transform.soilmoisture import soil_moisture_to_vol_fraction

ALL_TRANSFORMS = {
    "soil_moisture_to_vol_fraction": soil_moisture_to_vol_fraction,
    "msftmz_to_rapid": msftmz_to_rapid,
}
__all__ = ["soil_moisture_to_vol_fraction", "msftmz_to_rapid"]
