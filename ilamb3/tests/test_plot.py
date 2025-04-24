import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ilamb3.plot import plot_map


def test_polar():
    da = xr.DataArray(
        np.random.rand(4, 10),
        coords={"lat": np.linspace(65, 90, 4), "lon": np.linspace(-180, 180, 10)},
        dims=["lat", "lon"],
        attrs={"units": "1"},
    )
    plot_map(da)
    plt.close()
    da = xr.DataArray(
        np.random.rand(4, 10),
        coords={"lat": np.linspace(-90, 65, 4), "lon": np.linspace(-180, 180, 10)},
        dims=["lat", "lon"],
        attrs={"units": "1"},
    )
    plot_map(da)
    plt.close()
