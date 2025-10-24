import functools
import hashlib
from pathlib import Path

import pandas as pd
import xarray as xr


def dataset_cache(func):
    def _create_hash(*args):
        string_to_hash = ""
        for arg in args:
            if isinstance(arg, xr.Dataset):
                string_to_hash += str(arg.coords) + str(arg.attrs)
            else:
                string_to_hash += str(arg)
        return hashlib.sha256(string_to_hash.encode()).hexdigest()

    def _get_cache_dir() -> Path:
        cache_dir = Path.home() / f".cache/ilamb3/{func.__name__}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        hash = _create_hash(*args)
        cache_file = _get_cache_dir() / hash
        if cache_file.is_file():
            return xr.open_dataset(cache_file)
        result = func(*args, **kwargs)
        if not isinstance(result, xr.Dataset):
            raise ValueError(
                f"You decorated the function '{func.__name__}' but the return type is not a xarray.Dataset: {type(result)}"
            )
        result.to_netcdf(cache_file)
        return result

    return wrapper


def dataframe_cache(func):
    def _create_hash(*args):
        string_to_hash = ""
        for arg in args:
            if isinstance(arg, xr.Dataset):
                string_to_hash += str(arg.coords) + str(arg.attrs)
            else:
                string_to_hash += str(arg)
        return hashlib.sha256(string_to_hash.encode()).hexdigest()

    def _get_cache_dir() -> Path:
        cache_dir = Path.home() / f".cache/ilamb3/{func.__name__}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        hash = _create_hash(*args)
        cache_file = _get_cache_dir() / hash
        if cache_file.is_file():
            return pd.read_feather(cache_file)
        result = func(*args, **kwargs)
        if not isinstance(result, pd.DataFrame):
            raise ValueError(
                f"You decorated the function '{func.__name__}' but the return type is not a pd.DataFrame: {type(result)}"
            )
        result.to_feather(cache_file)
        return result

    return wrapper
