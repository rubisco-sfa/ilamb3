"""A class for abstracting the variables associated with an earth system model."""

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import xarray as xr
from esmodels.exceptions import ParsingError, VarNotInModel


def free_symbols(expression: str) -> list[str]:
    """
    Return the free symbols of the expression.

    Parameters
    ----------
    expression : str
        The expression.

    Returns
    -------
    list[str]
        A list of the free symbols in the expression.
    """
    return re.findall(r"\w+", expression)


@dataclass
class Model:

    name: str
    color: tuple[float] = (0.0, 0.0, 0.0)
    variables: dict = field(init=False, repr=False, default_factory=dict)
    synonyms: dict = field(init=False, repr=False, default_factory=dict)

    def find_files(
        self,
        path: str | Sequence[str],
    ) -> Self:
        """
        Find netCDF files and variables in the provided path(s).

        Parameters
        ----------
        path : str or a Sequence of str
            The path(s) in which to search for netCDF files.
        """
        if isinstance(path, str):
            path = [path]
        # Walk through all the paths given (following links)...
        for file_path in path:
            for root, _, files in os.walk(file_path, followlinks=True):
                for filename in files:
                    filepath = Path(root) / Path(filename)
                    if not filepath.suffix == ".nc":
                        continue
                    # Open the dataset and for each variable, insert a key into the
                    # variables dict appending the path where it was found.
                    with xr.open_dataset(filepath) as ds:
                        for key in ds.variables.keys():
                            if key not in self.variables:
                                self.variables[key] = []
                            self.variables[key].append(str(filepath))
        # Sort the file lists to make opening easier later
        self.variables = {var: sorted(paths) for var, paths in self.variables.items()}
        return self

    def add_synonym(self, expression: str) -> Self:
        """
        Associate variable synonyms or expressions with this model.

        Parameters
        ----------
        expression : str
            An expression for the synonym of the form, `new_var = var_in_model`.
        """
        assert isinstance(expression, str)
        if expression.count("=") != 1:
            raise ValueError("Add a synonym by providing a string of the form 'a = b'")
        key, expression = expression.split("=")
        # Check that the free symbols of the expression are variables in this model
        for arg in free_symbols(expression):
            assert arg in self.variables
        self.synonyms[key.strip()] = expression.strip()
        return self

    def get_variable(
        self,
        name: str,
    ) -> xr.Dataset:
        """
        Search the model database for the specified variable.

        Parameters
        ----------
        name : str
            The name of the variable we wish to load.

        Returns
        -------
        xr.Dataset
            The xarray Dataset.
        """

        def _load(symbols: list[str]):
            dsd = (
                {
                    sym: (
                        xr.open_dataset(self.variables[sym][0])
                        if len(self.variables[sym]) == 1
                        else xr.open_mfdataset(self.variables[sym])
                    )[sym]
                    for sym in symbols
                },
            )
            return dsd[0]

        if name in self.variables:
            return xr.Dataset(_load([name]))
        if name in self.synonyms:
            symbols = free_symbols(self.synonyms[name])
            ds = _load(symbols)
            expr = str(self.synonyms[name])
            for sym in symbols:
                expr = expr.replace(sym, f"ds['{sym}']")
            try:
                ds = xr.Dataset({name: eval(expr)})
            except Exception as ex:
                print(ex)
                raise ParsingError(self.name, name, self.synonyms[name])
            return ds
        raise VarNotInModel(name, self.synonyms[name])
