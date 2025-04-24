"""Exceptions for ilamb."""


class ILAMBException(Exception):
    """Exceptions from the intake-esgf package."""


class MissingVariable(ILAMBException):
    """A required variable is missing."""


class VarNotInModel(ILAMBException):
    """This variable is not in the model at all."""


class TemporalOverlapIssue(ILAMBException):
    """This variable does not have the correct temporal overal to be comparable."""


class MissingRegion(ILAMBException):
    """You are trying to use a region that is not registered."""


class NoDatabaseEntry(ILAMBException):
    """The quantile database does not contain that which you are searching."""


class NoSiteDimension(ILAMBException):
    """The dataset/dataarray does not contain a clear site dimension."""


class NoUncertainty(ILAMBException):
    """The dataset/dataarray does not contain uncertainty."""


class AnalysisNotAppropriate(ILAMBException):
    """The dataset/dataarray is not appropriate for an analysis"""


class AnalysisFailure(ILAMBException):
    """
    The analysis function you were running threw an exception.

    Parameters
    ----------
    analysis : str
        The name of the analysis which has failed.
    model : str
        The name of the model whose analysis failed.
    """

    def __init__(self, analysis: str, model: str):  # numpydoc ignore=GL08
        self.analysis = analysis
        self.model = model

    def __str__(self):  # numpydoc ignore=GL08
        return f"The '{self.analysis}' analysis failed for '{self.model}'"
