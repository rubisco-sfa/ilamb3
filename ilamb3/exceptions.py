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
