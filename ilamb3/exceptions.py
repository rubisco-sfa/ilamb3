"""Exceptions for ilamb."""


class ILAMBException(Exception):
    """Exceptions from the intake-esgf package."""


class MissingVariable(ILAMBException):
    """A required variable is missing."""


class VarNotInModel(ILAMBException):
    """This variable is not in the model at all."""


class TemporalOverlapIssue(ILAMBException):
    """This variable does not have the correct temporal overal to be comparable."""
