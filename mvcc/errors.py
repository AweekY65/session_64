"""Exception types raised by the MVCC store."""


class MVCCError(Exception):
    """Base class for all MVCC related errors."""


class TransactionStateError(MVCCError):
    """Raised when an operation is applied to a transaction in a bad state."""


class WriteConflictError(MVCCError):
    """Raised when a write-write conflict is detected.

    The transaction that observes this error has been (or must be) aborted;
    it can never silently commit.
    """
