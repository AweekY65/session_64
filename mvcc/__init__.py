"""Local MVCC transaction manager.

All record versions, transaction states, commit timestamps and logs live
in local memory or local files only. No external service is required.
"""

from .errors import MVCCError, TransactionStateError, WriteConflictError
from .store import MVCCStore, Transaction

__all__ = [
    "MVCCStore",
    "Transaction",
    "MVCCError",
    "TransactionStateError",
    "WriteConflictError",
]
