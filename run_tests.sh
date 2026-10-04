#!/bin/sh
# Run the full MVCC test suite from the terminal. No database or GUI needed.
cd "$(dirname "$0")"
exec python3 -m unittest discover -s tests -v
