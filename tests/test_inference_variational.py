"""
Inside the main directory, run with:
`python -m pytest tests/test_inference.py`
"""
# pylint: disable=missing-function-docstring,invalid-name,too-many-locals,logging-fstring-interpolation,f-string-without-interpolation
import logging
import numpy as np
import numpy.testing as npt
import cgm
logging.basicConfig(level=logging.INFO)
logging.getLogger('asyncio').setLevel(logging.WARNING)


