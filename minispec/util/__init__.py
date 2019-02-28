#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities
=========

Array operations
----------------
.. autosummary::
    :toctree: generated/

    frame
    pad_center
    fix_length

    normalize

    tiny


Input validation
----------------
.. autosummary::
    :toctree: generated/

    valid_audio

"""

from .utils import *  # pylint: disable=wildcard-import
from . import exceptions

__all__ = [_ for _ in dir() if not _.startswith('_')]
