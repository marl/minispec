#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feature extraction
==================

Spectral features
-----------------

.. autosummary::
    :toctree: generated/

    melspectrogram
"""
from .spectral import *  # pylint: disable=wildcard-import

__all__ = [_ for _ in dir() if not _.startswith('_')]
