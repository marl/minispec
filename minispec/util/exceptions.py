#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Exception classes for minispec'''


class MinispecError(Exception):
    '''The root minispec exception class'''
    pass


class ParameterError(MinispecError):
    '''Exception class for mal-formed inputs'''
    pass
