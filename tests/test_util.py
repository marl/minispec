#!/usr/bin/env python
# CREATED:2014-01-18 14:09:05 by Brian McFee <brm2132@columbia.edu>
# unit tests for util routines

# Disable cache
import os
try:
    os.environ.pop('MINISPEC_CACHE_DIR')
except:
    pass

import numpy as np
import pytest
import minispec

from test_core import srand

np.set_printoptions(precision=3)


def test_frame():

    # Generate a random time series
    def __test(P):
        srand()
        frame, hop = P

        y = np.random.randn(8000)
        y_frame = minispec.util.frame(y, frame_length=frame, hop_length=hop)

        for i in range(y_frame.shape[1]):
            assert np.allclose(y_frame[:, i], y[i * hop:(i * hop + frame)])

    for frame in [256, 1024, 2048]:
        for hop_length in [64, 256, 512]:
            yield (__test, [frame, hop_length])


def test_frame_fail():

    __test = pytest.mark.xfail(minispec.util.frame, raises=minispec.ParameterError)

    # First fail, not an ndarray
    yield __test, list(range(10)), 5, 1

    # Second fail: wrong ndims
    yield __test, np.zeros((10, 10)), 5, 1

    # Third fail: too short
    yield __test, np.zeros(10), 20, 1

    # Fourth fail: bad hop length
    yield __test, np.zeros(10), 20, -1

    # Fifth fail: discontiguous input
    yield __test, np.zeros(20)[::2], 10, 1


def test_pad_center():

    def __test(y, n, axis, mode):

        y_out = minispec.util.pad_center(y, n, axis=axis, mode=mode)

        n_len = y.shape[axis]
        n_pad = int((n - n_len) / 2)

        eq_slice = [slice(None)] * y.ndim
        eq_slice[axis] = slice(n_pad, n_pad + n_len)

        assert np.allclose(y, y_out[tuple(eq_slice)])

    @pytest.mark.xfail(raises=minispec.ParameterError)
    def __test_fail(y, n, axis, mode):
        minispec.util.pad_center(y, n, axis=axis, mode=mode)

    for shape in [(16,), (16, 16)]:
        y = np.ones(shape)

        for axis in [0, -1]:
            for mode in ['constant', 'edge', 'reflect']:
                for n in [0, 10]:
                    yield __test, y, n + y.shape[axis], axis, mode

                for n in [0, 10]:
                    yield __test_fail, y, n, axis, mode


def test_fix_length():

    def __test(y, n, axis):

        y_out = minispec.util.fix_length(y, n, axis=axis)

        eq_slice = [slice(None)] * y.ndim
        eq_slice[axis] = slice(y.shape[axis])

        if n > y.shape[axis]:
            assert np.allclose(y, y_out[tuple(eq_slice)])
        else:
            assert np.allclose(y[tuple(eq_slice)], y)

    for shape in [(16,), (16, 16)]:
        y = np.ones(shape)

        for axis in [0, -1]:
            for n in [-5, 0, 5]:
                yield __test, y, n + y.shape[axis], axis


def test_normalize():
    srand()

    def __test_pass(X, norm, axis):
        X_norm = minispec.util.normalize(X, norm=norm, axis=axis)

        # Shape and dtype checks
        assert X_norm.dtype == X.dtype
        assert X_norm.shape == X.shape

        if norm is None:
            assert np.allclose(X, X_norm)
            return

        X_norm = np.abs(X_norm)

        if norm == np.inf:
            values = np.max(X_norm, axis=axis)
        elif norm == -np.inf:
            values = np.min(X_norm, axis=axis)
        elif norm == 0:
            # XXX: normalization here isn't quite right
            values = np.ones(1)

        else:
            values = np.sum(X_norm**norm, axis=axis)**(1./norm)

        assert np.allclose(values, np.ones_like(values))

    @pytest.mark.xfail(raises=minispec.ParameterError)
    def _test_fail(X, norm, axis):
        minispec.util.normalize(X, norm=norm, axis=axis)

    __test_fail = pytest.mark.xfail(_test_fail, raises=minispec.ParameterError)

    for ndims in [1, 2, 3]:
        X = np.random.randn(* ([16] * ndims))

        for axis in range(X.ndim):
            for norm in [np.inf, -np.inf, 0, 0.5, 1.0, 2.0, None]:
                yield __test_pass, X, norm, axis

            for norm in ['inf', -0.5, -2]:
                yield __test_fail, X, norm, axis

        # And test for non-finite failure
        Xnan = X.copy()

        Xnan[0] = np.nan
        yield __test_fail, Xnan, np.inf, 0

        Xinf = X.copy()
        Xinf[0] = np.inf
        yield __test_fail, Xinf, np.inf, 0


def test_normalize_threshold():

    x = np.asarray([[0, 1, 2, 3]])

    def __test(threshold, result):
        assert np.allclose(minispec.util.normalize(x, threshold=threshold),
                           result)

    yield __test, None, [[0, 1, 1, 1]]
    yield __test, 1, [[0, 1, 1, 1]]
    yield __test, 2, [[0, 1, 1, 1]]
    yield __test, 3, [[0, 1, 2, 1]]
    yield __test, 4, [[0, 1, 2, 3]]
    tf = pytest.mark.xfail(__test, raises=minispec.ParameterError)
    yield tf, 0, [[0, 1, 1, 1]]
    yield tf, -1, [[0, 1, 1, 1]]


def test_normalize_fill():

    def __test(fill, norm, threshold, axis, x, result):
        xn = minispec.util.normalize(x,
                                    axis=axis,
                                    fill=fill,
                                    threshold=threshold,
                                    norm=norm)
        assert np.allclose(xn, result), (xn, np.asarray(result))

    x = np.asarray([[0, 1, 2, 3]], dtype=np.float32)

    axis = 0
    norm = np.inf
    threshold = 2
    # Test with inf norm
    yield __test, None, norm, threshold, axis, x, [[0, 1, 1, 1]]
    yield __test, False, norm, threshold, axis, x, [[0, 0, 1, 1]]
    yield __test, True, norm, threshold, axis, x, [[1, 1, 1, 1]]

    # Test with l0 norm
    norm = 0
    yield __test, None, norm, threshold, axis, x, [[0, 1, 2, 3]]
    yield __test, False, norm, threshold, axis, x, [[0, 0, 0, 0]]
    tf = pytest.mark.xfail(__test, raises=minispec.ParameterError)
    yield tf, True, norm, threshold, axis, x, [[0, 0, 0, 0]]

    # Test with l1 norm
    norm = 1
    yield __test, None, norm, threshold, axis, x, [[0, 1, 1, 1]]
    yield __test, False, norm, threshold, axis, x, [[0, 0, 1, 1]]
    yield __test, True, norm, threshold, axis, x, [[1, 1, 1, 1]]

    # And with l2 norm
    norm = 2
    x = np.repeat(x, 2, axis=0)
    s = np.sqrt(2)/2

    # First two columns are left as is, second two map to sqrt(2)/2
    yield __test, None, norm, threshold, axis, x, [[0, 1, s, s], [0, 1, s, s]]

    # First two columns are zeroed, second two map to sqrt(2)/2
    yield __test, False, norm, threshold, axis, x, [[0, 0, s, s], [0, 0, s, s]]

    # All columns map to sqrt(2)/2
    yield __test, True, norm, threshold, axis, x, [[s, s, s, s], [s, s, s, s]]

    # And test the bad-fill case
    yield tf, 3, norm, threshold, axis, x, x

    # And an all-axes test
    axis = None
    threshold = None
    norm = 2
    yield __test, None, norm, threshold, axis, np.asarray([[3, 0], [0, 4]]), np.asarray([[0, 0], [0, 0]])
    yield __test, None, norm, threshold, axis, np.asarray([[3., 0], [0, 4]]), np.asarray([[0.6, 0], [0, 0.8]])


def test_tiny():

    def __test(x, value):

        assert value == minispec.util.tiny(x)

    for x, value in [(1, np.finfo(np.float32).tiny),
                     (np.ones(3, dtype=int), np.finfo(np.float32).tiny),
                     (np.ones(3, dtype=np.float32), np.finfo(np.float32).tiny),
                     (1.0, np.finfo(np.float64).tiny),
                     (np.ones(3, dtype=np.float64), np.finfo(np.float64).tiny),
                     (1j, np.finfo(np.complex128).tiny),
                     (np.ones(3, dtype=np.complex64), np.finfo(np.complex64).tiny),
                     (np.ones(3, dtype=np.complex128), np.finfo(np.complex128).tiny)]:
        yield __test, x, value

