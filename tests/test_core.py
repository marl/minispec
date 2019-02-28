#!/usr/bin/env python
# CREATED:2013-03-08 15:25:18 by Brian McFee <brm2132@columbia.edu>
#  unit tests for minispec core (__init__.py)
#

from __future__ import print_function
# Disable cache
import os
try:
    os.environ.pop('MINISPEC_CACHE_DIR')
except:
    pass

import minispec
import glob
import numpy as np
import scipy.io
import six
import pytest
import resampy
import audioread


# -- utilities --#
def files(pattern):
    test_files = glob.glob(pattern)
    test_files.sort()
    return test_files


def srand(seed=628318530):
    np.random.seed(seed)
    pass


def load(infile):
    return scipy.io.loadmat(infile, chars_as_strings=True)


def load_audio(path, sr=22050, mono=True, offset=0.0, duration=None,
         dtype=np.float32, res_type='kaiser_best'):
    y = []
    with audioread.audio_open(os.path.realpath(path)) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels

        s_start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.round(sr_native * duration))
                               * n_channels)

        n = 0

        for frame in input_file:
            frame = buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[:s_end - n_prev]

            if n_prev <= s_start <= n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev):]

            # tack on the current frame
            y.append(frame)

    if y:
        y = np.concatenate(y)

        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
            if mono:
                y = to_mono(y)

        if sr is not None:
            y = resample(y, sr_native, sr, res_type=res_type)

        else:
            sr = sr_native

    # Final cleanup for dtype and contiguity
    y = np.ascontiguousarray(y, dtype=dtype)

    return (y, sr)


def to_mono(y):
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    return y


def resample(y, orig_sr, target_sr, res_type='kaiser_best', fix=True, scale=False, **kwargs):
    if orig_sr == target_sr:
        return y

    ratio = float(target_sr) / orig_sr

    n_samples = int(np.ceil(y.shape[-1] * ratio))

    y_hat = resampy.resample(y, orig_sr, target_sr, filter=res_type, axis=-1)

    if fix:
        y_hat = minispec.util.fix_length(y_hat, n_samples, **kwargs)

    if scale:
        y_hat /= np.sqrt(ratio)

    return np.ascontiguousarray(y_hat, dtype=y.dtype)


def buf_to_float(x, n_bytes=2, dtype=np.float32):
    # Invert the scale of the data
    scale = 1./float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = '<i{:d}'.format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)


def test_stft():

    def __test(infile):
        DATA = load(infile)

        # Load the file
        (y, sr) = load_audio(os.path.join('tests', DATA['wavfile'][0]),
                                          sr=None, mono=True)

        if DATA['hann_w'][0, 0] == 0:
            # Set window to ones, swap back to nfft
            window = np.ones
            win_length = None

        else:
            window = 'hann'
            win_length = DATA['hann_w'][0, 0]

        # Compute the STFT
        D = minispec.stft(y,
                         n_fft=DATA['nfft'][0, 0].astype(int),
                         hop_length=DATA['hop_length'][0, 0].astype(int),
                         win_length=win_length,
                         window=window,
                         center=False)

        # conjugate matlab stft to fix the ' vs .' bug
        assert np.allclose(D, DATA['D'].conj())

    for infile in files(os.path.join('tests', 'data', 'core-stft-*.mat')):
        yield (__test, infile)


def test_magphase():

    (y, sr) = load_audio(os.path.join('tests', 'data', 'test1_22050.wav'))

    D = minispec.stft(y)

    S, P = minispec.magphase(D)

    assert np.allclose(S * P, D)


def test_istft_reconstruction():
    from scipy.signal import bartlett, hann, hamming, blackman, blackmanharris

    def __test(x, n_fft, hop_length, window, atol, length):
        S = minispec.core.stft(
            x, n_fft=n_fft, hop_length=hop_length, window=window)
        x_reconstructed = minispec.core.istft(
            S, hop_length=hop_length, window=window, length=length)

        if length is not None:
            assert len(x_reconstructed) == length

        L = min(len(x), len(x_reconstructed))
        x = np.resize(x, L)
        x_reconstructed = np.resize(x_reconstructed, L)

        # NaN/Inf/-Inf should not happen
        assert np.all(np.isfinite(x_reconstructed))

        # should be almost approximately reconstucted
        assert np.allclose(x, x_reconstructed, atol=atol)

    srand()
    # White noise
    x1 = np.random.randn(2 ** 15)

    # Sin wave
    x2 = np.sin(np.linspace(-np.pi, np.pi, 2 ** 15))

    # Real music signal
    x3, sr = load_audio(os.path.join('tests', 'data', 'test1_44100.wav'),
                        sr=None, mono=True)
    assert sr == 44100

    for x, atol in [(x1, 1.0e-6), (x2, 1.0e-7), (x3, 1.0e-7)]:
        for window_func in [bartlett, hann, hamming, blackman, blackmanharris]:
            for n_fft in [512, 1024, 2048, 4096]:
                win = window_func(n_fft, sym=False)
                symwin = window_func(n_fft, sym=True)
                # tests with pre-computed window fucntions
                for hop_length_denom in six.moves.range(2, 9):
                    hop_length = n_fft // hop_length_denom
                    for length in [None, len(x) - 1000, len(x + 1000)]:
                        yield (__test, x, n_fft, hop_length, win, atol, length)
                        yield (__test, x, n_fft, hop_length, symwin, atol, length)
                # also tests with passing widnow function itself
                yield (__test, x, n_fft, n_fft // 9, window_func, atol, None)

        # test with default paramters
        x_reconstructed = minispec.core.istft(minispec.core.stft(x))
        L = min(len(x), len(x_reconstructed))
        x = np.resize(x, L)
        x_reconstructed = np.resize(x_reconstructed, L)

        assert np.allclose(x, x_reconstructed, atol=atol)


def test_load_options():

    filename = os.path.join('tests', 'data', 'test1_22050.wav')

    def __test(offset, duration, mono, dtype):

        y, sr = load_audio(filename, mono=mono, offset=offset,
                             duration=duration, dtype=dtype)

        if duration is not None:
            assert np.allclose(y.shape[-1], int(sr * duration))

        if mono:
            assert y.ndim == 1
        else:
            # This test file is stereo, so y.ndim should be 2
            assert y.ndim == 2

        # Check the dtype
        assert np.issubdtype(y.dtype, dtype)
        assert np.issubdtype(dtype, y.dtype)

    for offset in [0, 1, 2]:
        for duration in [None, 0, 0.5, 1, 2]:
            for mono in [False, True]:
                for dtype in [np.float32, np.float64]:
                    yield __test, offset, duration, mono, dtype
    pass


def test__spectrogram():

    y, sr = load_audio(os.path.join('tests', 'data', 'test1_22050.wav'))

    def __test(n_fft, hop_length, power):

        S = np.abs(minispec.stft(y, n_fft=n_fft, hop_length=hop_length))**power

        S_, n_fft_ = minispec.core.spectrum._spectrogram(y=y, S=S, n_fft=n_fft,
                                                        hop_length=hop_length,
                                                        power=power)

        # First check with all parameters
        assert np.allclose(S, S_)
        assert np.allclose(n_fft, n_fft_)

        # Then check with only the audio
        S_, n_fft_ = minispec.core.spectrum._spectrogram(y=y, n_fft=n_fft,
                                                        hop_length=hop_length,
                                                        power=power)
        assert np.allclose(S, S_)
        assert np.allclose(n_fft, n_fft_)

        # And only the spectrogram
        S_, n_fft_ = minispec.core.spectrum._spectrogram(S=S, n_fft=n_fft,
                                                        hop_length=hop_length,
                                                        power=power)
        assert np.allclose(S, S_)
        assert np.allclose(n_fft, n_fft_)

        # And only the spectrogram with no shape parameters
        S_, n_fft_ = minispec.core.spectrum._spectrogram(S=S, power=power)
        assert np.allclose(S, S_)
        assert np.allclose(n_fft, n_fft_)

        # And only the spectrogram but with incorrect n_fft
        S_, n_fft_ = minispec.core.spectrum._spectrogram(S=S, n_fft=2*n_fft,
                                                        power=power)
        assert np.allclose(S, S_)
        assert np.allclose(n_fft, n_fft_)

    for n_fft in [1024, 2048]:
        for hop_length in [None, 512]:
            for power in [1, 2]:
                yield __test, n_fft, hop_length, power
    assert minispec.core.spectrum._spectrogram(y)


def test_power_to_db():

    # Fake up some data
    def __test(x, ref, amin, top_db):

        y = minispec.power_to_db(x,
                                ref=ref,
                                amin=amin,
                                top_db=top_db)

        assert np.isrealobj(y)
        assert y.shape == x.shape

        if top_db is not None:
            assert y.min() >= y.max()-top_db

    for n in [1, 2, 10]:
        x = np.linspace(0, 2e5, num=n)
        phase = np.exp(1.j * x)

        for ref in [1.0, np.max]:
            for amin in [-1, 0, 1e-10, 1e3]:
                for top_db in [None, -10, 0, 40, 80]:
                    tf = __test
                    if amin <= 0 or (top_db is not None and top_db < 0):
                        tf = pytest.mark.xfail(__test, raises=minispec.ParameterError)
                    yield tf, x, ref, amin, top_db
                    yield tf, x * phase, ref, amin, top_db


def test_power_to_db_inv():

    def __test(y_true, x, rp):
        y = minispec.power_to_db(x, ref=rp, top_db=None)

        assert np.isclose(y, y_true)

    for erp in range(-5, 6):
        for k in range(-5, 6):
            yield __test, (k-erp)*10, 10.0**k, 10.0**erp


def test_amplitude_to_db():

    srand()

    NOISE_FLOOR = 1e-6

    # Make some noise
    x = np.abs(np.random.randn(1000)) + NOISE_FLOOR

    db1 = minispec.amplitude_to_db(x, top_db=None)
    db2 = minispec.power_to_db(x**2, top_db=None)

    assert np.allclose(db1, db2)


def test_db_to_power_inv():

    srand()

    NOISE_FLOOR = 1e-5

    # Make some noise
    xp = (np.abs(np.random.randn(1000)) + NOISE_FLOOR)**2

    def __test(ref):

        db = minispec.power_to_db(xp, ref=ref, top_db=None)
        xp2 = minispec.db_to_power(db, ref=ref)

        assert np.allclose(xp, xp2)

    for ref_p in range(-3, 4):
        yield __test, 10.0**ref_p


def test_db_to_power():

    def __test(y, rp, x_true):

        x = minispec.db_to_power(y, ref=rp)

        assert np.isclose(x, x_true), (x, x_true, y, rp)

    for erp in range(-5, 6):
        for db in range(-100, 101, 10):
            yield __test, db, 10.0**erp, 10.0**erp * 10.0**(0.1 * db)


def test_db_to_amplitude_inv():

    srand()

    NOISE_FLOOR = 1e-5

    # Make some noise
    xp = np.abs(np.random.randn(1000)) + NOISE_FLOOR

    def __test(ref):

        db = minispec.amplitude_to_db(xp, ref=ref, top_db=None)
        xp2 = minispec.db_to_amplitude(db, ref=ref)

        assert np.allclose(xp, xp2)

    for ref_p in range(-3, 4):
        yield __test, 10.0**ref_p


def test_db_to_amplitude():

    srand()

    NOISE_FLOOR = 1e-6

    # Make some noise
    x = np.abs(np.random.randn(1000)) + NOISE_FLOOR

    db = minispec.amplitude_to_db(x, top_db=None)
    x2 = minispec.db_to_amplitude(db)

    assert np.allclose(x, x2)


def test_show_versions():
    # Nothing to test here, except that everything passes.
    minispec.show_versions()


def test_padding():

    # A simple test to verify that pad_mode is used properly by giving
    # different answers for different modes.
    # Does not validate the correctness of each mode.

    y, sr = load_audio(os.path.join('tests', 'data', 'test1_44100.wav'),
                         sr=None, mono=True, duration=1)

    def __test_stft(center, pad_mode):
        D1 = minispec.stft(y, center=center, pad_mode='reflect')
        D2 = minispec.stft(y, center=center, pad_mode=pad_mode)

        assert D1.shape == D2.shape

        if center and pad_mode != 'reflect':
            assert not np.allclose(D1, D2)
        else:
            assert np.allclose(D1, D2)

    for pad_mode in ['reflect', 'constant']:
        for center in [False, True]:
            yield __test_stft, center, pad_mode


def test_get_fftlib():
    import numpy.fft as fft
    assert minispec.get_fftlib() is fft


def test_set_fftlib():
    minispec.set_fftlib('foo')
    assert minispec.get_fftlib() == 'foo'
    minispec.set_fftlib()


def test_reset_fftlib():
    import numpy.fft as fft
    minispec.set_fftlib()
    assert minispec.get_fftlib() is fft
