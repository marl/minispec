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


<<<<<<< HEAD
=======
def test_get_duration_wav():

    def __test_audio(filename, mono, sr, duration):
        y, sr = librosa.load(filename, sr=sr, mono=mono, duration=duration)

        duration_est = librosa.get_duration(y=y, sr=sr)

        assert np.allclose(duration_est, duration, rtol=1e-3, atol=1e-5)

    def __test_spec(filename, sr, duration, n_fft, hop_length, center):
        y, sr = librosa.load(filename, sr=sr, duration=duration)

        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=center)

        duration_est = librosa.get_duration(S=S, sr=sr, n_fft=n_fft,
                                            hop_length=hop_length,
                                            center=center)

        # We lose a little accuracy in framing without centering, so it's
        # not as precise as time-domain duration
        assert np.allclose(duration_est, duration, rtol=1e-1, atol=1e-2)

    test_file = os.path.join('tests', 'data', 'test1_22050.wav')

    for sr in [8000, 11025, 22050]:
        for duration in [1.0, 2.5]:
            for mono in [False, True]:
                yield __test_audio, test_file, mono, sr, duration

            for n_fft in [256, 512, 1024]:
                for hop_length in [n_fft // 8, n_fft // 4, n_fft // 2]:
                    for center in [False, True]:
                        yield (__test_spec, test_file, sr,
                               duration, n_fft, hop_length, center)


def test_get_duration_filename():

    filename = os.path.join('tests', 'data', 'test2_8000.wav')
    true_duration = 30.197625

    duration_fn = librosa.get_duration(filename=filename)
    y, sr = librosa.load(filename, sr=None)
    duration_y = librosa.get_duration(y=y, sr=sr)

    assert np.allclose(duration_fn, true_duration)
    assert np.allclose(duration_fn, duration_y)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_get_duration_fail():
    librosa.get_duration(y=None, S=None, filename=None)


def test_autocorrelate():

    def __test(y, truth, max_size, axis):

        ac = librosa.autocorrelate(y, max_size=max_size, axis=axis)

        my_slice = [slice(None)] * truth.ndim
        if max_size is not None and max_size <= y.shape[axis]:
            my_slice[axis] = slice(min(max_size, y.shape[axis]))

        if not np.iscomplexobj(y):
            assert not np.iscomplexobj(ac)

        assert np.allclose(ac, truth[tuple(my_slice)])

    srand()
    # test with both real and complex signals
    for y in [np.random.randn(256, 256), np.exp(1.j * np.random.randn(256, 256))]:

        # Make ground-truth autocorrelations along each axis
        truth = [np.asarray([scipy.signal.fftconvolve(yi, yi[::-1].conj(),
                                                      mode='full')[len(yi)-1:] for yi in y.T]).T,
                 np.asarray([scipy.signal.fftconvolve(yi, yi[::-1].conj(),
                                                      mode='full')[len(yi)-1:] for yi in y])]

        for axis in [0, 1, -1]:
            for max_size in [None, y.shape[axis]//2, y.shape[axis], 2 * y.shape[axis]]:
                yield __test, y, truth[axis], max_size, axis


def test_to_mono():

    def __test(filename, mono):
        y, sr = librosa.load(filename, mono=mono)

        y_mono = librosa.to_mono(y)

        assert y_mono.ndim == 1
        assert len(y_mono) == y.shape[-1]

        if mono:
            assert np.allclose(y, y_mono)

    filename = os.path.join('tests', 'data', 'test1_22050.wav')

    for mono in [False, True]:
        yield __test, filename, mono


def test_zero_crossings():

    def __test(data, threshold, ref_magnitude, pad, zp):

        zc = librosa.zero_crossings(y=data,
                                    threshold=threshold,
                                    ref_magnitude=ref_magnitude,
                                    pad=pad,
                                    zero_pos=zp)

        idx = np.flatnonzero(zc)

        if pad:
            idx = idx[1:]

        for i in idx:
            assert np.sign(data[i]) != np.sign(data[i-1])

    srand()
    data = np.random.randn(32)

    for threshold in [None, 0, 1e-10]:
        for ref_magnitude in [None, 0.1, np.max]:
            for pad in [False, True]:
                for zero_pos in [False, True]:

                    yield __test, data, threshold, ref_magnitude, pad, zero_pos


def test_pitch_tuning():

    def __test(hz, resolution, bins_per_octave, tuning):

        est_tuning = librosa.pitch_tuning(hz,
                                          resolution=resolution,
                                          bins_per_octave=bins_per_octave)

        assert np.abs(tuning - est_tuning) <= resolution

    for resolution in [1e-2, 1e-3]:
        for bins_per_octave in [12]:
            # Make up some frequencies
            for tuning in [-0.5, -0.375, -0.25, 0.0, 0.25, 0.375]:

                note_hz = librosa.midi_to_hz(tuning + np.arange(128))

                yield __test, note_hz, resolution, bins_per_octave, tuning


def test_piptrack_properties():

    def __test(S, n_fft, hop_length, fmin, fmax, threshold):

        pitches, mags = librosa.core.piptrack(S=S,
                                              n_fft=n_fft,
                                              hop_length=hop_length,
                                              fmin=fmin,
                                              fmax=fmax,
                                              threshold=threshold)

        # Shape tests
        assert S.shape == pitches.shape
        assert S.shape == mags.shape

        # Make sure all magnitudes are positive
        assert np.all(mags >= 0)

        # Check the frequency estimates for bins with non-zero magnitude
        idx = (mags > 0)
        assert np.all(pitches[idx] >= fmin)
        assert np.all(pitches[idx] <= fmax)

        # And everywhere else, pitch should be 0
        assert np.all(pitches[~idx] == 0)

    y, sr = librosa.load(os.path.join('tests', 'data', 'test1_22050.wav'))

    for n_fft in [2048, 4096]:
        for hop_length in [None, n_fft // 4, n_fft // 2]:
            S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
            for fmin in [0, 100]:
                for fmax in [4000, 8000, sr // 2]:
                    for threshold in [0.1, 0.2, 0.5]:
                        yield __test, S, n_fft, hop_length, fmin, fmax, threshold


def test_piptrack_errors():

    def __test(y, sr, S, n_fft, hop_length, fmin, fmax, threshold):
        pitches, mags = librosa.piptrack(
            y=y, sr=sr, S=S, n_fft=n_fft, hop_length=hop_length, fmin=fmin,
            fmax=fmax, threshold=threshold)

    S = np.asarray([[1, 0, 0]]).T
    np.seterr(divide='raise')
    yield __test, None, 22050, S, 4096, None, 150.0, 4000.0, 0.1


def test_piptrack():

    def __test(S, freq):
        pitches, mags = librosa.piptrack(S=S, fmin=100)

        idx = (mags > 0)

        assert len(idx) > 0

        recovered_pitches = pitches[idx]

        # We should be within one cent of the target
        assert np.all(np.abs(np.log2(recovered_pitches) - np.log2(freq)) <= 1e-2)

    sr = 22050
    duration = 3.0

    for freq in [110, 220, 440, 880]:
        # Generate a sine tone
        y = np.sin(2 * np.pi * freq * np.linspace(0, duration, num=int(duration*sr)))
        for n_fft in [1024, 2048, 4096]:
            # Using left-aligned frames eliminates reflection artifacts at the boundaries
            S = np.abs(librosa.stft(y, n_fft=n_fft, center=False))

            yield __test, S, freq


def test_estimate_tuning():

    def __test(target_hz, resolution, bins_per_octave, tuning):

        y = np.sin(2 * np.pi * target_hz * t)
        tuning_est = librosa.estimate_tuning(resolution=resolution,
                                             bins_per_octave=bins_per_octave,
                                             y=y,
                                             sr=sr,
                                             n_fft=2048,
                                             fmin=librosa.note_to_hz('C4'),
                                             fmax=librosa.note_to_hz('G#9'))

        # Round to the proper number of decimals
        deviation = np.around(tuning - tuning_est, int(-np.log10(resolution)))

        # Take the minimum floating point for positive and negative deviations
        max_dev = np.min([np.mod(deviation, 1.0), np.mod(-deviation, 1.0)])

        # We'll accept an answer within three bins of the resolution
        assert max_dev <= 3 * resolution

    for sr in [11025, 22050]:
        duration = 5.0

        t = np.linspace(0, duration, int(duration * sr))

        for resolution in [1e-2]:
            for bins_per_octave in [12]:
                # test a null-signal tuning estimate
                yield (__test, 0.0, resolution, bins_per_octave, 0.0)

                for center_note in [69, 84, 108]:
                    for tuning in np.linspace(-0.5, 0.5, 8, endpoint=False):
                        target_hz = librosa.midi_to_hz(center_note + tuning)

                        yield (__test, np.asscalar(target_hz), resolution,
                               bins_per_octave, tuning)


>>>>>>> acd4e6fbd5b3b361e8c2e89797cfe3eb0b1a3aba
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
<<<<<<< HEAD

=======
            yield __test_ifgram, center, pad_mode


def test_iirt():
    gt = scipy.io.loadmat(os.path.join('tests', 'data', 'features-CT-cqt'), squeeze_me=True)['f_cqt']

    y, sr = librosa.load(os.path.join('tests', 'data', 'test1_44100.wav'))
    mut1 = librosa.iirt(y, hop_length=2205, win_length=4410, flayout='ba')

    assert np.allclose(mut1, gt[23:108, :mut1.shape[1]], atol=1.8)

    mut2 = librosa.iirt(y, hop_length=2205, win_length=4410, flayout='sos')

    assert np.allclose(mut2, gt[23:108, :mut2.shape[1]], atol=1.8)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_iirt_flayout1():
    y, sr = librosa.load(os.path.join('tests', 'data', 'test1_44100.wav'))
    librosa.iirt(y, hop_length=2205, win_length=4410, flayout='foo')


def test_iirt_flayout2():
    y, sr = librosa.load(os.path.join('tests', 'data', 'test1_44100.wav'))
    with pytest.warns(FutureWarning):
        librosa.iirt(y, hop_length=2205, win_length=4410)


def test_pcen():

    def __test(gain, bias, power, b, time_constant, eps, ms, S, Pexp):

        with warnings.catch_warnings(record=True) as out:

            P = librosa.pcen(S, gain=gain, bias=bias, power=power,
                             time_constant=time_constant, eps=eps, b=b,
                             max_size=ms)

            if np.issubdtype(S.dtype, np.complexfloating):
                assert len(out) > 0
                assert 'complex' in str(out[0].message).lower()

        assert P.shape == S.shape
        assert np.all(P >= 0)
        assert np.all(np.isfinite(P))

        if Pexp is not None:
            assert np.allclose(P, Pexp)

    tf = pytest.mark.xfail(__test, raises=librosa.ParameterError)

    srand()
    S = np.abs(np.random.randn(9, 30))

    # Bounds tests (failures):
    #   gain < 0
    yield tf, -1, 1, 1, 0.5, 0.5, 1e-6, 1, S, S

    #   bias < 0
    yield tf, 1, -1, 1, 0.5, 0.5, 1e-6, 1, S, S

    #   power <= 0
    yield tf, 1, 1, 0, 0.5, 0.5, 1e-6, 1, S, S

    #   b < 0
    yield tf, 1, 1, 1, -2, 0.5, 1e-6, 1, S, S

    #   b > 1
    yield tf, 1, 1, 1, 2, 0.5, 1e-6, 1, S, S

    #   time_constant <= 0
    yield tf, 1, 1, 1, 0.5, -2, 1e-6, 1, S, S

    #   eps <= 0
    yield tf, 1, 1, 1, 0.5, 0.5, 0, 1, S, S

    #   max_size not int, < 1
    yield tf, 1, 1, 1, 0.5, 0.5, 1e-6, 1.5, S, S
    yield tf, 1, 1, 1, 0.5, 0.5, 1e-6, 0, S, S

    # Edge cases:
    #   gain=0, bias=0, power=p, b=1 => S**p
    for p in [0.5, 1, 2]:
        yield __test, 0, 0, p, 1.0, 0.5, 1e-6, 1, S, S**p

    #   gain=1, bias=0, power=1, b=1, eps=1e-20 => ones
    yield __test, 1, 0, 1, 1.0, 0.5, 1e-20, 1, S, np.ones_like(S)

    # Catch the complex warning
    yield __test, 1, 0, 1, 1.0, 0.5, 1e-20, 1, S * 1.j, np.ones_like(S)

    #   zeros to zeros
    Z = np.zeros_like(S)
    yield __test, 0.98, 2.0, 0.5, None, 0.395, 1e-6, 1, Z, Z
    yield __test, 0.98, 2.0, 0.5, None, 0.395, 1e-6, 3, Z, Z


def test_pcen_axes():

    srand()
    # Make a power spectrogram
    X = np.random.randn(3, 100, 50)**2

    # First, test that axis setting works
    P1 = librosa.pcen(X[0])
    P1a = librosa.pcen(X[0], axis=-1)
    P2 = librosa.pcen(X[0].T, axis=0).T

    assert np.allclose(P1, P2)
    assert np.allclose(P1, P1a)

    # Test that it works with max-filtering
    P1 = librosa.pcen(X[0], max_size=3)
    P1a = librosa.pcen(X[0], axis=-1, max_size=3)
    P2 = librosa.pcen(X[0].T, axis=0, max_size=3).T

    assert np.allclose(P1, P2)
    assert np.allclose(P1, P1a)

    # Test that it works with multi-dimensional input, no filtering
    P0 = librosa.pcen(X[0])
    P1 = librosa.pcen(X[1])
    P2 = librosa.pcen(X[2])
    Pa = librosa.pcen(X)

    assert np.allclose(P0, Pa[0])
    assert np.allclose(P1, Pa[1])
    assert np.allclose(P2, Pa[2])

    # Test that it works with multi-dimensional input, max-filtering
    P0 = librosa.pcen(X[0], max_size=3)
    P1 = librosa.pcen(X[1], max_size=3)
    P2 = librosa.pcen(X[2], max_size=3)
    Pa = librosa.pcen(X, max_size=3, max_axis=1)

    assert np.allclose(P0, Pa[0])
    assert np.allclose(P1, Pa[1])
    assert np.allclose(P2, Pa[2])

@pytest.mark.xfail(raises=librosa.ParameterError)
def test_pcen_axes_nomax():
    srand()
    # Make a power spectrogram
    X = np.random.randn(3, 100, 50)**2

    librosa.pcen(X, max_size=3)

@pytest.mark.xfail(raises=librosa.ParameterError)
def test_pcen_max1():

    librosa.pcen(np.arange(100), max_size=3)


def test_pcen_ref():

    srand()
    # Make a power spectrogram
    X = np.random.randn(100, 50)**2

    # Edge cases:
    #   gain=1, bias=0, power=1, b=1 => ones
    ones = np.ones_like(X)

    Y = librosa.pcen(X, gain=1, bias=0, power=1, b=1, eps=1e-20)
    assert np.allclose(Y, ones)

    # with ref=ones, we should get X / (eps + ones) == X
    Y2 = librosa.pcen(X, gain=1, bias=0, power=1, b=1, ref=ones, eps=1e-20)
    assert np.allclose(Y2, X)


def test_get_fftlib():
    import numpy.fft as fft
    assert librosa.get_fftlib() is fft


def test_set_fftlib():
    librosa.set_fftlib('foo')
    assert librosa.get_fftlib() == 'foo'
    librosa.set_fftlib()


def test_reset_fftlib():
    import numpy.fft as fft
    librosa.set_fftlib()
    assert librosa.get_fftlib() is fft
>>>>>>> acd4e6fbd5b3b361e8c2e89797cfe3eb0b1a3aba
