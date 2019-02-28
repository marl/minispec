#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filters
=======

Filter bank construction
------------------------
.. autosummary::
    :toctree: generated/

    mel

Window functions
----------------
.. autosummary::
    :toctree: generated/

    window_bandwidth
    get_window


Miscellaneous
-------------
.. autosummary::
    :toctree: generated/

    constant_q_lengths
    cq_to_chroma
    mr_frequencies
    window_sumsquare

"""
import warnings

import numpy as np
import scipy
import scipy.signal
import six

from ._cache import cache
from . import util
from .util.exceptions import ParameterError

from .core.time_frequency import fft_frequencies, mel_frequencies

__all__ = ['mel',
           'get_window',
           'window_sumsquare']


def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False,
        norm=1, dtype=np.float32):
    """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins

    Parameters
    ----------
    sr        : number > 0 [scalar]
        sampling rate of the incoming signal

    n_fft     : int > 0 [scalar]
        number of FFT components

    n_mels    : int > 0 [scalar]
        number of Mel bands to generate

    fmin      : float >= 0 [scalar]
        lowest frequency (in Hz)

    fmax      : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use `fmax = sr / 2.0`

    htk       : bool [scalar]
        use HTK formula instead of Slaney

    norm : {None, 1, np.inf} [scalar]
        if 1, divide the triangular mel weights by the width of the mel band
        (area normalization).  Otherwise, leave all the triangles aiming for
        a peak value of 1.0

    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.

    Returns
    -------
    M         : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
        Mel transform matrix


    Examples
    --------
    >>> melfb = minispec.filters.mel(22050, 2048)
    >>> melfb
    array([[ 0.   ,  0.016, ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           ...,
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ]])


    Clip the maximum frequency to 8KHz

    >>> minispec.filters.mel(22050, 2048, fmax=8000)
    array([[ 0.  ,  0.02, ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           ...,
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ]])

    """

    if fmax is None:
        fmax = float(sr) / 2

    if norm is not None and norm != 1 and norm != np.inf:
        raise ParameterError('Unsupported norm: {}'.format(repr(norm)))

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i+2] / fdiff[i+1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == 1:
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2:n_mels+2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn('Empty filters detected in mel frequency basis. '
                      'Some channels will produce empty responses. '
                      'Try increasing your sampling rate (and fmax) or '
                      'reducing n_mels.')

    return weights


def __float_window(window_spec):
    '''Decorator function for windows with fractional input.

    This function guarantees that for fractional `x`, the following hold:

    1. `__float_window(window_function)(x)` has length `np.ceil(x)`
    2. all values from `np.floor(x)` are set to 0.

    For integer-valued `x`, there should be no change in behavior.
    '''

    def _wrap(n, *args, **kwargs):
        '''The wrapped window'''
        n_min, n_max = int(np.floor(n)), int(np.ceil(n))

        window = get_window(window_spec, n_min)

        if len(window) < n_max:
            window = np.pad(window, [(0, n_max - len(window))],
                            mode='constant')

        window[n_min:] = 0.0

        return window

    return _wrap


<<<<<<< HEAD
=======
@cache(level=10)
def constant_q(sr, fmin=None, n_bins=84, bins_per_octave=12, tuning=0.0,
               window='hann', filter_scale=1, pad_fft=True, norm=1,
               dtype=np.complex64, **kwargs):
    r'''Construct a constant-Q basis.

    This uses the filter bank described by [1]_.

    .. [1] McVicar, Matthew.
            "A machine learning approach to automatic chord extraction."
            Dissertation, University of Bristol. 2013.


    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate

    fmin : float > 0 [scalar]
        Minimum frequency bin. Defaults to `C1 ~= 32.70`

    n_bins : int > 0 [scalar]
        Number of frequencies.  Defaults to 7 octaves (84 bins).

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    tuning : float in `[-0.5, +0.5)` [scalar]
        Tuning deviation from A440 in fractions of a bin

    window : string, tuple, number, or function
        Windowing function to apply to filters.

    filter_scale : float > 0 [scalar]
        Scale of filter windows.
        Small values (<1) use shorter windows for higher temporal resolution.

    pad_fft : boolean
        Center-pad all filters up to the nearest integral power of 2.

        By default, padding is done with zeros, but this can be overridden
        by setting the `mode=` field in *kwargs*.

    norm : {inf, -inf, 0, float > 0}
        Type of norm to use for basis function normalization.
        See librosa.util.normalize

    dtype : np.dtype
        The data type of the output basis.
        By default, uses 64-bit (single precision) complex floating point.

    kwargs : additional keyword arguments
        Arguments to `np.pad()` when `pad==True`.

    Returns
    -------
    filters : np.ndarray, `len(filters) == n_bins`
        `filters[i]` is `i`\ th time-domain CQT basis filter

    lengths : np.ndarray, `len(lengths) == n_bins`
        The (fractional) length of each filter

    Notes
    -----
    This function caches at level 10.

    See Also
    --------
    constant_q_lengths
    librosa.core.cqt
    librosa.util.normalize


    Examples
    --------
    Use a shorter window for each filter

    >>> basis, lengths = librosa.filters.constant_q(22050, filter_scale=0.5)

    Plot one octave of filters in time and frequency

    >>> import matplotlib.pyplot as plt
    >>> basis, lengths = librosa.filters.constant_q(22050)
    >>> plt.figure(figsize=(10, 6))
    >>> plt.subplot(2, 1, 1)
    >>> notes = librosa.midi_to_note(np.arange(24, 24 + len(basis)))
    >>> for i, (f, n) in enumerate(zip(basis, notes[:12])):
    ...     f_scale = librosa.util.normalize(f) / 2
    ...     plt.plot(i + f_scale.real)
    ...     plt.plot(i + f_scale.imag, linestyle=':')
    >>> plt.axis('tight')
    >>> plt.yticks(np.arange(len(notes[:12])), notes[:12])
    >>> plt.ylabel('CQ filters')
    >>> plt.title('CQ filters (one octave, time domain)')
    >>> plt.xlabel('Time (samples at 22050 Hz)')
    >>> plt.legend(['Real', 'Imaginary'], frameon=True, framealpha=0.8)
    >>> plt.subplot(2, 1, 2)
    >>> F = np.abs(np.fft.fftn(basis, axes=[-1]))
    >>> # Keep only the positive frequencies
    >>> F = F[:, :(1 + F.shape[1] // 2)]
    >>> librosa.display.specshow(F, x_axis='linear')
    >>> plt.yticks(np.arange(len(notes))[::12], notes[::12])
    >>> plt.ylabel('CQ filters')
    >>> plt.title('CQ filter magnitudes (frequency domain)')
    >>> plt.tight_layout()
    '''

    if fmin is None:
        fmin = note_to_hz('C1')

    # Pass-through parameters to get the filter lengths
    lengths = constant_q_lengths(sr, fmin,
                                 n_bins=n_bins,
                                 bins_per_octave=bins_per_octave,
                                 tuning=tuning,
                                 window=window,
                                 filter_scale=filter_scale)

    # Apply tuning correction
    correction = 2.0**(float(tuning) / bins_per_octave)
    fmin = correction * fmin

    # Q should be capitalized here, so we suppress the name warning
    # pylint: disable=invalid-name
    Q = float(filter_scale) / (2.0**(1. / bins_per_octave) - 1)

    # Convert lengths back to frequencies
    freqs = Q * sr / lengths

    # Build the filters
    filters = []
    for ilen, freq in zip(lengths, freqs):
        # Build the filter: note, length will be ceil(ilen)
        sig = np.exp(np.arange(-ilen//2, ilen//2, dtype=float) * 1j * 2 * np.pi * freq / sr)

        # Apply the windowing function
        sig = sig * __float_window(window)(len(sig))

        # Normalize
        sig = util.normalize(sig, norm=norm)

        filters.append(sig)

    # Pad and stack
    max_len = max(lengths)
    if pad_fft:
        max_len = int(2.0**(np.ceil(np.log2(max_len))))
    else:
        max_len = int(np.ceil(max_len))

    filters = np.asarray([util.pad_center(filt, max_len, **kwargs)
                          for filt in filters], dtype=dtype)

    return filters, np.asarray(lengths)


@cache(level=10)
def constant_q_lengths(sr, fmin, n_bins=84, bins_per_octave=12,
                       tuning=0.0, window='hann', filter_scale=1):
    r'''Return length of each filter in a constant-Q basis.

    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate

    fmin : float > 0 [scalar]
        Minimum frequency bin.

    n_bins : int > 0 [scalar]
        Number of frequencies.  Defaults to 7 octaves (84 bins).

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    tuning : float in `[-0.5, +0.5)` [scalar]
        Tuning deviation from A440 in fractions of a bin

    window : str or callable
        Window function to use on filters

    filter_scale : float > 0 [scalar]
        Resolution of filter windows. Larger values use longer windows.

    Returns
    -------
    lengths : np.ndarray
        The length of each filter.

    Notes
    -----
    This function caches at level 10.

    See Also
    --------
    constant_q
    librosa.core.cqt
    '''

    if fmin <= 0:
        raise ParameterError('fmin must be positive')

    if bins_per_octave <= 0:
        raise ParameterError('bins_per_octave must be positive')

    if filter_scale <= 0:
        raise ParameterError('filter_scale must be positive')

    if n_bins <= 0 or not isinstance(n_bins, int):
        raise ParameterError('n_bins must be a positive integer')

    correction = 2.0**(float(tuning) / bins_per_octave)

    fmin = correction * fmin

    # Q should be capitalized here, so we suppress the name warning
    # pylint: disable=invalid-name
    Q = float(filter_scale) / (2.0**(1. / bins_per_octave) - 1)

    # Compute the frequencies
    freq = fmin * (2.0 ** (np.arange(n_bins, dtype=float) / bins_per_octave))

    if freq[-1] * (1 + 0.5 * window_bandwidth(window) / Q) > sr / 2.0:
        raise ParameterError('Filter pass-band lies beyond Nyquist')

    # Convert frequencies to filter lengths
    lengths = Q * sr / freq

    return lengths


@cache(level=10)
def cq_to_chroma(n_input, bins_per_octave=12, n_chroma=12,
                 fmin=None, window=None, base_c=True, dtype=np.float32):
    '''Convert a Constant-Q basis to Chroma.


    Parameters
    ----------
    n_input : int > 0 [scalar]
        Number of input components (CQT bins)

    bins_per_octave : int > 0 [scalar]
        How many bins per octave in the CQT

    n_chroma : int > 0 [scalar]
        Number of output bins (per octave) in the chroma

    fmin : None or float > 0
        Center frequency of the first constant-Q channel.
        Default: 'C1' ~= 32.7 Hz

    window : None or np.ndarray
        If provided, the cq_to_chroma filter bank will be
        convolved with `window`.

    base_c : bool
        If True, the first chroma bin will start at 'C'
        If False, the first chroma bin will start at 'A'

    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.


    Returns
    -------
    cq_to_chroma : np.ndarray [shape=(n_chroma, n_input)]
        Transformation matrix: `Chroma = np.dot(cq_to_chroma, CQT)`

    Raises
    ------
    ParameterError
        If `n_input` is not an integer multiple of `n_chroma`

    Notes
    -----
    This function caches at level 10.

    Examples
    --------
    Get a CQT, and wrap bins to chroma

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> CQT = np.abs(librosa.cqt(y, sr=sr))
    >>> chroma_map = librosa.filters.cq_to_chroma(CQT.shape[0])
    >>> chromagram = chroma_map.dot(CQT)
    >>> # Max-normalize each time step
    >>> chromagram = librosa.util.normalize(chromagram, axis=0)

    >>> import matplotlib.pyplot as plt
    >>> plt.subplot(3, 1, 1)
    >>> librosa.display.specshow(librosa.amplitude_to_db(CQT,
    ...                                                  ref=np.max),
    ...                          y_axis='cqt_note')
    >>> plt.title('CQT Power')
    >>> plt.colorbar()
    >>> plt.subplot(3, 1, 2)
    >>> librosa.display.specshow(chromagram, y_axis='chroma')
    >>> plt.title('Chroma (wrapped CQT)')
    >>> plt.colorbar()
    >>> plt.subplot(3, 1, 3)
    >>> chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    >>> librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    >>> plt.title('librosa.feature.chroma_stft')
    >>> plt.colorbar()
    >>> plt.tight_layout()

    '''

    # How many fractional bins are we merging?
    n_merge = float(bins_per_octave) / n_chroma

    if fmin is None:
        fmin = note_to_hz('C1')

    if np.mod(n_merge, 1) != 0:
        raise ParameterError('Incompatible CQ merge: '
                             'input bins must be an '
                             'integer multiple of output bins.')

    # Tile the identity to merge fractional bins
    cq_to_ch = np.repeat(np.eye(n_chroma), n_merge, axis=1)

    # Roll it left to center on the target bin
    cq_to_ch = np.roll(cq_to_ch, - int(n_merge // 2), axis=1)

    # How many octaves are we repeating?
    n_octaves = np.ceil(np.float(n_input) / bins_per_octave)

    # Repeat and trim
    cq_to_ch = np.tile(cq_to_ch, int(n_octaves))[:, :n_input]

    # What's the note number of the first bin in the CQT?
    # midi uses 12 bins per octave here
    midi_0 = np.mod(hz_to_midi(fmin), 12)

    if base_c:
        # rotate to C
        roll = midi_0
    else:
        # rotate to A
        roll = midi_0 - 9

    # Adjust the roll in terms of how many chroma we want out
    # We need to be careful with rounding here
    roll = int(np.round(roll * (n_chroma / 12.)))

    # Apply the roll
    cq_to_ch = np.roll(cq_to_ch, roll, axis=0).astype(dtype)

    if window is not None:
        cq_to_ch = scipy.signal.convolve(cq_to_ch,
                                         np.atleast_2d(window),
                                         mode='same')

    return cq_to_ch


@cache(level=10)
def window_bandwidth(window, n=1000):
    '''Get the equivalent noise bandwidth of a window function.


    Parameters
    ----------
    window : callable or string
        A window function, or the name of a window function.
        Examples:
        - scipy.signal.hann
        - 'boxcar'

    n : int > 0
        The number of coefficients to use in estimating the
        window bandwidth

    Returns
    -------
    bandwidth : float
        The equivalent noise bandwidth (in FFT bins) of the
        given window function

    Notes
    -----
    This function caches at level 10.

    See Also
    --------
    get_window
    '''

    if hasattr(window, '__name__'):
        key = window.__name__
    else:
        key = window

    if key not in WINDOW_BANDWIDTHS:
        win = get_window(window, n)
        WINDOW_BANDWIDTHS[key] = n * np.sum(win**2) / np.sum(np.abs(win))**2

    return WINDOW_BANDWIDTHS[key]


@cache(level=10)
def get_window(window, Nx, fftbins=True):
    '''Compute a window function.

    This is a wrapper for `scipy.signal.get_window` that additionally
    supports callable or pre-computed windows.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        The window specification:

        - If string, it's the name of the window function (e.g., `'hann'`)
        - If tuple, it's the name of the window function and any parameters
          (e.g., `('kaiser', 4.0)`)
        - If numeric, it is treated as the beta parameter of the `'kaiser'`
          window, as in `scipy.signal.get_window`.
        - If callable, it's a function that accepts one integer argument
          (the window length)
        - If list-like, it's a pre-computed window of the correct length `Nx`

    Nx : int > 0
        The length of the window

    fftbins : bool, optional
        If True (default), create a periodic window for use with FFT
        If False, create a symmetric window for filter design applications.

    Returns
    -------
    get_window : np.ndarray
        A window of length `Nx` and type `window`

    See Also
    --------
    scipy.signal.get_window

    Raises
    ------
    ParameterError
        If `window` is supplied as a vector of length != `n_fft`,
        or is otherwise mis-specified.
    '''
    if six.callable(window):
        return window(Nx)

    elif (isinstance(window, (six.string_types, tuple)) or
          np.isscalar(window)):
        # TODO: if we add custom window functions in minispec, call them here

        return scipy.signal.get_window(window, Nx, fftbins=fftbins)

    elif isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)

        raise ParameterError('Window size mismatch: '
                             '{:d} != {:d}'.format(len(window), Nx))
    else:
        raise ParameterError('Invalid window specification: {}'.format(window))


def __window_ss_fill(x, win_sq, n_frames, hop_length):  # pragma: no cover
    '''Helper function for window sum-square calculation.'''

    n = len(x)
    n_fft = len(win_sq)
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]


def window_sumsquare(window, n_frames, hop_length=512, win_length=None, n_fft=2048,
                     dtype=np.float32, norm=None):
    '''
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing observations
    in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function

    Examples
    --------
    For a fixed frame length (2048), compare modulation effects for a Hann window
    at different hop lengths:

    >>> n_frames = 50
    >>> wss_256 = minispec.filters.window_sumsquare('hann', n_frames, hop_length=256)
    >>> wss_512 = minispec.filters.window_sumsquare('hann', n_frames, hop_length=512)
    >>> wss_1024 = minispec.filters.window_sumsquare('hann', n_frames, hop_length=1024)

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(3,1,1)
    >>> plt.plot(wss_256)
    >>> plt.title('hop_length=256')
    >>> plt.subplot(3,1,2)
    >>> plt.plot(wss_512)
    >>> plt.title('hop_length=512')
    >>> plt.subplot(3,1,3)
    >>> plt.plot(wss_1024)
    >>> plt.title('hop_length=1024')
    >>> plt.tight_layout()

    '''
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length)
    win_sq = util.normalize(win_sq, norm=norm)**2
    win_sq = util.pad_center(win_sq, n_fft)

    # Fill the envelope
    __window_ss_fill(x, win_sq, n_frames, hop_length)

    return x
