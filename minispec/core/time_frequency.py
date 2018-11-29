#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Time and frequency utilities'''

import numpy as np

__all__ = ['frames_to_samples', 'frames_to_time',
           'samples_to_frames', 'samples_to_time',
           'time_to_samples', 'time_to_frames',
           'hz_to_mel', 'mel_to_hz',
           'fft_frequencies',
           'mel_frequencies']


def frames_to_samples(frames, hop_length=512, n_fft=None):
    """Converts frame indices to audio sample indices.

    Parameters
    ----------
    frames     : number or np.ndarray [shape=(n,)]
        frame index or vector of frame indices

    hop_length : int > 0 [scalar]
        number of samples between successive frames

    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of `n_fft / 2`
        to counteract windowing effects when using a non-centered STFT.

    Returns
    -------
    times : number or np.ndarray
        time (in samples) of each given frame number:
        `times[i] = frames[i] * hop_length`

    See Also
    --------
    frames_to_time : convert frame indices to time values
    samples_to_frames : convert sample indices to frame indices

    Examples
    --------
    >>> y, sr = minispec.load(minispec.util.example_audio_file())
    >>> tempo, beats = minispec.beat.beat_track(y, sr=sr)
    >>> beat_samples = minispec.frames_to_samples(beats)
    """

    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)

    return (np.asanyarray(frames) * hop_length + offset).astype(int)


def samples_to_frames(samples, hop_length=512, n_fft=None):
    """Converts sample indices into STFT frames.

    Examples
    --------
    >>> # Get the frame numbers for every 256 samples
    >>> minispec.samples_to_frames(np.arange(0, 22050, 256))
    array([ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,
            7,  7,  8,  8,  9,  9, 10, 10, 11, 11, 12, 12, 13, 13,
           14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20,
           21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27,
           28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34,
           35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41,
           42, 42, 43])

    Parameters
    ----------
    samples : int or np.ndarray [shape=(n,)]
        sample index or vector of sample indices

    hop_length : int > 0 [scalar]
        number of samples between successive frames

    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of `- n_fft / 2`
        to counteract windowing effects in STFT.

        .. note:: This may result in negative frame indices.

    Returns
    -------
    frames : int or np.ndarray [shape=(n,), dtype=int]
        Frame numbers corresponding to the given times:
        `frames[i] = floor( samples[i] / hop_length )`

    See Also
    --------
    samples_to_time : convert sample indices to time values
    frames_to_samples : convert frame indices to sample indices
    """

    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)

    samples = np.asanyarray(samples)
    return np.floor((samples - offset) // hop_length).astype(int)


def frames_to_time(frames, sr=22050, hop_length=512, n_fft=None):
    """Converts frame counts to time (seconds).

    Parameters
    ----------
    frames     : np.ndarray [shape=(n,)]
        frame index or vector of frame indices

    sr         : number > 0 [scalar]
        audio sampling rate

    hop_length : int > 0 [scalar]
        number of samples between successive frames

    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of `n_fft / 2`
        to counteract windowing effects when using a non-centered STFT.

    Returns
    -------
    times : np.ndarray [shape=(n,)]
        time (in seconds) of each given frame number:
        `times[i] = frames[i] * hop_length / sr`

    See Also
    --------
    time_to_frames : convert time values to frame indices
    frames_to_samples : convert frame indices to sample indices

    Examples
    --------
    >>> y, sr = minispec.load(minispec.util.example_audio_file())
    >>> tempo, beats = minispec.beat.beat_track(y, sr=sr)
    >>> beat_times = minispec.frames_to_time(beats, sr=sr)
    """

    samples = frames_to_samples(frames,
                                hop_length=hop_length,
                                n_fft=n_fft)

    return samples_to_time(samples, sr=sr)


def time_to_frames(times, sr=22050, hop_length=512, n_fft=None):
    """Converts time stamps into STFT frames.

    Parameters
    ----------
    times : np.ndarray [shape=(n,)]
        time (in seconds) or vector of time values

    sr : number > 0 [scalar]
        audio sampling rate

    hop_length : int > 0 [scalar]
        number of samples between successive frames

    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of `- n_fft / 2`
        to counteract windowing effects in STFT.

        .. note:: This may result in negative frame indices.

    Returns
    -------
    frames : np.ndarray [shape=(n,), dtype=int]
        Frame numbers corresponding to the given times:
        `frames[i] = floor( times[i] * sr / hop_length )`

    See Also
    --------
    frames_to_time : convert frame indices to time values
    time_to_samples : convert time values to sample indices

    Examples
    --------
    Get the frame numbers for every 100ms

    >>> minispec.time_to_frames(np.arange(0, 1, 0.1),
    ...                         sr=22050, hop_length=512)
    array([ 0,  4,  8, 12, 17, 21, 25, 30, 34, 38])

    """

    samples = time_to_samples(times, sr=sr)

    return samples_to_frames(samples, hop_length=hop_length, n_fft=n_fft)


def time_to_samples(times, sr=22050):
    '''Convert timestamps (in seconds) to sample indices.

    Parameters
    ----------
    times : number or np.ndarray
        Time value or array of time values (in seconds)

    sr : number > 0
        Sampling rate

    Returns
    -------
    samples : int or np.ndarray [shape=times.shape, dtype=int]
        Sample indices corresponding to values in `times`

    See Also
    --------
    time_to_frames : convert time values to frame indices
    samples_to_time : convert sample indices to time values

    Examples
    --------
    >>> minispec.time_to_samples(np.arange(0, 1, 0.1), sr=22050)
    array([    0,  2205,  4410,  6615,  8820, 11025, 13230, 15435,
           17640, 19845])

    '''

    return (np.asanyarray(times) * sr).astype(int)


def samples_to_time(samples, sr=22050):
    '''Convert sample indices to time (in seconds).

    Parameters
    ----------
    samples : np.ndarray
        Sample index or array of sample indices

    sr : number > 0
        Sampling rate

    Returns
    -------
    times : np.ndarray [shape=samples.shape, dtype=int]
        Time values corresponding to `samples` (in seconds)

    See Also
    --------
    samples_to_frames : convert sample indices to frame indices
    time_to_samples : convert time values to sample indices

    Examples
    --------
    Get timestamps corresponding to every 512 samples

    >>> minispec.samples_to_time(np.arange(0, 22050, 512))
    array([ 0.   ,  0.023,  0.046,  0.07 ,  0.093,  0.116,  0.139,
            0.163,  0.186,  0.209,  0.232,  0.255,  0.279,  0.302,
            0.325,  0.348,  0.372,  0.395,  0.418,  0.441,  0.464,
            0.488,  0.511,  0.534,  0.557,  0.58 ,  0.604,  0.627,
            0.65 ,  0.673,  0.697,  0.72 ,  0.743,  0.766,  0.789,
            0.813,  0.836,  0.859,  0.882,  0.906,  0.929,  0.952,
            0.975,  0.998])
    '''

    return np.asanyarray(samples) / float(sr)


def hz_to_mel(frequencies, htk=False):
    """Convert Hz to Mels

    Examples
    --------
    >>> minispec.hz_to_mel(60)
    0.9
    >>> minispec.hz_to_mel([110, 220, 440])
    array([ 1.65,  3.3 ,  6.6 ])

    Parameters
    ----------
    frequencies   : number or np.ndarray [shape=(n,)] , float
        scalar or array of frequencies
    htk           : bool
        use HTK formula instead of Slaney

    Returns
    -------
    mels        : number or np.ndarray [shape=(n,)]
        input frequencies in Mels

    See Also
    --------
    mel_to_hz
    """

    frequencies = np.asanyarray(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = (frequencies >= min_log_hz)
        mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def mel_to_hz(mels, htk=False):
    """Convert mel bin numbers to frequencies

    Examples
    --------
    >>> minispec.mel_to_hz(3)
    200.

    >>> minispec.mel_to_hz([1,2,3,4,5])
    array([  66.667,  133.333,  200.   ,  266.667,  333.333])

    Parameters
    ----------
    mels          : np.ndarray [shape=(n,)], float
        mel bins to convert
    htk           : bool
        use HTK formula instead of Slaney

    Returns
    -------
    frequencies   : np.ndarray [shape=(n,)]
        input mels in Hz

    See Also
    --------
    hz_to_mel
    """

    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = (mels >= min_log_mel)
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs


def fft_frequencies(sr=22050, n_fft=2048):
    '''Alternative implementation of `np.fft.fftfreq`

    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate

    n_fft : int > 0 [scalar]
        FFT window size


    Returns
    -------
    freqs : np.ndarray [shape=(1 + n_fft/2,)]
        Frequencies `(0, sr/n_fft, 2*sr/n_fft, ..., sr/2)`


    Examples
    --------
    >>> minispec.fft_frequencies(sr=22050, n_fft=16)
    array([     0.   ,   1378.125,   2756.25 ,   4134.375,
             5512.5  ,   6890.625,   8268.75 ,   9646.875,  11025.   ])

    '''

    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_fft//2),
                       endpoint=True)


def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
    """Compute an array of acoustic frequencies tuned to the mel scale.

    The mel scale is a quasi-logarithmic function of acoustic frequency
    designed such that perceptually similar pitch intervals (e.g. octaves)
    appear equal in width over the full hearing range.

    Because the definition of the mel scale is conditioned by a finite number
    of subjective psychoaoustical experiments, several implementations coexist
    in the audio signal processing literature [1]_. By default, minispec replicates
    the behavior of the well-established MATLAB Auditory Toolbox of Slaney [2]_.
    According to this default implementation,  the conversion from Hertz to mel is
    linear below 1 kHz and logarithmic above 1 kHz. Another available implementation
    replicates the Hidden Markov Toolkit [3]_ (HTK) according to the following formula:

    `mel = 2595.0 * np.log10(1.0 + f / 700.0).`

    The choice of implementation is determined by the `htk` keyword argument: setting
    `htk=False` leads to the Auditory toolbox implementation, whereas setting it `htk=True`
    leads to the HTK implementation.

    .. [1] Umesh, S., Cohen, L., & Nelson, D. Fitting the mel scale.
        In Proc. International Conference on Acoustics, Speech, and Signal Processing
        (ICASSP), vol. 1, pp. 217-220, 1998.

    .. [2] Slaney, M. Auditory Toolbox: A MATLAB Toolbox for Auditory
        Modeling Work. Technical Report, version 2, Interval Research Corporation, 1998.

    .. [3] Young, S., Evermann, G., Gales, M., Hain, T., Kershaw, D., Liu, X.,
        Moore, G., Odell, J., Ollason, D., Povey, D., Valtchev, V., & Woodland, P.
        The HTK book, version 3.4. Cambridge University, March 2009.


    See Also
    --------
    hz_to_mel
    mel_to_hz
    minispec.feature.melspectrogram
    minispec.feature.mfcc


    Parameters
    ----------
    n_mels    : int > 0 [scalar]
        Number of mel bins.

    fmin      : float >= 0 [scalar]
        Minimum frequency (Hz).

    fmax      : float >= 0 [scalar]
        Maximum frequency (Hz).

    htk       : bool
        If True, use HTK formula to convert Hz to mel.
        Otherwise (False), use Slaney's Auditory Toolbox.

    Returns
    -------
    bin_frequencies : ndarray [shape=(n_mels,)]
        Vector of n_mels frequencies in Hz which are uniformly spaced on the Mel
        axis.

    Examples
    --------
    >>> minispec.mel_frequencies(n_mels=40)
    array([     0.   ,     85.317,    170.635,    255.952,
              341.269,    426.586,    511.904,    597.221,
              682.538,    767.855,    853.173,    938.49 ,
             1024.856,   1119.114,   1222.042,   1334.436,
             1457.167,   1591.187,   1737.532,   1897.337,
             2071.84 ,   2262.393,   2470.47 ,   2697.686,
             2945.799,   3216.731,   3512.582,   3835.643,
             4188.417,   4573.636,   4994.285,   5453.621,
             5955.205,   6502.92 ,   7101.009,   7754.107,
             8467.272,   9246.028,  10096.408,  11025.   ])

    """

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels, htk=htk)

