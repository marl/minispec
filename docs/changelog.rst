Changelog
=========

v0.6.2
------
2018-08-09

Bug fixes
    - `#730`_ Fixed cache support for ``joblib>=0.12``.  *Matt Vollrath*

New features
    - `#735`_ Added `core.times_like` and `core.samples_like` to generate time and sample indices
      corresponding to an existing feature matrix or shape specification. *Steve Tjoa*
    - `#750`_, `#753`_ Added `core.tone` and `core.chirp` signal generators. *Ziyao Wei*

Other changes
    - `#727`_ updated documentation for `core.get_duration`. *Zhen Wang*
    - `#731`_ fixed a typo in documentation for `core.fft_frequencies`. *Ziyao Wei*
    - `#734`_ expanded documentation for `feature.spectrall_rolloff`. *Ziyao Wei*
    - `#751`_ fixed example documentation for proper handling of phase in dB-scaling. *Vincent Lostanlen*
    - `#755`_ forward support and future-proofing for fancy indexing with ``numpy>1.15``. *Brian McFee*

.. _#730: https://github.com/minispec/minispec/pull/730
.. _#735: https://github.com/minispec/minispec/pull/735
.. _#750: https://github.com/minispec/minispec/pull/750
.. _#753: https://github.com/minispec/minispec/pull/753
.. _#727: https://github.com/minispec/minispec/pull/727
.. _#731: https://github.com/minispec/minispec/pull/731
.. _#734: https://github.com/minispec/minispec/pull/734
.. _#751: https://github.com/minispec/minispec/pull/751
.. _#755: https://github.com/minispec/minispec/pull/755

v0.6.1
------
2018-05-24

Bug fixes

  - `#677`_ `util.find_files` now correctly de-duplicates files on case-insensitive platforms. *Brian McFee*
  - `#713`_ `util.valid_intervals` now checks for non-negative durations. *Brian McFee, Dana Lee*
  - `#714`_ `util.match_intervals` can now explicitly fail when no matches are possible. *Brian McFee, Dana Lee*

New features

  - `#679`_, `#708`_ `core.pcen`, per-channel energy normalization. *Vincent Lostanlen, Brian McFee*
  - `#682`_ added different DCT modes to `feature.mfcc`. *Brian McFee*
  - `#687`_ `display` functions now accept target axes. *Pius Friesch*
  - `#688`_ numba-accelerated `util.match_events`. *Dana Lee*
  - `#710`_ `sequence` module and Viterbi decoding for generative, discriminative, and multi-label hidden Markov models. *Brian McFee*
  - `#714`_ `util.match_intervals` now supports tie-breaking for disjoint query intervals. *Brian McFee*

Other changes

  - `#677`_, `#705`_ added continuous integration testing for Windows. *Brian McFee*, *Ryuichi Yamamoto*
  - `#680`_ updated display module tests to support matplotlib 2.1. *Brian McFee*
  - `#684`_ corrected documentation for `core.stft` and `core.ifgram`. *Keunwoo Choi*
  - `#699`_, `#701`_ corrected documentation for `filters.semitone_filterbank` and `filters.mel_frequencies`. *Vincent Lostanlen*
  - `#704`_ eliminated unnecessary side-effects when importing `display`. *Brian McFee*
  - `#707`_ improved test coverage for dynamic time warping. *Brian McFee*
  - `#714`_ `util.match_intervals` matching logic has changed from raw intersection to Jaccard similarity.  *Brian McFee*


API Changes and compatibility

  - `#716`_ `core.dtw` has moved to `sequence.dtw`, and `core.fill_off_diagonal` has moved to
    `util.fill_off_diagonal`.  *Brian McFee*

.. _#716: https://github.com/minispec/minispec/pull/716
.. _#714: https://github.com/minispec/minispec/pull/714
.. _#713: https://github.com/minispec/minispec/pull/713
.. _#710: https://github.com/minispec/minispec/pull/710
.. _#708: https://github.com/minispec/minispec/pull/708
.. _#707: https://github.com/minispec/minispec/pull/707
.. _#705: https://github.com/minispec/minispec/pull/705
.. _#704: https://github.com/minispec/minispec/pull/704
.. _#701: https://github.com/minispec/minispec/pull/701
.. _#699: https://github.com/minispec/minispec/pull/699
.. _#688: https://github.com/minispec/minispec/pull/688
.. _#687: https://github.com/minispec/minispec/pull/687
.. _#684: https://github.com/minispec/minispec/pull/684
.. _#682: https://github.com/minispec/minispec/pull/682
.. _#680: https://github.com/minispec/minispec/pull/680
.. _#679: https://github.com/minispec/minispec/pull/679
.. _#677: https://github.com/minispec/minispec/pull/677

v0.6.0
------
2018-02-17

Bug fixes
  - `#663`_ fixed alignment errors in `feature.delta`. *Brian McFee*
  - `#646`_ `effects.trim` now correctly handles all-zeros signals. *Rimvydas Naktinis*
  - `#634`_ `stft` now conjugates the correct half of the spectrum. *Brian McFee*
  - `#630`_ fixed display decoration errors with `cqt_note` mode. *Brian McFee*
  - `#619`_ `effects.split` no longer returns out-of-bound sample indices. *Brian McFee*
  - `#616`_ Improved `util.valid_audio` to avoid integer type errors. *Brian McFee*
  - `#600`_ CQT basis functions are now correctly centered. *Brian McFee*
  - `#597`_ fixed frequency bin centering in `display.specshow`. *Brian McFee*
  - `#594`_ `dtw` fixed a bug which ignored weights when `step_sizes_sigma` did not match length. *Jackie Wu*
  - `#593`_ `stft` properly checks for valid input signals. *Erik Peterson*
  - `#587`_ `show_versions` now shows correct module names. *Ryuichi Yamamoto*

New features

  - `#648`_ `feature.spectral_flatness`. *Keunwoo Choi*
  - `#633`_ `feature.tempogram` now supports multi-band analysis. *Brian McFee*
  - `#439`_ `core.iirt` implements the multi-rate filterbank from Chroma Toolbox. *Stefan Balke*
  - `#435`_ `core.icqt` inverse constant-Q transform (unstable). *Brian McFee*

Other changes
  - `#674`_ Improved `write_wav` documentation with cross-references to `soundfile`. *Brian McFee*
  - `#671`_ Warn users when phase information is lost in dB conversion. *Carl Thome*
  - `#666`_ Expanded documentation for `load`'s resampling behavior. *Brian McFee*
  - `#656`_ Future-proofing numpy data type checks. *Carl Thome*
  - `#642`_ Updated unit tests for compatibility with matplotlib 2.1. *Brian McFee*
  - `#637`_ Improved documentation for advanced I/O. *Siddhartha Kumar*
  - `#636`_ `util.normalize` now preserves data type. *Brian McFee*
  - `#632`_ refined the validation requirements for `util.frame`. *Brian McFee*
  - `#628`_ all time/frequency conversion functions preserve input shape. *Brian McFee*
  - `#625`_ Numba is now a hard dependency. *Brian McFee*
  - `#622`_ `hz_to_midi` documentation corrections. *Carl Thome*
  - `#621`_ `dtw` is now symmetric with respect to input arguments. *Stefan Balke*
  - `#620`_ Updated requirements to prevent installation with (incompatible) sklearn 0.19.0. *Brian McFee*
  - `#609`_ Improved documentation for `segment.recurrence_matrix`. *Julia Wilkins*
  - `#598`_ Improved efficiency of `decompose.nn_filter`. *Brian McFee*
  - `#574`_ `dtw` now supports pre-computed distance matrices. *Curtis Hawthorne*

API changes and compatibility

  - `#627`_ The following functions and features have been removed:
      - `real=` parameter in `cqt`
      - `core.logamplitude` (replaced by `amplitude_to_db`)
      - `beat.estimate_tempo` (replaced by `beat.tempo`)
      - `n_fft=` parameter to `feature.rmse`
      - `ref_power=` parameter to `power_to_db`

  - The following features have been deprecated, and will be removed in 0.7.0:
      - `trim=` parameter to `feature.delta`

  - `#616`_ `write_wav` no longer supports integer-typed waveforms. This is due to enforcing
    consistency with `util.valid_audio` checks elsewhere in the codebase. If you have existing
    code that requires integer-valued output, consider using `soundfile.write` instead.

.. _#674: https://github.com/minispec/minispec/pull/674
.. _#671: https://github.com/minispec/minispec/pull/671
.. _#663: https://github.com/minispec/minispec/pull/663
.. _#646: https://github.com/minispec/minispec/pull/646
.. _#634: https://github.com/minispec/minispec/pull/634
.. _#630: https://github.com/minispec/minispec/pull/630
.. _#619: https://github.com/minispec/minispec/pull/619
.. _#616: https://github.com/minispec/minispec/pull/616
.. _#600: https://github.com/minispec/minispec/pull/600
.. _#597: https://github.com/minispec/minispec/pull/597
.. _#594: https://github.com/minispec/minispec/pull/594
.. _#593: https://github.com/minispec/minispec/pull/593
.. _#587: https://github.com/minispec/minispec/pull/587
.. _#648: https://github.com/minispec/minispec/pull/648
.. _#633: https://github.com/minispec/minispec/pull/633
.. _#439: https://github.com/minispec/minispec/pull/439
.. _#435: https://github.com/minispec/minispec/pull/435
.. _#666: https://github.com/minispec/minispec/pull/666
.. _#656: https://github.com/minispec/minispec/pull/656
.. _#642: https://github.com/minispec/minispec/pull/642
.. _#637: https://github.com/minispec/minispec/pull/637
.. _#636: https://github.com/minispec/minispec/pull/636
.. _#632: https://github.com/minispec/minispec/pull/632
.. _#628: https://github.com/minispec/minispec/pull/628
.. _#625: https://github.com/minispec/minispec/pull/625
.. _#622: https://github.com/minispec/minispec/pull/622
.. _#621: https://github.com/minispec/minispec/pull/621
.. _#620: https://github.com/minispec/minispec/pull/620
.. _#609: https://github.com/minispec/minispec/pull/609
.. _#598: https://github.com/minispec/minispec/pull/598
.. _#574: https://github.com/minispec/minispec/pull/574
.. _#627: https://github.com/minispec/minispec/pull/627

v0.5.1
------
2017-05-08

Bug fixes
  - `#555`_ added safety check for frequency bands in `spectral_contrast`. *Brian McFee*
  - `#554`_ fix interactive display for `tonnetz` visualization. *Brian McFee*
  - `#553`_ fix bug in `feature.spectral_bandwidth`. *Brian McFee*
  - `#539`_ fix `chroma_cens` to support scipy >=0.19. *Brian McFee*

New features

  - `#565`_ `feature.stack_memory` now supports negative delay. *Brian McFee*
  - `#563`_ expose padding mode in `stft/ifgram/cqt`. *Brian McFee*
  - `#559`_ explicit length option for `istft`. *Brian McFee*
  - `#557`_ added `show_versions`. *Brian McFee*
  - `#551`_ add `norm=` option to `filters.mel`. *Dan Ellis*

Other changes

  - `#569`_ `feature.rmse` now centers frames in the time-domain by default. *Brian McFee*
  - `#564`_ `display.specshow` now rasterizes images by default. *Brian McFee*
  - `#558`_ updated contributing documentation and issue templates. *Brian McFee*
  - `#556`_ updated tutorial for 0.5 API compatibility. *Brian McFee*
  - `#544`_ efficiency improvement in CQT. *Carl Thome*
  - `#523`_ support reading files with more than two channels. *Paul Brossier*

.. _#523: https://github.com/minispec/minispec/pull/523
.. _#544: https://github.com/minispec/minispec/pull/544
.. _#556: https://github.com/minispec/minispec/pull/556
.. _#558: https://github.com/minispec/minispec/pull/558
.. _#564: https://github.com/minispec/minispec/pull/564
.. _#551: https://github.com/minispec/minispec/pull/551
.. _#557: https://github.com/minispec/minispec/pull/557
.. _#559: https://github.com/minispec/minispec/pull/559
.. _#563: https://github.com/minispec/minispec/pull/563
.. _#565: https://github.com/minispec/minispec/pull/565
.. _#539: https://github.com/minispec/minispec/pull/539
.. _#553: https://github.com/minispec/minispec/pull/553
.. _#554: https://github.com/minispec/minispec/pull/554
.. _#555: https://github.com/minispec/minispec/pull/555
.. _#569: https://github.com/minispec/minispec/pull/569

v0.5.0
------
2017-02-17

Bug fixes

  - `#371`_ preserve integer hop lengths in constant-Q transforms. *Brian McFee*
  - `#386`_ fixed a length check in ``minispec.util.frame``. *Brian McFee*
  - `#416`_ ``minispec.output.write_wav`` only normalizes floating point, and normalization is disabled by
    default. *Brian McFee*
  - `#417`_ ``minispec.cqt`` output is now scaled continuously across octave boundaries. *Brian McFee, Eric
    Humphrey*
  - `#450`_ enhanced numerical stability for ``minispec.util.softmask``. *Brian McFee*
  - `#467`_ correction to chroma documentation. *Seth Kranzler*
  - `#501`_ fixed a numpy 1.12 compatibility error in ``pitch_tuning``. *Hojin Lee*

New features

  - `#323`_ ``minispec.dtw`` dynamic time warping. *Stefan Balke*
  - `#404`_ ``minispec.cache`` now supports priority levels, analogous to logging levels. *Brian McFee*
  - `#405`_ ``minispec.interp_harmonics`` for estimating harmonics of time-frequency representations. *Brian
    McFee*
  - `#410`_ ``minispec.beat.beat_track`` and ``minispec.onset.onset_detect`` can return output in frames,
    samples, or time units. *Brian McFee*
  - `#413`_ full support for scipy-style window specifications. *Brian McFee*
  - `#427`_ ``minispec.salience`` for computing spectrogram salience using harmonic peaks. *Rachel Bittner*
  - `#428`_ ``minispec.effects.trim`` and ``minispec.effects.split`` for trimming and splitting waveforms. *Brian
    McFee*
  - `#464`_ ``minispec.amplitude_to_db``, ``db_to_amplitude``, ``power_to_db``, and ``db_to_power`` for
    amplitude conversions.  This deprecates ``logamplitude``.  *Brian McFee*
  - `#471`_ ``minispec.util.normalize`` now supports ``threshold`` and ``fill_value`` arguments. *Brian McFee*
  - `#472`_ ``minispec.feature.melspectrogram`` now supports ``power`` argument. *Keunwoo Choi*
  - `#473`_ ``minispec.onset.onset_backtrack`` for backtracking onset events to previous local minima of
    energy. *Brian McFee*
  - `#479`_ ``minispec.beat.tempo`` replaces ``minispec.beat.estimate_tempo``, supports time-varying estimation.
    *Brian McFee*
  

Other changes

  - `#352`_ removed ``seaborn`` integration. *Brian McFee*
  - `#368`_ rewrite of the ``minispec.display`` submodule.  All plots are now in natural coordinates. *Brian
    McFee*
  - `#402`_ ``minispec.display`` submodule is not automatically imported. *Brian McFee*
  - `#403`_ ``minispec.decompose.hpss`` now returns soft masks. *Brian McFee*
  - `#407`_ ``minispec.feature.rmse`` can now compute directly in the time domain. *Carl Thome*
  - `#432`_ ``minispec.feature.rmse`` renames ``n_fft`` to ``frame_length``. *Brian McFee*
  - `#446`_ ``minispec.cqt`` now disables tuning estimation by default. *Brian McFee*
  - `#452`_ ``minispec.filters.__float_window`` now always uses integer length windows. *Brian McFee*
  - `#459`_ ``minispec.load`` now supports ``res_type`` argument for resampling. *CJ Carr*
  - `#482`_ ``minispec.filters.mel`` now warns if parameters will generate empty filter channels. *Brian McFee*
  - `#480`_ expanded documentation for advanced IO use-cases. *Fabian Robert-Stoeter*

API changes and compatibility

  - The following functions have permanently moved:
        - ``core.peak_peak`` to ``util.peak_pick``
        - ``core.localmax`` to ``util.localmax``
        - ``feature.sync`` to ``util.sync``

  - The following functions, classes, and constants have been removed:
        - ``core.ifptrack``
        - ``feature.chromagram``
        - ``feature.logfsgram``
        - ``filters.logfrequency``
        - ``output.frames_csv``
        - ``segment.structure_Feature``
        - ``display.time_ticks``
        - ``util.FeatureExtractor``
        - ``util.buf_to_int``
        - ``util.SMALL_FLOAT``

  - The following parameters have been removed:
        - ``minispec.cqt``: `resolution`
        - ``minispec.cqt``: `aggregate`
        - ``feature.chroma_cqt``: `mode`
        - ``onset_strength``: `centering`

  - Seaborn integration has been removed, and the ``display`` submodule now requires matplotlib >= 1.5.
        - The `use_sns` argument has been removed from `display.cmap`
        - `magma` is now the default sequential colormap.

  - The ``minispec.display`` module has been rewritten.
        - ``minispec.display.specshow`` now plots using `pcolormesh`, and supports non-uniform time and frequency axes.
        - All plots can be rendered in natural coordinates (e.g., time or Hz)
        - Interactive plotting is now supported via ticker and formatter objects

  - ``minispec.decompose.hpss`` with `mask=True` now returns soft masks, rather than binary masks.

  - ``minispec.filters.get_window`` wraps ``scipy.signal.get_window``, and handles generic callables as well pre-registered
    window functions.  All windowed analyses (e.g., ``stft``, ``cqt``, or ``tempogram``) now support the full range
    of window functions and parameteric windows via tuple parameters, e.g., ``window=('kaiser', 4.0)``.
        
  - ``stft`` windows are now explicitly asymmetric by default, which breaks backwards compatibility with the 0.4 series.

  - ``cqt`` now returns properly scaled outputs that are continuous across octave boundaries.  This breaks
    backwards compatibility with the 0.4 series.

  - ``cqt`` now uses `tuning=0.0` by default, rather than estimating the tuning from the signal.  Tuning
    estimation is still supported, and enabled by default for chroma analysis (``minispec.feature.chroma_cqt``).

  - ``logamplitude`` is deprecated in favor of ``amplitude_to_db`` or ``power_to_db``.  The `ref_power` parameter
    has been renamed to `ref`.


.. _#501: https://github.com/minispec/minispec/pull/501
.. _#480: https://github.com/minispec/minispec/pull/480
.. _#467: https://github.com/minispec/minispec/pull/467
.. _#450: https://github.com/minispec/minispec/pull/450
.. _#417: https://github.com/minispec/minispec/pull/417
.. _#416: https://github.com/minispec/minispec/pull/416
.. _#386: https://github.com/minispec/minispec/pull/386
.. _#371: https://github.com/minispec/minispec/pull/371
.. _#479: https://github.com/minispec/minispec/pull/479
.. _#473: https://github.com/minispec/minispec/pull/473
.. _#472: https://github.com/minispec/minispec/pull/472
.. _#471: https://github.com/minispec/minispec/pull/471
.. _#464: https://github.com/minispec/minispec/pull/464
.. _#428: https://github.com/minispec/minispec/pull/428
.. _#427: https://github.com/minispec/minispec/pull/427
.. _#413: https://github.com/minispec/minispec/pull/413
.. _#410: https://github.com/minispec/minispec/pull/410
.. _#405: https://github.com/minispec/minispec/pull/405
.. _#404: https://github.com/minispec/minispec/pull/404
.. _#323: https://github.com/minispec/minispec/pull/323
.. _#482: https://github.com/minispec/minispec/pull/482
.. _#459: https://github.com/minispec/minispec/pull/459
.. _#452: https://github.com/minispec/minispec/pull/452
.. _#446: https://github.com/minispec/minispec/pull/446
.. _#432: https://github.com/minispec/minispec/pull/432
.. _#407: https://github.com/minispec/minispec/pull/407
.. _#403: https://github.com/minispec/minispec/pull/403
.. _#402: https://github.com/minispec/minispec/pull/402
.. _#368: https://github.com/minispec/minispec/pull/368
.. _#352: https://github.com/minispec/minispec/pull/352



v0.4.3
------
2016-05-17

Bug fixes
  - `#315`_ fixed a positioning error in ``display.specshow`` with logarithmic axes. *Brian McFee*
  - `#332`_ ``minispec.cqt`` now throws an exception if the signal is too short for analysis. *Brian McFee*
  - `#341`_ ``minispec.hybrid_cqt`` properly matches the scale of ``minispec.cqt``. *Brian McFee*
  - `#348`_ ``minispec.cqt`` fixed a bug introduced in v0.4.2. *Brian McFee*
  - `#354`_ Fixed a minor off-by-one error in ``minispec.beat.estimate_tempo``. *Brian McFee*
  - `#357`_ improved numerical stability of ``minispec.decompose.hpss``. *Brian McFee*

New features
  - `#312`_ ``minispec.segment.recurrence_matrix`` can now construct sparse self-similarity matrices. *Brian
    McFee*
  - `#337`_ ``minispec.segment.recurrence_matrix`` can now produce weighted affinities and distances. *Brian
    McFee*
  - `#311`_ ``minispec.decompose.nl_filter`` implements several self-similarity based filtering operations
    including non-local means. *Brian McFee*
  - `#320`_ ``minispec.feature.chroma_cens`` implements chroma energy normalized statistics (CENS) features.
    *Stefan Balke*
  - `#354`_ ``minispec.core.tempo_frequencies`` computes tempo (BPM) frequencies for autocorrelation and
    tempogram features. *Brian McFee*
  - `#355`_ ``minispec.decompose.hpss`` now supports harmonic-percussive-residual separation. *CJ Carr, Brian McFee*
  - `#357`_ ``minispec.util.softmask`` computes numerically stable soft masks. *Brian McFee*

Other changes
  - ``minispec.cqt``, ``minispec.hybrid_cqt`` parameter `aggregate` is now deprecated.
  - Resampling is now handled by the ``resampy`` library
  - ``minispec.get_duration`` can now operate directly on filenames as well as audio buffers and feature
    matrices.
  - ``minispec.decompose.hpss`` no longer supports ``power=0``.

.. _#315: https://github.com/minispec/minispec/pull/315
.. _#332: https://github.com/minispec/minispec/pull/332
.. _#341: https://github.com/minispec/minispec/pull/341
.. _#348: https://github.com/minispec/minispec/pull/348
.. _#312: https://github.com/minispec/minispec/pull/312
.. _#337: https://github.com/minispec/minispec/pull/337
.. _#311: https://github.com/minispec/minispec/pull/311
.. _#320: https://github.com/minispec/minispec/pull/320
.. _#354: https://github.com/minispec/minispec/pull/354
.. _#355: https://github.com/minispec/minispec/pull/355
.. _#357: https://github.com/minispec/minispec/pull/357

v0.4.2
------
2016-02-20

Bug fixes
  - Support for matplotlib 1.5 color properties in the ``display`` module
  - `#308`_ Fixed a per-octave scaling error in ``minispec.cqt``. *Brian McFee*

New features
  - `#279`_ ``minispec.cqt`` now provides complex-valued output with argument `real=False`.
    This will become the default behavior in subsequent releases.
  - `#288`_ ``core.resample`` now supports multi-channel inputs. *Brian McFee*
  - `#295`_ ``minispec.display.frequency_ticks``: like ``time_ticks``. Ticks can now dynamically
    adapt to scale (mHz, Hz, KHz, MHz, GHz) and use automatic precision formatting (``%g``). *Brian McFee*


Other changes
  - `#277`_ improved documentation for OSX. *Stefan Balke*
  - `#294`_ deprecated the ``FeatureExtractor`` object. *Brian McFee*
  - `#300`_ added dependency version requirements to install script. *Brian McFee*
  - `#302`_, `#279`_ renamed the following parameters
      - ``minispec.display.time_ticks``: `fmt` is now `time_fmt`
      - ``minispec.feature.chroma_cqt``: `mode` is now `cqt_mode`
      - ``minispec.cqt``, ``hybrid_cqt``, ``pseudo_cqt``, ``minispec.filters.constant_q``: `resolution` is now `filter_scale`
  - `#308`_ ``minispec.cqt`` default `filter_scale` parameter is now 1 instead of 2.

.. _#277: https://github.com/minispec/minispec/pull/277
.. _#279: https://github.com/minispec/minispec/pull/279
.. _#288: https://github.com/minispec/minispec/pull/288
.. _#294: https://github.com/minispec/minispec/pull/294
.. _#295: https://github.com/minispec/minispec/pull/295
.. _#300: https://github.com/minispec/minispec/pull/300
.. _#302: https://github.com/minispec/minispec/pull/302
.. _#308: https://github.com/minispec/minispec/pull/308

v0.4.1
------
2015-10-17

Bug fixes
  - Improved safety check in CQT for invalid hop lengths
  - Fixed division by zero bug in ``core.pitch.pip_track``
  - Fixed integer-type error in ``util.pad_center`` on numpy v1.10
  - Fixed a context scoping error in ``minispec.load`` with some audioread backends
  - ``minispec.autocorrelate`` now persists type for complex input

New features
  - ``minispec.clicks`` sonifies timed events such as beats or onsets
  - ``minispec.onset.onset_strength_multi`` computes onset strength within multiple sub-bands
  - ``minispec.feature.tempogram`` computes localized onset strength autocorrelation
  - ``minispec.display.specshow`` now supports ``*_axis='tempo'`` for annotating tempo-scaled data
  - ``minispec.fmt`` implements the Fast Mellin Transform

Other changes

  - Rewrote ``display.waveplot`` for improved efficiency
  - ``decompose.deompose()`` now supports pre-trained transformation objects
  - Nullified side-effects of optional seaborn dependency
  - Moved ``feature.sync`` to ``util.sync`` and expanded its functionality
  - ``minispec.onset.onset_strength`` and ``onset_strength_multi`` support superflux-style lag and max-filtering
  - ``minispec.core.autocorrelate`` can now operate along any axis of multi-dimensional input
  - the ``segment`` module functions now support arbitrary target axis
  - Added proper window normalization to ``minispec.core.istft`` for better reconstruction
    (`PR #235 <https://github.com/minispec/minispec/pull/235>`_).
  - Standardized ``n_fft=2048`` for ``piptrack``, ``ifptrack`` (deprecated), and
    ``logfsgram`` (deprecated)
  - ``onset_strength`` parameter ``'centering'`` has been deprecated and renamed to
    ``'center'``
  - ``onset_strength`` always trims to match the input spectrogram duration
  - added tests for ``piptrack``
  - added test support for Python 3.5




v0.4.0
------
2015-07-08

Bug fixes

-  Fixed alignment errors with ``offset`` and ``duration`` in ``load()``
-  Fixed an edge-padding issue with ``decompose.hpss()`` which resulted
   in
   percussive noise leaking into the harmonic component.
-  Fixed stability issues with ``ifgram()``, added options to suppress
   negative frequencies.
-  Fixed scaling and padding errors in ``feature.delta()``
-  Fixed some errors in ``note_to_hz()`` string parsing
-  Added robust range detection for ``display.cmap``
-  Fixed tick placement in ``display.specshow``
-  Fixed a low-frequency filter alignment error in ``cqt``
-  Added aliasing checks for ``cqt`` filterbanks
-  Fixed corner cases in ``peak_pick``
-  Fixed bugs in ``find_files()`` with negative slicing
-  Fixed tuning estimation errors
-  Fixed octave numbering in to conform to scientific pitch notation

New features

-  python 3 compatibility
-  Deprecation and moved-function warnings
-  added ``norm=None`` option to ``util.normalize()``
-  ``segment.recurrence_to_lag``, ``lag_to_recurrence``
-  ``core.hybrid_cqt()`` and ``core.pseudo_cqt()``
-  ``segment.timelag_filter``
-  Efficiency enhancements for ``cqt``
-  Major rewrite and reformatting of documentation
-  Improvements to ``display.specshow``:

   -  added the ``lag`` axis format
   -  added the ``tonnetz`` axis format
   -  allow any combination of axis formats

-  ``effects.remix()``
-  Added new time and frequency converters:

   -  ``note_to_hz()``, ``hz_to_note()``
   -  ``frames_to_samples()``, ``samples_to_frames()``
   -  ``time_to_samples()``, ``samples_to_time()``

-  ``core.zero_crossings``
-  ``util.match_events()``
-  ``segment.subsegment()`` for segmentation refinement
-  Functional examples in almost all docstrings
-  improved numerical stability in ``normalize()``
-  audio validation checks
-  ``to_mono()``
-  ``minispec.cache`` for storing pre-computed features
-  Stereo output support in ``write_wav``
-  Added new feature extraction functions:

   -  ``feature.spectral_contrast``
   -  ``feature.spectral_bandwidth``
   -  ``feature.spectral_centroid``
   -  ``feature.spectral_rolloff``
   -  ``feature.poly_features``
   -  ``feature.rmse``
   -  ``feature.zero_crossing_rate``
   -  ``feature.tonnetz``

- Added ``display.waveplot``

Other changes

-  Internal refactoring and restructuring of submodules
-  Removed the ``chord`` module
-  input validation and better exception reporting for most functions
-  Changed the default colormaps in ``display``
-  Changed default parameters in onset detection, beat tracking
-  Changed default parameters in ``cqt``
-  ``filters.constant_q`` now returns filter lengths
-  Chroma now starts at ``C`` by default, instead of ``A``
-  ``pad_center`` supports multi-dimensional input and ``axis``
   parameter
- switched from ``np.fft`` to ``scipy.fftpack`` for FFT operations
- changed all minispec-generated exception to a new class minispec.ParameterError

Deprecated functions

-  ``util.buf_to_int``
-  ``output.frames_csv``
-  ``segment.structure_feature``
-  ``filters.logfrequency``
-  ``feature.logfsgram``

v0.3.1
------
2015-02-18

Bug fixes

-  Fixed bug #117: ``minispec.segment.agglomerative`` now returns a
   numpy.ndarray instead of a list
-  Fixed bug #115: off-by-one error in ``minispec.core.load`` with fixed
   duration
-  Fixed numerical underflow errors in ``minispec.decompose.hpss``
-  Fixed bug #104: ``minispec.decompose.hpss`` failed with silent,
   complex-valued input
-  Fixed bug #103: ``minispec.feature.estimate_tuning`` fails when no
   bins exceed the threshold

Features

-  New function ``minispec.core.get_duration()`` computes the duration of
   an audio signal
   or spectrogram-like input matrix
-  ``minispec.util.pad_center`` now accepts multi-dimensional input

Other changes

-  Adopted the ISC license
-  Python 3 compatibility via futurize
-  Fixed issue #102: segment.agglomerative no longer depends on the
   deprecated
   Ward module of sklearn; it now depends on the newer Agglomerative
   module.
-  Issue #108: set character encoding on all source files
-  Added dtype persistence for resample, stft, istft, and effects
   functions

v0.3.0
------
2014-06-30

Bug fixes

-  Fixed numpy array indices to force integer values
-  ``minispec.util.frame`` now warns if the input data is non-contiguous
-  Fixed a formatting error in ``minispec.display.time_ticks()``
-  Added a warning if ``scikits.samplerate`` is not detected

Features

-  New module ``minispec.chord`` for training chord recognition models
-  Parabolic interpolation piptracking ``minispec.feature.piptrack()``
-  ``minispec.localmax()`` now supports multi-dimensional slicing
-  New example scripts
-  Improved documentation
-  Added the ``minispec.util.FeatureExtractor`` class, which allows
   minispec functions
   to act as feature extraction stages in ``sklearn``
-  New module ``minispec.effects`` for time-domain audio processing
-  Added demo notebooks for the ``minispec.effects`` and
   ``minispec.util.FeatureExtractor``
-  Added a full-track audio example,
   ``minispec.util.example_audio_file()``
-  Added peak-frequency sorting of basis elements in
   ``minispec.decompose.decompose()``

Other changes

-  Spectrogram frames are now centered, rather than left-aligned. This
   removes the
   need for window correction in ``minispec.frames_to_time()``
-  Accelerated constant-Q transform ``minispec.cqt()``
-  PEP8 compliance
-  Removed normalization from ``minispec.feature.logfsgram()``
-  Efficiency improvements by ensuring memory contiguity
-  ``minispec.logamplitude()`` now supports functional reference power,
   in addition
   to scalar values
-  Improved ``minispec.feature.delta()``
-  Additional padding options to ``minispec.feature.stack_memory()``
-  ``minispec.cqt`` and ``minispec.feature.logfsgram`` now use the same
   parameter
   formats ``(fmin, n_bins, bins_per_octave)``.
-  Updated demo notebook(s) to IPython 2.0
-  Moved ``perceptual_weighting()`` from ``minispec.feature`` into
   ``minispec.core``
-  Moved ``stack_memory()`` from ``minispec.segment`` into
   ``minispec.feature``
-  Standardized ``minispec.output.annotation`` input format to match
   ``mir_eval``
-  Standardized variable names (e.g., ``onset_envelope``).

v0.2.1
------
2014-01-21

Bug fixes

-  fixed an off-by-one error in ``minispec.onset.onset_strength()``
-  fixed a sign-flip error in ``minispec.output.write_wav()``
-  removed all mutable object default parameters

Features

-  added option ``centering`` to ``minispec.onset.onset_strength()`` to
   resolve frame-centering issues with sliding window STFT
-  added frame-center correction to ``minispec.core.frames_to_time()``
   and ``minispec.core.time_to_frames()``
-  added ``minispec.util.pad_center()``
-  added ``minispec.output.annotation()``
-  added ``minispec.output.times_csv()``
-  accelerated ``minispec.core.stft()`` and ``ifgram()``
-  added ``minispec.util.frame`` for in-place signal framing
-  ``minispec.beat.beat_track`` now supports user-supplied tempo
-  added ``minispec.util.normalize()``
-  added ``minispec.util.find_files()``
-  added ``minispec.util.axis_sort()``
-  new module: ``minispec.util()``
-  ``minispec.filters.constant_q`` now support padding
-  added boolean input support for ``minispec.display.cmap()``
-  speedup in ``minispec.core.cqt()``

Other changes

-  optimized default parameters for ``minispec.onset.onset_detect``
-  set ``minispec.filters.mel`` parameter ``n_mels=128`` by default
-  ``minispec.feature.chromagram()`` and ``logfsgram()`` now use power
   instead of energy
-  ``minispec.display.specshow()`` with ``y_axis='chroma'`` now labels as
   ``pitch class``
-  set ``minispec.core.cqt`` parameter ``resolution=2`` by default
-  set ``minispec.feature.chromagram`` parameter ``octwidth=2`` by
   default

v0.2.0
------
2013-12-14

Bug fixes

-  fixed default ``minispec.core.stft, istft, ifgram`` to match
   specification
-  fixed a float->int bug in peak\_pick
-  better memory efficiency
-  ``minispec.segment.recurrence_matrix`` corrects for width suppression
-  fixed a divide-by-0 error in the beat tracker
-  fixed a bug in tempo estimation with short windows
-  ``minispec.feature.sync`` now supports 1d arrays
-  fixed a bug in beat trimming
-  fixed a bug in ``minispec.core.stft`` when calculating window size
-  fixed ``minispec.core.resample`` to support stereo signals

Features

-  added filters option to cqt
-  added window function support to istft
-  added an IPython notebook demo
-  added ``minispec.features.delta`` for computing temporal difference
   features
-  new ``examples`` scripts: tuning, hpss
-  added optional trimming to ``minispec.segment.stack_memory``
-  ``minispec.onset.onset_strength`` now takes generic spectrogram
   function ``feature``
-  compute reference power directly in ``minispec.core.logamplitude``
-  color-blind-friendly default color maps in ``minispec.display.cmap``
-  ``minispec.core.onset_strength`` now accepts an aggregator
-  added ``minispec.feature.perceptual_weighting``
-  added tuning estimation to ``minispec.feature.chromagram``
-  added ``minispec.core.A_weighting``
-  vectorized frequency converters
-  added ``minispec.core.cqt_frequencies`` to get CQT frequencies
-  ``minispec.core.cqt`` basic constant-Q transform implementation
-  ``minispec.filters.cq_to_chroma`` to convert log-frequency to chroma
-  added ``minispec.core.fft_frequencies``
-  ``minispec.decompose.hpss`` can now return masking matrices
-  added reversal for ``minispec.segment.structure_feature``
-  added ``minispec.core.time_to_frames``
-  added cent notation to ``minispec.core.midi_to_note``
-  added time-series or spectrogram input options to ``chromagram``,
   ``logfsgram``, ``melspectrogram``, and ``mfcc``
-  new module: ``minispec.display``
-  ``minispec.output.segment_csv`` => ``minispec.output.frames_csv``
-  migrated frequency converters to ``minispec.core``
-  new module: ``minispec.filters``
-  ``minispec.decompose.hpss`` now supports complex-valued STFT matrices
-  ``minispec.decompose.decompose()`` supports ``sklearn`` decomposition
   objects
-  added ``minispec.core.phase_vocoder``
-  new module: ``minispec.onset``; migrated onset strength from
   ``minispec.beat``
-  added ``minispec.core.pick_peaks``
-  ``minispec.core.load()`` supports offset and duration parameters
-  ``minispec.core.magphase()`` to separate magnitude and phase from a
   complex matrix
-  new module: ``minispec.segment``

Other changes

-  ``onset_estimate_bpm => estimate_tempo``
-  removed ``n_fft`` from ``minispec.core.istft()``
-  ``minispec.core.mel_frequencies`` returns ``n_mels`` values by default
-  changed default ``minispec.decompose.hpss`` window to 31
-  disabled onset de-trending by default in
   ``minispec.onset.onset_strength``
-  added complex-value warning to ``minispec.display.specshow``
-  broke compatibilty with ``ifgram.m``; ``minispec.core.ifgram`` now
   matches ``stft``
-  changed default beat tracker settings
-  migrated ``hpss`` into ``minispec.decompose``
-  changed default ``minispec.decompose.hpss`` power parameter to ``2.0``
-  ``minispec.core.load()`` now returns single-precision by default
-  standardized ``n_fft=2048``, ``hop_length=512`` for most functions
-  refactored tempo estimator

v0.1.0
------

Initial public release.
