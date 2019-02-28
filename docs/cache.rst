Caching
^^^^^^^

This section covers the *minispec* function cache.  This allows you
to store and re-use intermediate computations across sessions.

Enabling the cache
------------------
By default, caching is disabled.  To enable caching, the environment 
variable `MINISPEC_CACHE_DIR` must be set prior to loading *minispec*.
This can be done on the command line prior to instantiating a python interpreter::

    $ export MINISPEC_CACHE_DIR=/tmp/minispec_cache
    $ ipython

or from within python, prior to importing *minispec*::

    >>> import os
    >>> os.environ['MINISPEC_CACHE_DIR'] = '/tmp/minispec_cache'
    >>> import minispec

.. warning::
    The cache does not implement any eviction policy.  As such, 
    it can grow without bound on disk if not purged.
    To purge the cache directly, call::

        >>> minispec.cache.clear()


Cache configuration
-------------------
The cache is implemented on top of `joblib.Memory <https://pythonhosted.org/joblib/memory.html>`_.
The default configuration can be overridden by setting the following environment variables

  - `MINISPEC_CACHE_DIR` : path (on disk) to the cache directory
  - `MINISPEC_CACHE_MMAP` : optional memory mapping mode `{None, 'r+', 'r', 'w+', 'c'}`
  - `MINISPEC_CACHE_COMPRESS` : flag to enable compression of data on disk `{0, 1}`
  - `MINISPEC_CACHE_VERBOSE` : controls how much debug info is displayed. `{int, non-negative}`
  - `MINISPEC_CACHE_LEVEL` : controls the caching level: the larger this value, the more data is cached. `{int}`

Please refer to the `joblib.Memory` `documentation
<https://pythonhosted.org/joblib/memory.html#memory-reference>`_ for a detailed explanation of these
parameters.

As of 0.7, minispec's cache wraps (rather than extends) the `joblib.Memory` object.
The memory object can be directly accessed by `minispec.cache.memory`.


Cache levels
------------

Cache levels operate in a fashion similar to logging levels.
For small values of `MINISPEC_CACHE_LEVEL`, only the most important (frequently used) data are cached.
As the cache level increases, broader classes of functions are cached.
As a result, application code may run faster at the expense of larger disk usage.

The caching levels are described as follows:

    - 10: filter bases, independent of audio data (mel)
    - 20: low-level features (stft, etc)
    - 30: high-level features (tempo, beats, decomposition, recurrence, etc)
    - 40: post-processing (stack_memory, normalize, sync)

The default cache level is 10.
