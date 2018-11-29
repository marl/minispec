minispec
=======
A minimal module for computing audio spectrograms.

[![PyPI](https://img.shields.io/pypi/v/minispec.svg)](https://pypi.python.org/pypi/minispec)
[![License](https://img.shields.io/pypi/l/minispec.svg)](https://github.com/marl/minispec/blob/master/LICENSE.md)
[![Build Status](https://travis-ci.org/marl/minispec.png?branch=master)](http://travis-ci.org/marl/minispec?branch=master)
[![Coverage Status](https://coveralls.io/repos/marl/minispec/badge.svg?branch=master)](https://coveralls.io/r/marl/minispec?branch=master)
[![Documentation Status](https://readthedocs.org/projects/minispec/badge/?version=latest)](http://minispec.readthedocs.org/en/latest/?badge=latest)

This module merely strips out all of the spectrogram and Mel spectrogram implementations from [librosa](https://github.com/librosa/librosa).



Documentation
-------------
See http://minispec.readthedocs.org for a reference manual.


Installation
------------

The latest stable release is available on PyPI, and you can install it by saying
```
pip install minispec
```

To build minispec from source, say `python setup.py build`.
Then, to install minispec, say `python setup.py install`.

Alternatively, you can download or clone the repository and use `pip` to handle dependencies:

```
unzip minispec.zip
pip install -e minispec
```
or
```
git clone https://github.com/marl/minispec.git
pip install -e minispec
```

By calling `pip list` you should see `minispec` now as an installed package:
```
minispec (0.x.x, /path/to/minispec)
```

### Hints for the Installation

Citing
------
- If you wish to cite minispec for its design, motivation etc., please cite the librosa paper
  published at SciPy 2015:

    McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. "librosa: Audio and music signal analysis in python." In Proceedings of the 14th python in science conference, pp. 18-25. 2015.
