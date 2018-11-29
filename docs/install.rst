Installation instructions
^^^^^^^^^^^^^^^^^^^^^^^^^

pypi
~~~~
The simplest way to install *minispec* is through the Python Package Index (PyPI).
This will ensure that all required dependencies are fulfilled.
This can be achieved by executing the following command::

    pip install minispec

or::

    sudo pip install minispec

to install system-wide, or::

    pip install -u minispec

to install just for your own user.

Source
~~~~~~

If you've downloaded the archive manually from the `releases
<https://github.com/minispec/minispec/releases/>`_ page, you can install using the
`setuptools` script::

    tar xzf minispec-VERSION.tar.gz
    cd minispec-VERSION/
    python setup.py install

If you intend to develop minispec or make changes to the source code, you can
install with `pip install -e` to link to your actively developed source tree::

    tar xzf minispec-VERSION.tar.gz
    cd minispec-VERSION/
    pip install -e .

Alternately, the latest development version can be installed via pip::

    pip install git+https://github.com/minispec/minispecbrew install ffmpeg` or get a binary version from their website https://www.ffmpeg.org.
