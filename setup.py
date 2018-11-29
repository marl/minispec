from setuptools import setup, find_packages
import sys


if sys.version_info.major == 2:
    import imp

    version = imp.load_source('minispec.version', 'minispec/version.py')
else:
    from importlib.machinery import SourceFileLoader

    version = SourceFileLoader('minispec.version',
                               'minispec/version.py').load_module()

with open('README.md', 'r') as fdesc:
    long_description = fdesc.read()

setup(
    name='minispec',
    version=version.version,
    description='Minimal module for computing audio spectrograms',
    author='Jason Cramer, Ho-Hsiang Wu, Justin Salamon, and Mark Cartwright',
    author_email='jtcramer@nyu.edu',
    url='http://github.com/marl/minispec',
    download_url='http://github.com/marl/minispec/releases',
    packages=find_packages(),
    package_data={'': ['example_data/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    keywords='audio music sound',
    license='ISC',
    install_requires=[
        'numpy >= 1.8.0',
        'scipy >= 0.14.0',
        'six >= 1.3',
    ],
    extras_require={
        'docs': ['numpydoc', 'sphinx!=1.3.1', 'sphinx_rtd_theme',
                 'sphinxcontrib-versioning >= 2.2.1'],
        'tests': ['resampy >= 0.2.0',
                  'audioread >= 2.0.0',
                  'pytest-mpl',
                  'pytest-cov',
                  'pytest < 4']
    }
)
