from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "MC: is a project to analyse EEG and psychophysical data collected to examine motion-onset VEPs."
# Long description will go up on the pypi page
long_description = """

MC
========
MC: is a project to analyse EEG and psychophysical data collected to examine motion-onset VEPs.

It contains software implementations of an analysis of EEG data and psychophysical data.
It contains infrastructure for testing, documentation,
continuous integration and deployment, which can hopefully be easily adapted
to use in other projects.

To get started using these components in your own software, please go to the
repository README_.

.. _README: https://github.com/mert-ozkan/Psyc161FinalProject/edit/master/README.md


"""

NAME = "MC"
MAINTAINER = "Mert Ozkan"
MAINTAINER_EMAIL = "mertozkan42@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/mert-ozkan/MotionCoherence"
DOWNLOAD_URL = ""
LICENSE = ""
AUTHOR = "Mert Ozkan"
AUTHOR_EMAIL = "mertozkan42@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'MC': [pjoin('data', '*')]}
REQUIRES = ["numpy"]
