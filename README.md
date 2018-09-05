# MINESweeper
=====

**M**IST **I**sochrones with **NE**sted **S**ampling

Isochrone fitting code using latest mass-tracks from the MIST models.

Version 2.0

Model interpolation and nested-sampling inference of observed stellar SED and/or parameters using the latest MIST stellar evolution models. The code has the following functionality:

* Uses a quick and efficient nearest-neighbor look up to quickly do N-D linear interpolation of MIST mass-tracks.

* Samples MIST models in natural parameter space (EEP,initial_[Fe/H],initial_mass).

* Allows for likelihood comparison of any parameter predicted in the MIST models.

A write up for the MINESweeper code is in prep., however, information regarding the MIST models can be found in [Choi et al. (2016)](http://adsabs.harvard.edu/abs/2016ApJ...823..102C), and detailed information about how we interpolate stellar models is given in [Dotter (2016)](http://adsabs.harvard.edu/abs/2016ApJS..222....8D). MINESweeper has been used in various studies, including: [Rodriguez et al. (2017)](http://adsabs.harvard.edu/abs/2017AJ....153..256R) and [Dotter et al. (2017)](http://adsabs.harvard.edu/abs/2017ApJ...840...99D).

The current version of the MINESweeper code is still under development. Anyone interested in using this version of the code, please first contact <pcargile@cfa.harvard.edu>.


Author
-------
* **Phillip Cargile** (Harvard)

See [Authors](authors.rst) for a full list of contributors to MINESweeper.

Requirements
-------

Currently, the code is only tested on Python 2.7.X. Use it with Python 3.X at your own risk.

Python modules:

* numpy
* scipy
* h5py
* dynesty 

* torch 

All of these modules can be installed using a simple pip install [package] except torch. See PyTorch website for installation instructions.

Installation
------
```
cd <install_dir>
git clone https://github.com/pacargile/MINESweeper.git
cd MINESweeper
python setup.py install (--user)
```

Then in Python
```python
import minesweeper
```

MINESweeper is pure python.
See the [tutorial](demo/) for scripts used to fit mock dwarf and giant stars.

To use MINESweepr the user must have the MIST mass-tracks in a HDF5 format, as well as the ANN for calculating synthetic photometry. Please contact <pcargile@cfa.harvard.edu> to get the latest model files.


License
--------

Copyright 2018. MINESweeper is open-source software released under 
the MIT License. See the file ``LICENSE`` for details.
