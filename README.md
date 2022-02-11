# MINESweeper
=====

**M**IST **I**sochrones with **NE**sted **S**ampling

Spectrophotometric fitting code using latest mass-tracks from the MIST models as priors.

Version 2.0

Model interpolation and nested-sampling inference of observed stellar SED and/or spectra using the latest MIST stellar evolution models. The code has the following functionality:

* Uses a quick and efficient nearest-neighbor look up to quickly do N-D linear interpolation of MIST mass-tracks.

* Samples MIST models in natural parameter space (EEP, initial_mass, initial_[Fe/H], and  initial_[a/Fe]).

* Allows for likelihood comparison with predicted spectra and/or photometry from the latest set of C3K models.

* Flexible prior functions allows for the handling of observables (e.g., Gaia parallax), as well as any predicted parameter from the MIST models (e.g., 

The MINESweeper code is described in detail in [Cargile et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...900...28C). Information regarding the MIST models can be found in [Choi et al. (2016)](http://adsabs.harvard.edu/abs/2016ApJ...823..102C), and detailed information about how we interpolate stellar models is given in [Dotter (2016)](http://adsabs.harvard.edu/abs/2016ApJS..222....8D). Information about the nested-sampler used in MINESweeper can be found in [Speagle (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S). MINESweeper has been developed in collaboration with The Payne, see [Ting et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...879...69T/).

MINESweeper has been used in various studies, a current list of references can be found here: [MINESweeper Refs](https://ui.adsabs.harvard.edu/abs/2020ApJ...900...28C/citations).

The current version of the MINESweeper code is still under development. Anyone interested in using this version of the code, please first contact <pcargile@cfa.harvard.edu>.


Author
-------
* **Phillip Cargile** (Harvard)

See [Authors](AUTHORS.md) for a full list of contributors to MINESweeper.

Requirements
-------

The code is being developed and tested with Python 3.X.

Python modules:

* numpy
* scipy
* h5py
* dynesty 

All of these modules can be installed using a simple pip install [package] except torch. See PyTorch website for installation instructions.

Installation
------
```
cd <install_dir>
git clone https://github.com/pacargile/MINESweeper.git
cd MINESweeper
python -m pip install .
```

Then in Python
```python
import minesweeper
```

MINESweeper is pure python.
See the [tutorial](demo/) for scripts used to fit a dwarf star.

To use MINESweepr the user must have the MIST mass-tracks in a HDF5 format, as well as the ANN for calculating synthetic spectra and photometry. Please contact <pcargile@cfa.harvard.edu> to get the latest model files.


License
--------

Copyright 2018. MINESweeper is open-source software released under 
the MIT License. See the file ``LICENSE`` for details.
