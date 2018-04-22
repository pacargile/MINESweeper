from copy import deepcopy
from itertools import product
import time, sys
import numpy as np
import h5py
#try:
#    from sklearn.neighbors import KDTree
#except(ImportError):
from scipy.spatial import cKDTree as KDTree

rename = {# Input
          "mini": "initial_mass",
          "eep": "EEP",
          "feh": "initial_[Fe/H]",
          "afe": "initial_[a/Fe]",
          # Output
          "mass": "star_mass",
          "feh_surf": "[Fe/H]",
          "loga": "log_age",
          "logt": "log_Teff",
          "logg": "log_g",
          "logl": "log_L",
          "logr": "log_R"
          }

rename_out = deepcopy(rename)
rename_out["loga"] = "log(Age)"
rename_out["mass"] = "Mass"
rename_out["logr"] = "log(Rad)"
rename_out["logl"] = "log(L)"
rename_out["logt"] = "log(Teff)"
rename_out["logg"] = "log(g)"


class fastMISTgen(object):


    def __init__(self, mistfile=None, labels=["mini", "eep", "feh"],
                 predictions=["loga", "logl", "logt", "logg"]):

        if mistfile is None:
            self.mistfile = minesweeper.__abspath__+'data/MIST/MIST_1.2_EEPtrk.h5'
        else:
            self.mistfile = mistfile
        
        self.labels = labels
        self.predictions = predictions
        self.ndim = len(self.labels)

        self._strictness = 0.0
        self.null = np.zeros(len(self.predictions)) + np.nan

        with h5py.File(self.mistfile, "r") as misth5:
            self.make_lib(misth5)
        self.lib_as_grid()

    def make_lib(self, misth5):
        """Convert the HDF5 input to ndarrays for labels and outputs.
        """
        cols = [rename[p] for p in self.labels]
        self.libparams = np.concatenate([np.array(misth5[z])[cols] for z in misth5["index"]])
        self.libparams.dtype.names = tuple(self.labels)

        cols = [rename[p] for p in self.predictions]
        self.output = [np.concatenate([misth5[z][p] for z in misth5["index"]])
                       for p in cols]
        self.output = np.array(self.output).T

    def gemMIST(self, mini=1.0, eep=300, feh=0.0):
        """
        """
        try:
            inds, wghts = self.weights(mini=mini, eep=eep, feh=feh)
            return np.dot(wghts, self.output[inds, :])
        except(ValueError):
            return self.null

    def lib_as_grid(self):
        """Convert the library parameters to pixel indices in each dimension,
        and build and store a KDTree for the pixel coordinates.
        """
        # Get the unique gridpoints in each param
        self.gridpoints = {}
        self.binwidths = {}
        for p in self.labels:
            self.gridpoints[p] = np.unique(self.libparams[p])
            self.binwidths[p] = np.diff(self.gridpoints[p])
        # Digitize the library parameters
        X = np.array([np.digitize(self.libparams[p], bins=self.gridpoints[p],
                                  right=True) for p in self.labels])
        self.X = X.T
        # Build the KDTree
        self._kdt = KDTree(self.X)  # , metric='euclidean')

    def params_to_grid(self, **targ):
        """Convert a set of parameters to grid pixel coordinates.

        :param targ:
            The target parameter location, as keyword arguments.  The elements
            of ``labels`` must be present as keywords.

        :returns x:
            The target parameter location in pixel coordinates.
        """
        # Get bin index
        inds = np.array([np.digitize([targ[p]], bins=self.gridpoints[p], right=False) - 1
                         for p in self.labels])
        inds = np.squeeze(inds)
        # Get fractional index.
        try:
            find = [(targ[p] - self.gridpoints[p][i]) / self.binwidths[p][i]
                    for i, p in zip(inds, self.labels)]
        except(IndexError):
            pstring = "{0}: min={2} max={3} targ={1}\n"
            s = [pstring.format(p, targ[p], *self.gridpoints[p][[0, -1]])
                 for p in self.labels]
            raise ValueError("At least one parameter outside grid.\n{}".format(' '.join(s)))
        return inds + np.squeeze(find)

    def weights(self, **params):
        xtarg = self.params_to_grid(**params)
        inds = self.knearest_inds(xtarg)
        wghts = self.linear_weights(inds, xtarg)
        if wghts.sum() <= self._strictness:
            raise ValueError("Something is wrong with the weights")
        good = wghts > 0
        inds = inds[good]
        wghts = wghts[good]
        wghts /= wghts.sum()
        return inds, wghts

    def knearest_inds(self, xtarg):
        """Find all parameter ``vertices`` within a sphere of radius
        sqrt(ndim).  The parameter values are converted to pixel coordinates
        before a search of the KDTree.

        :param xtarg:
             The target location, in units of grid indices.

        :returns inds:
             The sorted indices of all vertices within sqrt(ndim) of the pixel
             coordinates, corresponding to **params.
        """
        # Query the tree within radius sqrt(ndim)
        #try:
        #    inds = self._kdt.query_radius(xtarg.reshape(1, -1),
        #                                  r=np.sqrt(self.ndim))
        #except(AttributeError):
        inds = self._kdt.query_ball_point(xtarg.reshape(1, -1),
                                          np.sqrt(self.ndim))
        return np.sort(inds[0])

    def linear_weights(self, knearest, xtarg):
        """Use ND-linear interpolation over the knearest neighbors.


        :param params:
            The target parameter location, as keyword arguments.

        :returns wght:
            The weight for each vertex, computed as the volume of the hypercube
            formed by the target parameter and each vertex.  Vertices more than
            1 away from the target in any dimension are given a weight of zero.
        """
        x = self.X[knearest, :]
        dx = xtarg - x
        # Fractional pixel weights
        wght = ((1 - dx) * (dx >= 0) + (1 + dx) * (dx < 0))
        # set weights to zero if model is more than a pixel away
        wght *= (dx > -1) * (dx < 1)
        # compute hyperarea for each model and return
        return wght.prod(axis=-1)
