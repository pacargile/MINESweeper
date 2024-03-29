from copy import deepcopy
from itertools import product
import time, sys
import numpy as np
import h5py
#try:
#    from sklearn.neighbors import KDTree
#except(ImportError):
from scipy.spatial import cKDTree as KDTree

from numpy.lib import recfunctions
import minesweeper


# dictionary to translate par names to MIST names
MISTrename = {
    'log(Age)':'log_age',
    'Mass':'star_mass',
    'log(R)':'log_R',
    'log(L)':'log_L',
    'log(Teff)':'log_Teff',
    'log(g)':'log_g',
}

class GenMIST(object):


    def __init__(self, **kwargs):

        self.verbose = kwargs.get('verbose',True)

        mistfile = kwargs.get('MISTpath',None)
        if mistfile is None:
            self.mistfile = minesweeper.__abspath__+'data/MIST/MIST_2.0_EEPtrk.h5'
            # self.mistfile = 
        else:
            self.mistfile = mistfile

        if self.verbose:
            print('Using Model: {0}'.format(self.mistfile))

        # turn on age weighting
        self.ageweight = kwargs.get('ageweight',True)
        
        self.labels = kwargs.get('labels',['EEP','initial_Mass','initial_[Fe/H]','initial_[a/Fe]'])
        # list of output parametrs you want from MIST 
        # in addition to EEP, init_mass, init_FeH
        self.predictions = kwargs.get('predictions',
            ['log(Age)','Mass','log(R)','log(L)',
            'log(Teff)','[Fe/H]','[a/Fe]','log(g)'])
        if type(self.predictions) == type(None):
            self.predictions = (['log(Age)','Mass','log(R)',
                'log(L)','log(Teff)','[Fe/H]','[a/Fe]','log(g)'])
        self.ndim = len(self.labels)
        self.modpararr = self.labels+self.predictions

        if self.ageweight:
            # print('... Fitting w/ equal Age weighting')
            self.predictions.append('Agewgt')
            self.modpararr.append('Agewgt')

        self._strictness = 0.0
        self.null = np.zeros(len(self.predictions)) + np.nan

        with h5py.File(self.mistfile, "r") as misth5:
            self.make_lib(misth5)
        self.lib_as_grid()

        searchrad = 2.0
        self.dist = np.sqrt( 
            (searchrad**2.0) + 
            (searchrad**2.0) + 
            (searchrad**2.0) + 
            (searchrad**2.0))

    def make_lib(self, misth5):
        """Convert the HDF5 input to ndarrays for labels and outputs.
        """
        # cols = self.labels
        cols = ['EEP','initial_mass','initial_[Fe/H]','initial_[a/Fe]']
        self.libparams = np.concatenate([np.array(misth5[z])[cols] for z in misth5["index"]])
        self.libparams.dtype.names = tuple(self.labels)

        cols = [MISTrename[x] if x in MISTrename.keys() else x for x in self.predictions]
        self.output = [np.concatenate([misth5[z][p] for z in misth5["index"]])
                       for p in cols]
        self.output = np.array(self.output)

        self.libparams['initial_Mass']   = np.around(self.libparams['initial_Mass'],decimals=2)
        self.libparams['initial_[Fe/H]'] = np.around(self.libparams['initial_[Fe/H]'],decimals=2)
        self.libparams['initial_[a/Fe]'] = np.around(self.libparams['initial_[a/Fe]'],decimals=2)

        # if self.ageweight:
        #     if self.verbose:
        #         print('... Fitting w/ equal Age weighting')
        #     self.addagewgt()

        self.output = self.output.T

    def getMIST(self, mass=1.0, eep=300, feh=0.0, afe=0.0, **kwargs):
        """
        """
        try:
            inds, wghts = self.weights(mass=mass, eep=eep, feh=feh, afe=afe)
            predpars = np.dot(wghts, self.output[inds, :])
            return [eep,mass,feh,afe]+list(predpars)
        except(ValueError):
            return None

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
        # translate keys into MIST model names
        params['EEP'] = params.pop('eep')
        params['initial_Mass'] = params.pop('mass')
        params['initial_[Fe/H]'] = params.pop('feh')
        params['initial_[a/Fe]'] = params.pop('afe')

        xtarg = self.params_to_grid(**params)
        inds = self.knearest_inds(xtarg)
        if len(inds) == 0:
            raise ValueError
        wghts = self.linear_weights(inds, xtarg)
        if wghts.sum() <= self._strictness:
            raise ValueError("Something is wrong with the weights, sum == {}".format(wghts.sum()))
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
                                          # np.sqrt(self.ndim))
                                          self.dist)
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
