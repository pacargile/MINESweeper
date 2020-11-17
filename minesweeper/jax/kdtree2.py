# Copyright Anne M. Archibald 2008
# Released under the scipy license
import jax.numpy as np
from jax import lax, jit
from jax.ops import index, index_add, index_update
from datetime import datetime

__all__ = ['minkowski_distance_p', 'minkowski_distance',
           'Rectangle', 'KDTree']


def minkowski_distance_p(x, y, p=2):
    """
    Compute the pth power of the L**p distance between two arrays.

    For efficiency, this function computes the L**p distance but does
    not extract the pth root. If `p` is 1 or infinity, this is equal to
    the actual L**p distance.

    Parameters
    ----------
    x : (M, K) array_like
        Input array.
    y : (N, K) array_like
        Input array.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.

    Examples
    --------
    >>> from scipy.spatial import minkowski_distance_p
    >>> minkowski_distance_p([[0,0],[0,0]], [[1,1],[0,1]])
    array([2, 1])

    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Find smallest common datatype with float64 (return type of this function) - addresses #10262.
    # Don't just cast to float64 for complex input case.
    common_datatype = np.promote_types(np.promote_types(x.dtype, y.dtype), 'float64')

    # Make sure x and y are NumPy arrays of correct datatype.
    x = x.astype(common_datatype)
    y = y.astype(common_datatype)

    if p == np.inf:
        return np.amax(np.abs(y-x), axis=-1)
    elif p == 1:
        return np.sum(np.abs(y-x), axis=-1)
    else:
        return np.sum(np.abs(y-x)**p, axis=-1)


def minkowski_distance(x, y, p=2):
    """
    Compute the L**p distance between two arrays.

    Parameters
    ----------
    x : (M, K) array_like
        Input array.
    y : (N, K) array_like
        Input array.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.

    Examples
    --------
    >>> from scipy.spatial import minkowski_distance
    >>> minkowski_distance([[0,0],[0,0]], [[1,1],[0,1]])
    array([ 1.41421356,  1.        ])

    """
    x = np.asarray(x)
    y = np.asarray(y)
    if p == np.inf or p == 1:
        return minkowski_distance_p(x, y, p)
    else:
        return minkowski_distance_p(x, y, p)**(1./p)


class Rectangle(object):
    """Hyperrectangle class.

    Represents a Cartesian product of intervals.
    """
    def __init__(self, maxes, mins):
        """Construct a hyperrectangle."""
        self.maxes = np.maximum(maxes,mins).astype(float)
        self.mins = np.minimum(maxes,mins).astype(float)
        self.m, = self.maxes.shape

    def __repr__(self):
        return "<Rectangle %s>" % list(zip(self.mins, self.maxes))

    def volume(self):
        """Total volume."""
        return np.prod(self.maxes-self.mins)

    def split(self, d, split):
        """
        Produce two hyperrectangles by splitting.

        In general, if you need to compute maximum and minimum
        distances to the children, it can be done more efficiently
        by updating the maximum and minimum distances to the parent.

        Parameters
        ----------
        d : int
            Axis to split hyperrectangle along.
        split : float
            Position along axis `d` to split at.

        """
        mid = self.maxes.copy()
        mid = index_update(mid,index[d],split)
        # mid[d] = split
        less = Rectangle(self.mins, mid)
        mid = self.mins.copy()
        # mid[d] = split
        mid = index_update(mid,index[d],split)
        greater = Rectangle(mid, self.maxes)
        return less, greater

    def min_distance_point(self, x, p=2.):
        """
        Return the minimum distance between input and points in the hyperrectangle.

        Parameters
        ----------
        x : array_like
            Input.
        p : float, optional
            Input.

        """
        return minkowski_distance(0, np.maximum(0,np.maximum(self.mins-x,x-self.maxes)),p)

    def max_distance_point(self, x, p=2.):
        """
        Return the maximum distance between input and points in the hyperrectangle.

        Parameters
        ----------
        x : array_like
            Input array.
        p : float, optional
            Input.

        """
        return minkowski_distance(0, np.maximum(self.maxes-x,x-self.mins),p)

    def min_distance_rectangle(self, other, p=2.):
        """
        Compute the minimum distance between points in the two hyperrectangles.

        Parameters
        ----------
        other : hyperrectangle
            Input.
        p : float
            Input.

        """
        return minkowski_distance(0, np.maximum(0,np.maximum(self.mins-other.maxes,other.mins-self.maxes)),p)

    def max_distance_rectangle(self, other, p=2.):
        """
        Compute the maximum distance between points in the two hyperrectangles.

        Parameters
        ----------
        other : hyperrectangle
            Input.
        p : float, optional
            Input.

        """
        return minkowski_distance(0, np.maximum(self.maxes-other.mins,other.maxes-self.mins),p)


class KDTree(object):
    """
    kd-tree for quick nearest-neighbor lookup

    This class provides an index into a set of k-D points which
    can be used to rapidly look up the nearest neighbors of any point.

    Parameters
    ----------
    data : (N,K) array_like
        The data points to be indexed. This array is not copied, and
        so modifying this data will result in bogus results.
    leafsize : int, optional
        The number of points at which the algorithm switches over to
        brute-force.  Has to be positive.

    Raises
    ------
    RuntimeError
        The maximum recursion limit can be exceeded for large data
        sets.  If this happens, either increase the value for the `leafsize`
        parameter or increase the recursion limit by::

            >>> import sys
            >>> sys.setrecursionlimit(10000)

    See Also
    --------
    cKDTree : Implementation of `KDTree` in Cython

    Notes
    -----
    The algorithm used is described in Maneewongvatana and Mount 1999.
    The general idea is that the kd-tree is a binary tree, each of whose
    nodes represents an axis-aligned hyperrectangle. Each node specifies
    an axis and splits the set of points based on whether their coordinate
    along that axis is greater than or less than a particular value.

    During construction, the axis and splitting point are chosen by the
    "sliding midpoint" rule, which ensures that the cells do not all
    become long and thin.

    The tree can be queried for the r closest neighbors of any given point
    (optionally returning only those within some maximum distance of the
    point). It can also be queried, with a substantial gain in efficiency,
    for the r approximate closest neighbors.

    For large dimensions (20 is already large) do not expect this to run
    significantly faster than brute force. High-dimensional nearest-neighbor
    queries are a substantial open problem in computer science.

    The tree also supports all-neighbors queries, both with arrays of points
    and with other kd-trees. These do use a reasonably efficient algorithm,
    but the kd-tree is not necessarily the best data structure for this
    sort of calculation.

    """
    def __init__(self, data, leafsize=10):
        self.data = np.asarray(data)
        # if self.data.dtype.kind == 'c':
        #     raise TypeError("KDTree does not work with complex data")

        self.n, self.m = np.shape(self.data)
        self.leafsize = int(leafsize)
        # if self.leafsize < 1:
        #     raise ValueError("leafsize must be at least 1")
        self.maxes = np.amax(self.data,axis=0)
        self.mins = np.amin(self.data,axis=0)

        print('building tree')
        self.tree = self.__build(np.arange(self.n), self.maxes, self.mins)

    class node(object):
        def __lt__(self, other):
            return id(self) < id(other)

        def __gt__(self, other):
            return id(self) > id(other)

        def __le__(self, other):
            return id(self) <= id(other)

        def __ge__(self, other):
            return id(self) >= id(other)

        def __eq__(self, other):
            return id(self) == id(other)

    class leafnode(node):
        def __init__(self, idx):
            self.idx = np.asarray(idx)
            self.children = len(idx)


    class innernode(node):
        def __init__(self, split_dim, split, less, greater):
            self.split_dim = split_dim
            self.split = split
            self.less = less
            self.greater = greater
            self.children = less.children+greater.children

    def __build(self, idx, maxes, mins):
        if len(idx) <= self.leafsize:
            return KDTree.leafnode(idx)
        else:
            data = self.data[idx]
            # maxes = np.amax(data,axis=0)
            # mins = np.amin(data,axis=0)
            d = np.argmax(maxes-mins)
            maxval = maxes[d]
            minval = mins[d]
            if maxval == minval:
                # all points are identical; warn user?
                return KDTree.leafnode(idx)
            data = data[:,d]

            # sliding midpoint rule; see Maneewongvatana and Mount 1999
            # for arguments that this is a good idea.
            split = (maxval+minval)/2
            less_idx = np.nonzero(data <= split)[0]
            greater_idx = np.nonzero(data > split)[0]
            if len(less_idx) == 0:
                split = np.amin(data)
                less_idx = np.nonzero(data <= split)[0]
                greater_idx = np.nonzero(data > split)[0]
            if len(greater_idx) == 0:
                split = np.amax(data)
                less_idx = np.nonzero(data < split)[0]
                greater_idx = np.nonzero(data >= split)[0]
            if len(less_idx) == 0:
                # _still_ zero? all must have the same value
                if not np.all(data == data[0]):
                    raise ValueError("Troublesome data array: %s" % data)
                split = data[0]
                less_idx = np.arange(len(data)-1)
                greater_idx = np.array([len(data)-1])

            # lessmaxes = maxes.copy()
            # lessmaxes = index_update(lessmaxes, index[d], split)
            lessmaxes = np.asarray([x if (ii != d) else split for ii,x in enumerate(maxes)])
            # lessmaxes[d] = split
            # greatermins = mins.copy()
            # greatermins = index_update(greatermins, index[d], split)
            greatermins = np.asarray([x if (ii != d) else split for ii,x in enumerate(mins)])
            # greatermins[d] = split
            return KDTree.innernode(d, split,
                    self.__build(idx[less_idx],lessmaxes,mins),
                    self.__build(idx[greater_idx],maxes,greatermins))

    def __query_ball_point(self, x, r, p=2., eps=0):
        R = Rectangle(self.maxes, self.mins)

        def f1(node):
            return []

        def f2(node):
            return traverse_no_checking(node)

        def f3(node):
            d = self.data[node.idx]
            cond = minkowski_distance(d, x, p) <= r
            return node.idx[cond].tolist()

        def f4(node):
            less, greater = rect.split(node.split_dim, node.split)
            return (traverse_checking(node.less, less) + 
                traverse_checking(node.greater, greater))

        def traverse_checking(node, rect):
            if rect.min_distance_point(x, p) > r / (1. + eps):
                return []
            elif rect.max_distance_point(x, p) < r * (1. + eps):
                return traverse_no_checking(node)
            elif isinstance(node, KDTree.leafnode):
                d = self.data[node.idx]
                cond = minkowski_distance(d, x, p) <= r
                return node.idx[cond].tolist()
            else:
                less, greater = rect.split(node.split_dim, node.split)
                return (traverse_checking(node.less, less) + 
                    traverse_checking(node.greater, greater))

            # return lax.cond(
            #     rect.min_distance_point(x, p) > r / (1. + eps),
            #     lambda x : f1(x),
            #     lax.cond(
            #         rect.max_distance_point(x, p) < r * (1. + eps),
            #         lambda x : f2(x),
            #         lax.cond(
            #             isinstance(node, KDTree.leafnode),
            #             lambda x : f3(x),
            #             lambda x : f4(x),
            #             operand=node
            #             ),
            #         operand=node
            #         ),
            #     operand=node
            #     )

        def traverse_no_checking(node):
            if isinstance(node, KDTree.leafnode):
                return node.idx.tolist()
            else:
                return traverse_no_checking(node.less) + \
                       traverse_no_checking(node.greater)
            # return lax.cond(
            #     isinstance(node, KDTree.leafnode),
            #     lambda x : x.idx.tolist(),
            #     lambda x : traverse_no_checking(x.less) + traverse_no_checking(x.greater),
            #     operand=node
            #     )
        return traverse_checking(self.tree, R)

    def query_ball_point(self, x, r, p=2., eps=0):
        """Find all points within distance r of point(s) x.

        Parameters
        ----------
        x : array_like, shape tuple + (self.m,)
            The point or points to search for neighbors of.
        r : positive float
            The radius of points to return.
        p : float, optional
            Which Minkowski p-norm to use.  Should be in the range [1, inf].
        eps : nonnegative float, optional
            Approximate search. Branches of the tree are not explored if their
            nearest points are further than ``r / (1 + eps)``, and branches are
            added in bulk if their furthest points are nearer than
            ``r * (1 + eps)``.

        Returns
        -------
        results : list or array of lists
            If `x` is a single point, returns a list of the indices of the
            neighbors of `x`. If `x` is an array of points, returns an object
            array of shape tuple containing lists of neighbors.

        Notes
        -----
        If you have many points whose neighbors you want to find, you may save
        substantial amounts of time by putting them in a KDTree and using
        query_ball_tree.

        Examples
        --------
        >>> from scipy import spatial
        >>> x, y = np.mgrid[0:5, 0:5]
        >>> points = np.c_[x.ravel(), y.ravel()]
        >>> tree = spatial.KDTree(points)
        >>> tree.query_ball_point([2, 0], 1)
        [5, 10, 11, 15]

        Query multiple points and plot the results:

        >>> import matplotlib.pyplot as plt
        >>> points = np.asarray(points)
        >>> plt.plot(points[:,0], points[:,1], '.')
        >>> for results in tree.query_ball_point(([2, 0], [3, 3]), 1):
        ...     nearby_points = points[results]
        ...     plt.plot(nearby_points[:,0], nearby_points[:,1], 'o')
        >>> plt.margins(0.1, 0.1)
        >>> plt.show()

        """
        x = np.asarray(x)
        # if x.dtype.kind == 'c':
        #     raise TypeError("KDTree does not work with complex data")
        # if x.shape[-1] != self.m:
        #     raise ValueError("Searching for a %d-dimensional point in a "
        #                      "%d-dimensional KDTree" % (x.shape[-1], self.m))
        # if len(x.shape) == 1:
        return self.__query_ball_point(x, r, p, eps)
        # else:
        #     retshape = x.shape[:-1]
        #     result = np.empty(retshape, dtype=object)
        #     for c in np.ndindex(retshape):
        #         result[c] = self.__query_ball_point(x[c], r, p=p, eps=eps)
        #     return result
