from __future__ import absolute_import


import numbers

import numpy as np

from scipy import special
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array
from scipy.stats import entropy


def _check_parameters(min_samples_split, min_samples_leaf, max_candidates):
    if (not isinstance(min_samples_split, numbers.Integral) or
            min_samples_split < 2):
        raise ValueError("min_samples_split must be a positive integer >= 2; "
                         "got {}.".format(min_samples_split))

    if (not isinstance(min_samples_leaf, numbers.Integral) or
            min_samples_leaf < 1):
        raise ValueError("min_samples_leaf must be a positive integer >= 1; "
                         "got {}.".format(min_samples_leaf))

    if not isinstance(max_candidates, numbers.Integral) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer >= 1; "
                         "got {}.".format(max_candidates))
        

from math import floor, log10

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import (
    check_array,
    check_X_y,
    column_or_1d,
    check_random_state,
)
from sklearn.utils.validation import check_is_fitted
# from mdlp._mdlp import MDLPDiscretize


# def normalize(cut_points, _range, precision):
#     # if len(cut_points) == 0:
#         # return cut_points
#     # _range = np.max(col) - np.min(col)
#     multiplier = 10**(-floor(log10(_range))) / precision
#     return (cut_points * multiplier).astype(np.int) / multiplier


class MDLP2(BaseEstimator, TransformerMixin):
    """Bins continuous values using MDLP "expert binning" method.

    Implements the MDLP discretization algorithm from Usama Fayyad's
    paper "Multi-Interval Discretization of Continuous-Valued
    Attributes for Classification Learning". Given the class labels
    for each sample, this transformer attempts to discretize a
    continuous attribute by minimizing the entropy at each interval.

    Parameters
    ----------
    continuous_features : 
        - None (default): All features are treated as continuous for discretization.
        - array of indices: Array of continous feature indices.
        - mask: Array of length n_features and with dtype=bool.

        If `X` is a 1-D array, then continuous_features is neglected.

    min_depth : int (default=0)
        The minimum depth of the interval splitting. Overrides
        the MDLP stopping criterion. If the entropy at a given interval
        is found to be zero before `min_depth`, the algorithm will stop.

    random_state : int (default=None)
        Seed of pseudo RNG to use when shuffling the data. Affects the
        outcome of MDLP if there are multiple samples with the same
        continuous value, but with different class labels.

    min_split : float (default=1e-3)
        The minmum size to split a bin

    dtype : np.dtype (default=np.int)
        The dtype of the transformed X

    Attributes
    ----------
    continuous_features_ : array-like of type int
        Similar to continous_features. However, for 2-D arrays, this
        attribute cannot be None.

    cut_points_ : dict of type {int : np.array}
        Dictionary mapping indices to a numpy array. Each
        numpy array is a sorted list of cut points found from
        discretization.

    dimensions_ : int
        Number of dimensions to input `X`. Either 1 or 2.

    Examples
    --------
    ```
        >>> from mdlp.discretization import MDLP
        >>> from sklearn.datasets import load_iris
        >>> iris = load_iris()
        >>> X = iris.data
        >>> y = iris.target
        >>> mdlp = MDLP()
        >>> conv_X = mdlp.fit_transform(X, y)

        `conv_X` will be the same shape as `X`, except it will contain
        integers instead of continuous attributes representing the results
        of the discretization process.

        To retrieve the explicit intervals of the discretization of, say,
        the third column (index 2), one can do

        >>> mdlp.cat2intervals(conv_X, 2)

        which would return a list of tuples `(a, b)`. Each tuple represents
        the contnuous interval (a, b], where `a` can be `float("-inf")`,
        and `b` can be `float("inf")`.
    ```
    """

    def __init__(self, continuous_features=None, dtype=np.int, min_samples_split=1000, min_samples_leaf=1000,
                 max_candidates=32, random_state=2021):
        # Parameters
        # self.continuous_features = None
#         self.min_depth = min_depth
        self.random_state = random_state
#         self.min_split = min_split
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_candidates = max_candidates
        self.continuous_features = continuous_features
        self.dtype = dtype

        # Attributes
        self.continuous_features_ = None
        self.cut_points_ = None
        self.mins_ = None
        self.maxs_ = None

    def fit(self, X, y):
        """Finds the intervals of interest from the input data.

        Parameters
        ----------
        X : The array containing features to be discretized. Continuous
            features should be specified by the `continuous_features`
            attribute if `X` is a 2-D array.

        y : A list or array of class labels corresponding to `X`.

        continuous_features : (default None) a list of indices that you want to discretize
                              or a list (or array) of bools indicating the continuous features
        """
        X = check_array(X, force_all_finite=True, ensure_2d=True, dtype=np.float64)
        y = column_or_1d(y)
        y = check_array(y, ensure_2d=False, dtype=np.int)
        X, y = check_X_y(X, y)

        if len(X.shape) != 2:
            raise ValueError("Invalid input dimension for `X`. "
                             "Input shape is expected to be 2D, but is {0}".format(X.shape))

        state = check_random_state(self.random_state)
        perm = state.permutation(len(y))
        X = X[perm]
        y = y[perm]

        if self.continuous_features is None:
            self.continuous_features_ = np.arange(X.shape[1])
        else:
            continuous_features = np.array(self.continuous_features)
            if continuous_features.dtype == np.bool:
                continuous_features = np.arange(len(continuous_features))[continuous_features]
            else:
                continuous_features = continuous_features.astype(np.int, casting='safe')
                assert np.max(continuous_features) < X.shape[1] and np.min(continuous_features) >= 0
            self.continuous_features_ = continuous_features

        self.cut_points_ = [None] * X.shape[1]
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)

        for index in self.continuous_features_:
            col = X[:, index]
            est = MDLPer(self.min_samples_leaf,
                self.min_samples_split,
                self.max_candidates)
            est.fit(col, y)
#             cut_points = MDLPDiscretize(col, y, self.min_depth, self.min_split)
            self.cut_points_[index] = est.splits
            # self.cut_points_[index] = normalize(cut_points, maxs[index] - mins[index], self.precision)

        self.mins_ = mins
        self.maxs_ = maxs
        return self

    def transform(self, X):
        """Discretizes values in X into {0, ..., k-1}.

        `k` is the number of bins the discretizer creates from a continuous
        feature.
        """
        X = check_array(X, force_all_finite=True, ensure_2d=False)
        check_is_fitted(self, "cut_points_")

        output = X.copy()
        for i in self.continuous_features_:
            output[:, i] = np.searchsorted(self.cut_points_[i], X[:, i])
        return output.astype(self.dtype)

    def cat2intervals(self, X, index):
        """Converts categorical data into intervals.

        Parameters
        ----------
        X : The discretized array

        index: which feature index to convert
        """

        cp_indices = X[:, index]
        return self.assign_intervals(cp_indices, index)

    def cts2cat(self, col, index):
        """Converts each continuous feature from index `index` into
        a categorical feature from the input column `col`.
        """
        return np.searchsorted(self.cut_points_[index], col)

    def assign_intervals(self, cp_indices, index):
        """Assigns the cut point indices `cp_indices` (representing
        categorical features) into a list of intervals.
        """

        # Case for a 1-D array
        cut_points = self.cut_points_[index]
        if cut_points is None:
            raise ValueError("The given index %d has not been discretized!")
        non_zero_mask = cp_indices[cp_indices - 1 != -1].astype(int) - 1
        fronts = np.zeros(cp_indices.shape)
        fronts[cp_indices == 0] = float("-inf")
        fronts[cp_indices != 0] = cut_points[non_zero_mask]

        n_cuts = len(cut_points)
        backs = np.zeros(cp_indices.shape)
        non_n_cuts_mask = cp_indices[cp_indices != n_cuts].astype(int)
        backs[cp_indices == n_cuts] = float("inf")
        backs[cp_indices != n_cuts] = cut_points[non_n_cuts_mask]

        return [(front, back) for front, back in zip(fronts, backs)]

    # def



class MDLPer(BaseEstimator):
    """
    Minimum Description Length Principle (MDLP) discretization algorithm.
    Parameters
    ----------
    min_samples_split : int (default=2)
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int (default=2)
        The minimum number of samples required to be at a leaf node.
    max_candidates : int (default=32)
        The maximum number of split points to evaluate at each partition.
    Notes
    -----
    Implementation of the discretization algorithm in [FI93]. A dynamic
    split strategy based on binning the number of candidate splits [CMR2001]
    is implemented to increase efficiency. For large size datasets, it is
    recommended to use a smaller ``max_candidates`` (e.g. 16) to get a
    significant speed up.
    References
    ----------
    .. [FI93] U. M. Fayyad and K. B. Irani. "Multi-Interval Discretization of
              Continuous-Valued Attributes for Classification Learning".
              International Joint Conferences on Artificial Intelligence,
              13:1022–1027, 1993.
    .. [CMR2001] D. M. Chickering, C. Meek and R. Rounthwaite. "Efficient
                 Determination of Dynamic Split Points in a Decision Tree". In
                 Proceedings of the 2001 IEEE International Conference on Data
                 Mining, 91-98, 2001.
    """
    def __init__(self, min_samples_split=2, min_samples_leaf=2,
                 max_candidates=32):

        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_candidates = max_candidates

        # auxiliary
        self._splits = []

        self._is_fitted = None

    def fit(self, x, y):
        """Fit MDLP discretization algorithm.
        Parameters
        ----------
        x : array-like, shape = (n_samples)
            Data samples, where n_samples is the number of samples.
        y : array-like, shape = (n_samples)
            Target vector relative to x.
        Returns
        -------
        self : object
        """
        return self._fit(x, y)

    def _fit(self, x, y):
        _check_parameters(**self.get_params())

        x = check_array(x, ensure_2d=False, force_all_finite=True)
        y = check_array(y, ensure_2d=False, force_all_finite=True)

        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

        self._recurse(x, y, 0)

        self._is_fitted = True

        return self

    def _recurse(self, x, y, id):
        u_x = np.unique(x)
        n_x = len(u_x)
        n_y = len(np.bincount(y))

        split = self._find_split(u_x, x, y)

        if split is not None:
            self._splits.append(split)
            t = np.searchsorted(x, split, side="right")

            if not self._terminate(n_x, n_y, y, y[:t], y[t:]):
                self._recurse(x[:t], y[:t], id + 1)
                self._recurse(x[t:], y[t:], id + 2)

    def _find_split(self, u_x, x, y):
        n_x = len(x)
        u_x = np.unique(0.5 * (x[1:] + x[:-1])[(y[1:] - y[:-1]) != 0])

        if len(u_x) > self.max_candidates:
            percentiles = np.linspace(1, 100, self.max_candidates)
            splits = np.percentile(u_x, percentiles)
        else:
            splits = u_x

        max_entropy_gain = 0
        best_split = None

        tt = np.searchsorted(x, splits, side="right")
        for i, t in enumerate(tt):
            samples_l = t >= self.min_samples_leaf
            samples_r = n_x - t >= self.min_samples_leaf

            if samples_l and samples_r:
                entropy_gain = self._entropy_gain(y, y[:t], y[t:])
                if entropy_gain > max_entropy_gain:
                    max_entropy_gain = entropy_gain
                    best_split = splits[i]

        return best_split

    def _entropy(self, x):
        n = len(x)
#         ns1 = np.sum(x)
#         ns0 = n - ns1
#         p = np.array([ns0, ns1]) / n
        counts = np.bincount(x)
        vals = np.true_divide(counts, n)
        return entropy(vals)

    def _entropy_gain(self, y, y1, y2):
        n = len(y)
        n1 = len(y1)
        n2 = n - n1
        ent_y = self._entropy(y)
        ent_y1 = self._entropy(y1)
        ent_y2 = self._entropy(y2)
        return ent_y - (n1 * ent_y1 + n2 * ent_y2) / n

    def _terminate(self, n_x, n_y, y, y1, y2):
        splittable = (n_x >= self.min_samples_split) and (n_y >= 2)

        n = len(y)
        n1 = len(y1)
        n2 = n - n1
        ent_y = self._entropy(y)
        ent_y1 = self._entropy(y1)
        ent_y2 = self._entropy(y2)
        gain = ent_y - (n1 * ent_y1 + n2 * ent_y2) / n

        k = len(np.bincount(y))
        k1 = len(np.bincount(y1))
        k2 = len(np.bincount(y2))

        t0 = np.log(3**k - 2)
        t1 = k * ent_y
        t2 = k1 * ent_y1
        t3 = k2 * ent_y2
        delta = t0 - (t1 - t2 - t3)

        return gain <= (np.log(n - 1) + delta) / n or not splittable

    @property
    def splits(self):
        """List of split points
        Returns
        -------
        splits : numpy.ndarray
        """
        if not self._is_fitted:
            raise NotFittedError("This {} instance is not fitted yet. Call "
                                 "'fit' with appropriate arguments."
                                 .format(self.__class__.__name__))

        return np.sort(self._splits)