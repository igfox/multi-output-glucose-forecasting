import numpy as np
from numpy.polynomial import polynomial as pn
import joblib
from joblib import Parallel, delayed
from torch.utils.data import Dataset


class GlucoseDataset(Dataset):
    """
    Blood glucose dataset for pytorch
    Entry: (data, y_bin, y_real, len)
    loading everything into memory as small
    """
    def __init__(self,
                 data_pkl,
                 max_pad,
                 output_len,
                 output_dim,
                 polynomial=False,
                 degree=None,
                 range_low=None,
                 range_high=None,
                 coeff_file=None,
                 real_values=False,
                 parallel_cache=False,
                 max_size=None,
                 flip_signal=False):

        # for one-hot encoding, assumes 40-400 mg/dL range
        self.output_dim = output_dim
        self.polynomial = polynomial
        self.degree = degree
        self.range_low = range_low
        self.range_high = range_high
        self.real_values = real_values
        self.max_pad = max_pad

        self.data = joblib.load(data_pkl)
        if flip_signal:
            self.data_flip = []
            for i in range(len(self.data)):
                self.data_flip.append(np.flip(self.data[i], axis=0))
            self.data = self.data_flip
        if max_size is not None:
            self.data = self.data[0:max_size]
        self.output_len = output_len

        if not self.real_values:
            if self.polynomial:
                # from degree, calculate ranges
                # from ranges, get bins
                # should make more flexible, function parameter binning
                # old hand-defined range, captured 100% variation
                #[[40, 400],
                # [-36, 36],
                # [-5.5, 5.5]]
                for var in [self.degree, self.range_low, self.range_high]:
                    assert var is not None, 'Must set degree, range_high, and range_low for polynomial'

                if coeff_file is None:
                    ranges = self.auto_poly_range()
                else:
                    ranges = self.precomputed_poly_range(coeff_file)

                self.bin_step = [(ranges[i][1]-ranges[i][0])/(self.output_dim-1) for i in range(degree+1)]
                self.bins = [
                    np.linspace(ranges[i][0],
                                ranges[i][1],
                                self.output_dim) + (0.5 * self.bin_step[i])
                    for i in range(self.degree+1)]
            else:
                # simple value binning
                self.bin_step = (400-40)/(self.output_dim-1)
                # the half step appraoch is an artifact of wanting perfect bins with output_dim=361
                self.bins = np.linspace(40, 400, self.output_dim)+(self.bin_step * 0.5)

        # trying out precaching results for less intensive load
        count = 0
        self.x_out = []
        self.y_out = []
        self.y_real = []
        self.lens = []
        print('caching results')
        if parallel_cache:
            res_tuples = Parallel(n_jobs=5, verbose=10)(delayed(self.prepare_output)(idx) for idx in range(len(self.data)))
            for idx in range(len(self.data)):
                x_pad, y_pad, y_real_pad, lens = res_tuples[idx]
                self.x_out.append(x_pad)
                self.y_out.append(y_pad)
                self.y_real.append(y_real_pad)
                self.lens.append(lens)
        else:
            for idx in range(len(self.data)):
                if idx % 1000 == 0:
                    print('{}/{}'.format(idx, len(self.data)))
                x_pad, y_pad, y_real_pad, lens = self.prepare_output(idx)
                self.x_out.append(x_pad)
                self.y_out.append(y_pad)
                self.y_real.append(y_real_pad)
                self.lens.append(lens)

    def prepare_output(self, idx, real_y=True):
        x_dat = self.data[idx]
        length = self.max_pad - len(x_dat)
        x_pad = np.pad(x_dat,
                       (0, length),
                       mode='constant',
                       constant_values=-1)
        y_dat = self.window_stack(x_dat[1::].reshape(-1, 1))
        if self.real_values:
            y_bins = y_dat
        else:
            y_bins = self.values_to_bins(y_dat)
        y_pad = np.pad(y_bins,
                       ((0, length), (0, 0)),
                       mode='constant',
                       constant_values=-1)
        if real_y:
            y_real_pad = np.pad(y_dat,
                                ((0, length), (0, 0)),
                                mode='constant',
                                constant_values=-1)
            return x_pad, y_pad, y_real_pad, self.max_pad - length
        else:
            return x_pad, y_pad, self.max_pad - length

    def auto_poly_range(self, percentile):
        """
        Using degree and training data, creates
        range that captures percentile% of variation of the best fit
        coefficient values.
        """
        raise NotImplementedError('TODO')

    def precomputed_poly_range(self, coeff_file):
        """
        Simple function that uses precomputed coefficient
        percentile dict

        low, high can be integers in 0-100

        Requires precomputed coeff dict
        """
        assert self.range_low < self.range_high

        coeff = joblib.load(coeff_file)

        ranges = []
        for i in range(self.degree+1):
            low_val = coeff[self.degree][i][self.range_low]
            high_val = coeff[self.degree][i][self.range_high]
            ranges.append([low_val, high_val])
        return ranges

    def scale(self, x):
        """
        turn glucose signal with 40-400 to range -1 to 1
        can add more intelligent scaling for balencing hypo/hyper,
        though real concern is moving over to classification
        """
        return (x-220)/180.

    def one_hot(self, seq):
        """
        turn glucose signal into one hot distribution
        with size=output_dim, linearly bins glucose
        range 40-400
        don't need for NLLLoss
        """
        dist = np.zeros((seq.size, self.output_dim))
        dist[np.arange(seq.size), np.digitize(seq, self.bins)] = 1.
        return dist

    def polymerize(self, y):
        """
        Turns output window into best fit polynomial
        with output [x'_0, ..., x'_d] where x' is
        bin number that x would be in (using ranges)
        """
        x_inds = []
        if len(y.shape) > 1:
            for j in range(y.shape[0]):
                coeffs = pn.polyfit(np.arange(len(y[j])), y[j], deg=self.degree)
                x_inds.append([np.digitize(coeffs[i], self.bins[i]).item() for i in range(self.degree+1)])
        else:
            coeffs = pn.polyfit(np.arange(len(y)), y, deg=self.degree)
            for i in range(self.degree+1):
                x_inds.append(np.digitize(coeffs[i], self.bins[i]).item())
        return np.clip(x_inds, 0, self.output_dim-1)

    def bins_to_coeff_values(self, pred):
        """
        Given bins for polynomial coefficients,
        return estimate of real coefficient values
        """
        if len(pred.shape) > 1:
            vals = [np.array([self.bins[i][np.clip(np.array(pred[:, i], dtype=int), 0, self.output_dim-1)]])
                    - (0.5 * self.bin_step[i]) for i in range(self.degree+1)]
            coeffs = np.concatenate(vals, axis=0).T
        else:
            coeffs = [self.bins[i][np.clip(np.array(pred[i], dtype=int), 0, self.output_dim-1)]
                      - (0.5 * self.bin_step[i]) for i in range(self.degree+1)]
        return np.array(coeffs)

    def reverse_polymerize(self, pred):
        """
        Given bins for polynomial coefficients, returns forecast
        For new foreacsting system, flexible degree and doesn't assume
        adding mistake
        """
        coeffs = self.bins_to_coeff_values(pred)
        return pn.polyval(np.arange(self.output_len), coeffs.T)

    def values_to_bins(self, y):
        """
        Gvien a y sample (or batch of y samples), changes from
        value to categorical representation
        """
        if self.real_values:
            return y
        if self.polynomial:
            return self.polymerize(y)
        else:
            return np.digitize(y, self.bins)

    def bins_to_values(self, y):
        """
        Given a y sample (or batch of y samples), changes from categorical
        to value representation
        """
        if type(y) is not np.ndarray:
            y = y.numpy()
        if self.real_values:
            return y
        if self.polynomial:
            return self.reverse_polymerize(y)
        else:
            vals = self.bins[np.clip(np.array(y, dtype=int), 0, self.output_dim-1)]
            return vals - (0.5 * self.bin_step)

    def index_to_values(self, x, i):
        """
        Given i index for output value: y[i]
        returns ground truth x values
        bins_to_values can also be used, but ignores
        polynomial residual
        """
        return x[i+1:i+1+self.output_len]

    def window_stack(self, seq, stepsize=1):
        """
        Gets rolling window from seq of length self.output_len
        stepsize determines dilation
        """
        length = self.output_len
        n = seq.shape[0]
        return np.hstack(seq[i:1+n+i-length:stepsize] for i in range(length))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.x_out[idx], self.y_out[idx], self.y_real[idx], self.lens[idx]
