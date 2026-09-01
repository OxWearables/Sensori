import math

import numpy as np
import torch
from scipy.interpolate import CubicSpline

# The code below was adapted from
# https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data/


class Jitter(torch.nn.Module):
    "Input shape of T x 3"

    def __init__(self, sigma=0.002):
        super(Jitter, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        noise = torch.randn_like(x, dtype=x.dtype) * self.sigma
        return x + noise


class Scale(torch.nn.Module):
    """Independently scale the channels of each batch item in place."""

    def __init__(self, sigma=0.1):
        super(Scale, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        scale = x.new_empty((x.shape[0], x.shape[-1])).normal_(1.0, self.sigma)
        scale = scale.reshape((x.shape[0],) + (1,) * (x.ndim - 2) + (x.shape[-1],))
        return x.mul_(scale)


def GenerateRandomCurves(X, sigma=0.001, knot=2):
    xx = (np.ones((X.shape[1], 1)) * (np.arange(0, X.shape[0], (X.shape[0] - 1) / (knot + 1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:, 0], yy[:, 0])
    cs_y = CubicSpline(xx[:, 1], yy[:, 1])
    cs_z = CubicSpline(xx[:, 2], yy[:, 2])
    return torch.as_tensor(np.array([cs_x(x_range), cs_y(x_range), cs_z(x_range)]).transpose(), dtype=X.dtype)


class MagnitudeWarping(torch.nn.Module):
    "Input shape of T x 3"

    def __init__(self, sigma=0.001, knot=2):
        super(MagnitudeWarping, self).__init__()
        self.sigma = sigma
        self.knot = knot

    def forward(self, x):
        random_curve = GenerateRandomCurves(x, self.sigma, self.knot)
        return x * random_curve


class ChannelSwap(torch.nn.Module):
    """Independently permute the channels of each batch item."""

    def __init__(self):
        super(ChannelSwap, self).__init__()

    def forward(self, x):
        permutations = torch.rand((x.shape[0], x.shape[-1]), device=x.device).argsort(dim=-1)
        index_shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (x.shape[-1],)
        return torch.gather(x, -1, permutations.reshape(index_shape).expand_as(x))


def axangle2mat(axis, angle, is_normalized=False):
    """Rotation matrix for rotation angle `angle` around `axis`

    Parameters
    ----------
    axis : 3 element sequence
       vector specifying axis for rotation.
    angle : scalar
       angle of rotation in radians.
    is_normalized : bool, optional
       True if `axis` is already normalized (has norm of 1).  Default False.

    Returns
    -------
    mat : array shape (3,3)
       rotation matrix for specified rotation

    Notes
    -----
    From: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    """
    x, y, z = axis
    if not is_normalized:
        n = math.sqrt(x * x + y * y + z * z)
        x = x / n
        y = y / n
        z = z / n
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1 - c
    xs = x * s
    ys = y * s
    zs = z * s
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC
    return np.array(
        [[x * xC + c, xyC - zs, zxC + ys], [xyC + zs, y * yC + c, yzC - xs], [zxC - ys, yzC + xs, z * zC + c]]
    )


class Rotation(torch.nn.Module):
    "Input shape of T x F x 3"

    def __init__(self):
        super(Rotation, self).__init__()

    def forward(self, x):
        axis = np.random.uniform(low=-1, high=1, size=x.shape[-1])
        angle = np.random.uniform(low=-np.pi, high=np.pi)

        my_sample = x.reshape(-1, 3)
        my_sample = my_sample @ torch.as_tensor(axangle2mat(axis, angle), dtype=x.dtype)
        return my_sample.reshape(x.shape)


class Time_shift(torch.nn.Module):
    "Input shape of B x F x 3"

    def __init__(self, num_shifts=60):
        super(Time_shift, self).__init__()
        self.num_shifts = num_shifts

    def forward(self, x):
        shift = np.random.randint(-self.num_shifts, self.num_shifts)
        return torch.roll(x, shift, dims=0)


class RandomCrop(torch.nn.Module):
    "Input shape of T x 3"

    def __init__(self, crop_ratio=0.5):
        """
        Randomly crop a contiguous segment from the time series.
        Args:
            crop_ratio: fraction of the sequence to keep (e.g., 0.5 for 12 hours from 24 hours)
        """
        super(RandomCrop, self).__init__()
        self.crop_ratio = crop_ratio

    def forward(self, x):
        T = x.shape[0]
        crop_len = int(T * self.crop_ratio)

        # Randomly select start position for the crop
        max_start = T - crop_len
        if max_start <= 0:
            return x

        start_idx = np.random.randint(0, max_start + 1)
        return x[start_idx : start_idx + crop_len]


class Time_mask(torch.nn.Module):
    "Input shape of B x F x 3"

    def __init__(self, mask_segment_len=30, masking_ratio=0.1):
        """
        mask_segment_len: number of 30 seconds frames to be masked for each consecutive masking segment
        masking_ratio: percentage of the whole time series to be selected for as masking starting points

        """
        super(Time_mask, self).__init__()
        self.mask_segment_len = mask_segment_len
        self.masking_ratio = masking_ratio

    def forward(self, x):
        T = x.shape[0]

        assert T > self.mask_segment_len, 'Time series is shorter than mask segment length'

        # sample_mean = np.mean(x, axis=0)
        # my_sample = x.copy()
        sample_mean = x.mean(dim=0)
        my_sample = x.clone()
        my_mask_num = int(np.random.normal(loc=T * self.masking_ratio, scale=5))

        for _ in range(my_mask_num):
            start_idx = np.random.randint(0, T - self.mask_segment_len)
            segment_len = int(np.random.normal(loc=self.mask_segment_len, scale=5))
            segment_len = min(segment_len, T - start_idx)
            my_sample[start_idx : start_idx + segment_len, :] = sample_mean

        return my_sample
