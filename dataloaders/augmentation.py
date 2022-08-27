import numpy as np
import pandas as pd
from typing import List

# x1: all channel values first sample
# x2: all channel values second sample
# lambda: mixup parameter
# TODO: sample lambda value from beta distribution # https://arxiv.org/abs/1710.09412
# returns x: all channel values mixed up
# completed
def mixup(x1: np.ndarray, x2: np.ndarray, _lambda: float=0.5):
    x = _lambda * x1 + (1 - _lambda) * x2
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,3))
    plt.plot(x1[:, 0], label='acc_x', c = 'green', linewidth=2)
    plt.plot(x2[:, 0], c = 'blue', label='other_acc_x', linewidth=2)
    plt.plot(x[:, 0], c = 'red', linestyle='dashed', label='mixup_acc_x', linewidth=2)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    return x

# TODO https://mohcinemadkour.github.io/posts/2019/10/Machine%20Learning,%20timeseriesAI,%20Time%20Series%20Classification,%20fastai_timeseries,%20data%20augmentation/
# https://arxiv.org/abs/1708.04552
def cutout():
    pass

# TODO https://arxiv.org/abs/1905.04899
def cutmix():
    pass

class RandomAugment(object):
    def __init__(self, transformation_count, p):
        self.tranformation_count = transformation_count
        self.all_transformations = [
            self.jitter,
            self.exponential_smoothing,
            self.moving_average,
            self.magnitude_scaling,
            self.magnitude_warp,
            self.magnitude_shift,
            self.time_warp,
            self.window_warp,
            self.window_slice,
            self.random_sampling,
            self.slope_adding,
        ]
        self.p = p

    def __call__(self, org_sample_x):
        # if used with torch, use torch's rng
        sample_x = org_sample_x.copy()
        transformations = np.random.choice(
            self.all_transformations, size=self.tranformation_count, replace=False
        )
        used = []
        for t in transformations:
            # print(t)
            # print(self.p[t.__name__])
            if self.p[t.__name__] > 0.5:
                sample_x = t(sample_x)
                used.append(t)
        return (np.asarray([sample_x]), used)

    # x: all channel values
    # sigma: determines the strength of noise
    # scale = sigma * variance of the channel might be a good idea
    # original x: single channel values as numpy.ndarray
    # [-0.0957651  -0.08101412 -0.04705619 -0.00337204  0.01070699]
    # completed
    @staticmethod
    def jitter(x: np.ndarray, sigma: float = 0.1):
        channel_variance = np.var(x, axis=0)
        result = np.array(
            [
                x[:, idx]
                + np.random.normal(
                    loc=0.0, scale=sigma * channel_variance[idx], size=x.shape[0]
                )
                for idx in range(x.shape[1])
            ]
        ).transpose()
        return result

    # x: all channel values
    # alpha: smoothing parameter
    # original x: single channel values as numpy.ndarray of single numpy.ndarray
    # [[-0.0957651  -0.08101412 -0.04705619 -0.00337204  0.01070699]]
    # completed
    @staticmethod
    def exponential_smoothing(x: np.ndarray, alpha: float = 0.5):
        x = x.transpose()
        length = x.shape[1]
        ret = np.zeros_like(x)
        for dim in range(x.shape[0]):
            ret[dim, 0] = x[dim, 0]
            for index in range(1, length):
                ret[dim, index] = (
                    alpha * x[dim, index] + (1 - alpha) * ret[dim, index - 1]
                )
        return ret.transpose()

    # x: all channel values
    # original x: single channel values as numpy.ndarray of single numpy.ndarray
    # [[-0.0957651  -0.08101412 -0.04705619 -0.00337204  0.01070699]]
    # completed
    @staticmethod
    def moving_average(x: np.ndarray):
        df = pd.DataFrame(x)
        ret = df.rolling(window=5).mean()
        ret = ret.interpolate(method="linear", limit_direction="both")
        return ret.values

    # x: all channel values
    # original x: single channel values as numpy.ndarray of single numpy.ndarray
    # [[-0.0957651  -0.08101412 -0.04705619 -0.00337204  0.01070699]]
    # completed
    @staticmethod
    def magnitude_scaling(x: np.ndarray, sigma: float = 0.5):
        # https://arxiv.org/pdf/1706.00527.pdf
        factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[1]))
        factor = np.ones_like(factor) * 0.7
        return np.multiply(x, factor[np.newaxis, :])

    # x: all channel values
    # original x: single channel values as numpy.ndarray of multiple numpy.ndarrays with single value
    # [[-0.0957651 ] [-0.08101412] [-0.04705619] [-0.00337204] [ 0.01070699]]
    # completed
    @staticmethod
    def magnitude_warp(x: np.ndarray, sigma: float = 0.4, knot: int = 6):
        from scipy.interpolate import CubicSpline

        orig_steps = np.arange(x.shape[0])
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot, x.shape[1]))
        warp_steps = (
            np.ones((x.shape[1], 1)) * (np.linspace(0, x.shape[0] - 1.0, num=knot))
        ).T
        warper = np.array(
            [
                CubicSpline(warp_steps[:, dim], random_warps[:, dim])(orig_steps)
                for dim in range(x.shape[1])
            ]
        ).T
        ret = x * warper
        return ret

    # x: all channel values
    # original x: single channel values as numpy.ndarray of single numpy.ndarray
    # [[-0.0957651  -0.08101412 -0.04705619 -0.00337204  0.01070699]]
    # completed
    @staticmethod
    def magnitude_shift(x: np.ndarray, ratio: float = 0.2):
        x = x.transpose()
        ret = np.zeros_like(x)
        for dim in range(x.shape[0]):
            # print(x[dim,:].mean())
            ret[dim, :] = x[dim, :] + max(
                np.mean(x[dim, :]) * ratio, np.std(x[dim, :]) * ratio
            )
        return ret.transpose()

    # x : all channel values
    # original x: single channel values as numpy.ndarray of multiple numpy.ndarrays with single value
    # [[-0.0957651 ] [-0.08101412] [-0.04705619] [-0.00337204] [ 0.01070699]]
    # completed
    @staticmethod
    def time_warp(x: np.ndarray, sigma: float = 0.05, knot: int = 5):
        from scipy.interpolate import CubicSpline

        orig_steps = np.arange(x.shape[0])

        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2))
        warp_steps = (
            np.ones((x.shape[1], 1)) * (np.linspace(0, x.shape[0] - 1.0, num=knot + 2))
        ).T

        ret = np.zeros_like(x)

        for dim in range(x.shape[1]):
            time_warp = CubicSpline(
                warp_steps[:, dim], warp_steps[:, dim] * random_warps
            )(orig_steps)
            # plt.figure()
            # plt.plot(time_warp)
            scale = (x.shape[0] - 1) / time_warp[-1]
            ret[:, dim] = np.interp(
                orig_steps, np.clip(scale * time_warp, 0, x.shape[0] - 1), x[:, dim]
            ).T
        return ret

    # x : all channel values
    # original x: single channel values as numpy.ndarray of single numpy.ndarray
    # [[-0.0957651  -0.08101412 -0.04705619 -0.00337204  0.01070699]]
    # completed
    @staticmethod
    def window_warp(
        x: np.ndarray, window_ratio: float = 0.15, scales: List[float] = [0.5, 2.0]
    ):
        # https://halshs.archives-ouvertes.fr/halshs-01357973/document
        x = x.transpose()
        warp_scales = np.random.choice(scales, 1)
        warp_size = np.ceil(window_ratio * x.shape[1]).astype(int)
        window_steps = np.arange(warp_size)
        window_starts = np.random.randint(
            low=1, high=x.shape[1] - warp_size - 1, size=1
        ).astype(int)
        window_ends = (window_starts + warp_size).astype(int)
        ret = np.zeros_like(x)

        for dim in range(x.shape[0]):
            start_seg = x[dim, : window_starts[0]]
            window_seg = np.interp(
                np.linspace(0, warp_size - 1, num=int(warp_size * warp_scales[0])),
                window_steps,
                x[dim, window_starts[0] : window_ends[0]],
            )
            end_seg = x[dim, window_ends[0] :]

            warped = np.concatenate((start_seg, window_seg, end_seg))
            ret[dim, :] = np.interp(
                np.arange(x.shape[1]),
                np.linspace(0, x.shape[1] - 1.0, num=warped.size),
                warped,
            ).T
        return ret.transpose()

    # x : all channel values
    # original x: single channel values as numpy.ndarray of single numpy.ndarray
    # [[-0.0957651  -0.08101412 -0.04705619 -0.00337204  0.01070699]]
    # completed
    @staticmethod
    def window_slice(x: np.ndarray, reduce_ratio: float = 0.9):
        x = x.transpose()
        # https://halshs.archives-ouvertes.fr/halshs-01357973/document
        target_len = np.ceil(reduce_ratio * x.shape[1]).astype(int)
        if target_len >= x.shape[1]:
            return x
        starts = np.random.randint(
            low=0, high=x.shape[1] - target_len, size=(1)
        ).astype(int)
        ends = (target_len + starts).astype(int)
        # print(starts)
        # print(ends)
        ret = np.zeros_like(x)

        for dim in range(x.shape[0]):
            ret[dim, :] = np.interp(
                np.linspace(0, target_len, num=x.shape[1]),
                np.arange(target_len),
                x[dim, starts[0] : ends[0]],
            ).T
        return ret.transpose()

    # x : all channel values
    # original x: single channel values as numpy.ndarray of single numpy.ndarray
    # [[-0.0957651  -0.08101412 -0.04705619 -0.00337204  0.01070699]]
    # completed
    @staticmethod
    def random_sampling(x: np.ndarray, reduce_ratio: float = 0.8):
        import random

        x = x.transpose()
        target_len = np.ceil(reduce_ratio * x.shape[1]).astype(int)
        if target_len >= x.shape[1]:
            return x

        index_list = list(np.arange(x.shape[1]))
        sampled_index = random.sample(index_list, target_len)
        sampled_index = sorted(sampled_index)
        # print(sampled_index)

        ret = np.zeros_like(x)

        for dim in range(x.shape[0]):
            ret[dim, :] = np.interp(
                np.linspace(0, target_len, num=x.shape[1]),
                np.arange(target_len),
                x[dim, sampled_index],
            ).T
        return ret.transpose()

    # x : all channel values
    # original x: single channel values as numpy.ndarray of single numpy.ndarray
    # [[-0.0957651  -0.08101412 -0.04705619 -0.00337204  0.01070699]]
    # completed
    @staticmethod
    def slope_adding(x: np.ndarray, slope: float = 0.3):
        import random

        x = x.transpose()
        anchor = random.randint(0, x.shape[1])
        # print(anchor)
        anchor = 80
        ret = np.zeros_like(x)
        for dim in range(x.shape[0]):
            slope = slope / x.shape[1]
            slope = np.linspace(0, x.shape[1] - 1, x.shape[1]) * slope
            shift = slope[anchor]
            slope = slope - shift

            ret[dim, :] = x[dim, :] + slope
        return ret.transpose()
