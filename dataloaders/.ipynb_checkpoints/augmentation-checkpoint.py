import numpy as np
import pandas as pd
from typing import List
from scipy.interpolate import CubicSpline
import torch
import random
# x1: all channel values first sample
# x2: all channel values second sample
# _alpha: mixup parameter for beta distribution
# https://arxiv.org/abs/1710.09412
# returns x: all channel values mixed up
# completed
def mixup_data(x, y, alpha=0.4, argmax = False, device="cpu"):

    """
    Returns mixed inputs, pairs of targets, and lambda
    """

    batch_size = x.shape[0]
    lam = np.random.beta(alpha, alpha, batch_size)
    # t = max(t, 1-t)
    lam = np.concatenate([lam[:, None], 1 - lam[:, None]], 1).max(1)
    # tensor and cuda version of lam
    lam = x.new(lam)

    shuffle = torch.randperm(batch_size).to(device)

    x1, y1 = x[shuffle], y[shuffle]
    # out_shape = [bs, 1, 1]
    out_shape = [lam.size(0)] + [1 for _ in range(len(x1.shape) - 1)]

    # [bs, temporal, sensor]
    mixed_x = (x * lam.view(out_shape) + x1 * (1 - lam).view(out_shape))
    # [bs, 3]
    if not argmax:
        y_a_y_b_lam = torch.cat([y[:, None].float(), y1[:, None].float(), lam[:, None].float()], 1)
    else:
        y_a_y_b_lam = []
        for index,lam_ in enumerate(lam):
            if lam_ >= 0.5:
                y_a_y_b_lam.append(y[index])
            else:
                y_a_y_b_lam.append(y1[index])
        y_a_y_b_lam = torch.tensor(y_a_y_b_lam)

    return mixed_x, y_a_y_b_lam

# TODO https://mohcinemadkour.github.io/posts/2019/10/Machine%20Learning,%20timeseriesAI,%20Time%20Series%20Classification,%20fastai_timeseries,%20data%20augmentation/
# https://arxiv.org/abs/1708.04552
def cutout():
    pass

# TODO https://arxiv.org/abs/1905.04899
def cutmix():
    pass

class RandomAugment(object):
    def __init__(self, transformation_count, random_augmentation_config, max_aug):

        self.tranformation_count = transformation_count
        self.random_augmentation_config = random_augmentation_config
        self.max_aug = max_aug

        self.all_transformations_dict = {
            "jitter":self.jitter,
            #self.exponential_smoothing,
            "moving_average":self.moving_average,
            "magnitude_scaling":self.magnitude_scaling,
            "magnitude_warp":self.magnitude_warp,
            "magnitude_shift":self.magnitude_shift,
            "time_warp":self.time_warp,
            "window_warp":self.window_warp,
            "window_slice":self.window_slice,
            "random_sampling":self.random_sampling,
            "slope_adding":self.slope_adding,
        }

        self.all_selected_transformations = []
        for key in random_augmentation_config.keys():
            if random_augmentation_config[key]:
                self.all_selected_transformations.append(self.all_transformations_dict[key])

        assert len(self.all_selected_transformations)  == transformation_count


    def __call__(self, org_sample_x):
        # TODO: if used with torch, use torch's rng
        sample_x = org_sample_x.copy()
        nr = np.random.randint(1,self.max_aug+1,1)[0]

        transformations = np.random.choice(
            self.all_selected_transformations, size=nr, replace=False
        )
        #print(nr, transformations)
        for t in transformations:
            sample_x = t(sample_x)
        return sample_x


    @staticmethod

    def jitter(x: np.ndarray, sigma: float = 0.5):
        """
        input x shape should be [Batch, Seq_length, Channel]  or [Seq_length, Channel]
        
        """
        if len(x.shape)==2:
            x = np.expand_dims(x, axis=0)
        batch = x.shape[0]
        seq_l = x.shape[1]
        sensr = x.shape[2]

        scale = np.random.uniform(0,sigma,(batch,sensr))
        channel_variance = np.std(x, axis=1)

        assert scale.shape == channel_variance.shape
        
        var_random = scale*channel_variance

        ret = np.zeros_like(x)

        for i in range(batch):
            for j in range(sensr):
                ret[i,:,j] = x[i,:,j] + np.random.normal( loc=0.0, scale=var_random[i,j], size=seq_l )
            
        return ret







    # @staticmethod
    # def exponential_smoothing(x: np.ndarray, alpha: float = 0.5):
    #     x = x.transpose()
    #     length = x.shape[1]
    #     ret = np.zeros_like(x)
    #     for dim in range(x.shape[0]):
    #         ret[dim, 0] = x[dim, 0]
    #         for index in range(1, length):
    #             ret[dim, index] = (
    #                 alpha * x[dim, index] + (1 - alpha) * ret[dim, index - 1]
    #             )
    #     return ret.transpose()


    @staticmethod
    def moving_average(x: np.ndarray, window_ratio_max = 0.06):
        if len(x.shape)==2:
            x = np.expand_dims(x, axis=0)
        batch = x.shape[0]
        seq_l = x.shape[1]
        sensr = x.shape[2]
        ret = np.zeros_like(x)
        for i in range(batch):
            df = pd.DataFrame(x[i])
            window_size = df.shape[0]

            window = max(1,int(window_size*np.random.uniform(1/window_size,window_ratio_max,1)[0]))

            avaraged_df = df.rolling(window =window).mean()
            avaraged_df = avaraged_df.interpolate(method='linear', limit_direction='both')
            ret[i] = avaraged_df.values
        return ret

    @staticmethod
    def magnitude_scaling(x: np.ndarray, sigma: float = 0.15):
        # https://arxiv.org/pdf/1706.00527.pdf
        if len(x.shape)==2:
            x = np.expand_dims(x, axis=0)
        batch = x.shape[0]
        seq_l = x.shape[1]
        sensr = x.shape[2]
        
        ret = np.zeros_like(x)
        
        for i in range(batch):

            factor = np.random.normal(loc=1., scale=sigma, size=(sensr))
            ret[i] = np.multiply(x[i], factor[np.newaxis,:])


        return ret


    @staticmethod
    def magnitude_warp(x, sigma=0.4, knot=4):
        from scipy.interpolate import CubicSpline
        
        if len(x.shape)==2:
            x = np.expand_dims(x, axis=0)
        batch = x.shape[0]
        seq_l = x.shape[1]
        sensr = x.shape[2]
        
        ret = np.zeros_like(x)
        orig_steps = np.arange(seq_l)

        
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(batch, knot+2, sensr))


        warp_steps = (np.ones((sensr,1))*(np.linspace(0, seq_l-1., num=knot+2))).T

        ret = np.zeros_like(x)
        
        for i,pat in enumerate(x):
            warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(sensr)]).T

            ret[i] = pat * warper


        return ret




    @staticmethod
    def magnitude_shift(x, max_ratio = 0.1):
        if len(x.shape)==2:
            x = np.expand_dims(x, axis=0)
        batch = x.shape[0]
        seq_l = x.shape[1]
        sensr = x.shape[2]
        
        ret = np.zeros_like(x)
        
        ratio = np.random.uniform(-max_ratio,max_ratio,(batch,sensr))

        for i in range(batch):
            for j in range(sensr):
                ret[i,:,j] =  x[i,:,j] + max( np.mean(x[i,:,j])*ratio[i,j],     np.std(x[i,:,j])*ratio[i,j] )
        return ret


    @staticmethod
    def time_warp(x, sigma=0.05, knot=4):
        
        if len(x.shape)==2:
            x = np.expand_dims(x, axis=0)
        batch = x.shape[0]
        seq_l = x.shape[1]
        sensr = x.shape[2]

        from scipy.interpolate import CubicSpline
        orig_steps = np.arange(seq_l)
        
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(batch, knot+2, sensr))

        warp_steps = (np.ones((sensr,1))*(np.linspace(0, seq_l-1., num=knot+2))).T
        
        ret = np.zeros_like(x)

        for i, pat in enumerate(x):
            for dim in range(sensr):
                time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
                scale = (seq_l-1)/time_warp[-1]
                ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, seq_l-1), pat[:,dim]).T

        return ret


    @staticmethod
    def window_warp(x, window_ratio_min=0.1, window_ratio_max=0.2, scales=[0.5, 2.]):
        x = np.array(x)
        if len(x.shape)==2:
            x = np.expand_dims(x, axis=0)
        batch = x.shape[0]
        seq_l = x.shape[1]
        sensr = x.shape[2]
        # https://halshs.archives-ouvertes.fr/halshs-01357973/document
        window_ratio_list = np.random.uniform(window_ratio_min,window_ratio_max,batch)
        
        warp_scales = np.random.choice(scales, batch)

        warp_size_list = np.ceil(window_ratio_list*seq_l).astype(int)


            
        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            warp_size     = warp_size_list[i]
            window_steps  = np.arange(warp_size)
            window_starts = np.random.randint(low=1, high=seq_l-warp_size-1, size=1).astype(int)[0]
            window_ends   = (window_starts + warp_size).astype(int)
            for dim in range(sensr):
                start_seg = pat[:window_starts,dim]
                window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts:window_ends,dim])
                end_seg = pat[window_ends:,dim]

                warped = np.concatenate((start_seg, window_seg, end_seg))                
                ret[i,:,dim] = np.interp(np.arange(seq_l), np.linspace(0, seq_l-1., num=warped.size), warped)
        return ret


    @staticmethod
    def window_slice(x, reduce_ratio_min=0.85,reduce_ratio_max=0.95):
        if len(x.shape)==2:
            x = np.expand_dims(x, axis=0)
        batch = x.shape[0]
        seq_l = x.shape[1]
        sensr = x.shape[2]
        
        reduce_ratio = np.random.uniform(reduce_ratio_min, reduce_ratio_max,batch)
        
        # https://halshs.archives-ouvertes.fr/halshs-01357973/document
        target_len_list = np.ceil(reduce_ratio*seq_l).astype(int)
        

        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            starts = np.random.randint(low=0, high=seq_l-target_len_list[i], size=(1)).astype(int)
            ends = (target_len_list[i] + starts).astype(int)
            for dim in range(sensr):
                ret[i,:,dim] = np.interp(np.linspace(0, target_len_list[i], num=seq_l), np.arange(target_len_list[i]), pat[starts[0]:ends[0],dim])
        return ret


    @staticmethod
    def random_sampling(x, reduce_ratio_min=0.85,reduce_ratio_max=0.95):
        
        if len(x.shape)==2:
            x = np.expand_dims(x, axis=0)
        batch = x.shape[0]
        seq_l = x.shape[1]
        sensr = x.shape[2]
        reduce_ratio_list = np.random.uniform(reduce_ratio_min, reduce_ratio_max,batch)

        target_len_list = np.ceil(reduce_ratio_list*seq_l).astype(int)


        index_list = list(np.arange(seq_l))

        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            sampled_index = np.random.choice(seq_l, target_len_list[i], replace=False)
            sampled_index = sorted(sampled_index)
            
            for dim in range(sensr):
                ret[i,:,dim] = np.interp(np.linspace(0, target_len_list[i], num=seq_l), np.arange(target_len_list[i]), pat[sampled_index,dim])
        return ret


    @staticmethod
    def slope_adding(x):
        if len(x.shape)==2:
            x = np.expand_dims(x, axis=0)
        batch = x.shape[0]
        seq_l = x.shape[1]
        sensr = x.shape[2]

        

        ret = np.zeros_like(x)
        for i in range(batch):
            for dim in range(sensr):
                anchor = random.randint(0, seq_l-1)
                slope = random.uniform(-0.2,0.2)
                slope = slope/seq_l
                slope = np.linspace(0, seq_l-1,seq_l)*slope
                shift = slope[anchor]
                slope = slope -shift
                ret[i,:,dim] =   x[i,:,dim] + slope
        return ret
