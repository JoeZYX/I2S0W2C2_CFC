#!/usr/bin/env python

#SBATCH --job-name=AugFramework

#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kurnaz@teco.edu

#SBATCH --export=ALL

#SBATCH --time=48:00:00

#SBATCH --partition=sdil
#SBATCH --gres=gpu:1


# usage: python train.py ITERATION_NO(SEED) MODEL_NAME DATASET_NAME CHANNEL_AUG PREDEF_AUG_NAME RNDAUG_PROB MAX_RNDAUG_CNT MIXUP_PROB MIXUP_ALPHA 
# python train.py 1 deepconvlstm hapt True undef 0.5 3 0.25 0.4
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("/pfs/data5/home/kit/tm/px3192/Time-Series-Data-Augmentation-Framework/")
sys.path.append('/pfs/data5/home/kit/tm/px3192/Time-Series-Data-Augmentation-Framework/notebooks/model/')
sys.path.append("../../")

from experiment import Exp

from dataloaders import data_set, data_dict
import torch
import yaml
import os
from math import floor, isclose

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

args = dotdict()

args.iteration        = int(sys.argv[1])
args.model_type       = sys.argv[2]

args.to_save_path     = "./saved_models/it-{}/Run_logs".format(args.iteration)
args.freq_save_path   = "./saved_models/it-{}/Freq_data".format(args.iteration)
args.window_save_path = "./saved_models/it-{}/Sliding_window".format(args.iteration)
args.root_path        = "../Datasets"


args.drop_transition  = False
args.datanorm_type    = "standardization" # None ,"standardization", "minmax"


args.batch_size       = 256                                                     
args.shuffle          = True
args.drop_last        = False
args.train_vali_quote = 0.90                                           


# training setting 
args.train_epochs            = 150

args.learning_rate           = 0.001  
args.learning_rate_patience  = 5
args.learning_rate_factor    = 0.1


args.early_stop_patience     = 15

args.use_gpu                 = True if torch.cuda.is_available() else False
args.gpu                     = 0
args.use_multi_gpu           = False

args.optimizer               = "Adam"
args.criterion               = "CrossEntropy"

args.seed                             = args.iteration


args.data_name                        =  sys.argv[3]

args.wavelet_filtering                = False
args.wavelet_filtering_regularization = False
args.wavelet_filtering_finetuning     = False
args.wavelet_filtering_finetuning_percent = 0.3

args.regulatization_tradeoff          = 0.0001
args.number_wavelet_filtering         = 10


args.use_channel      = sys.argv[4] == "True"
args.difference       = args.use_channel
args.filtering        = args.use_channel
args.magnitude        = args.use_channel
args.weighted_sampler = False




args.pos_select       = None
args.sensor_select    = None


args.representation_type = "time"
args.exp_mode            = "LOCV"

config_file = open('./configs/data.yaml', mode='r')
data_config = yaml.load(config_file, Loader=yaml.FullLoader)
config = data_config[args.data_name]

args.root_path       = os.path.join(args.root_path,config["filename"])
args.sampling_freq   = config["sampling_freq"]
args.num_classes     =  config["num_classes"]
window_seconds       = config["window_seconds"]
args.windowsize      =   int(window_seconds * args.sampling_freq) 
args.input_length    =  args.windowsize
# input information
args.c_in            = config["num_channels"]
if args.use_channel:
    args.c_in       += config["num_channels"] # differencing
    args.c_in       += config["num_acc"]      # filtering
    args.c_in       += config["num_channels"] # magnitude

rnd_augs = ['jitter',
            'exponential_smoothing',
            'moving_average',
            'magnitude_scaling',
            'magnitude_warp',
            'magnitude_shift',
            'time_warp',
            'window_warp',
            'window_slice',
            'random_sampling',
            'slope_adding'
            ]

assert sys.argv[5] == "undef" or sys.argv[5] in rnd_augs, "PREDEF AUG NOT FOUND"

args.predef_rndaug   = None if sys.argv[5] == "undef" else sys.argv[5]
args.rnd_aug_prob = float(sys.argv[6])

args.p = {x:args.rnd_aug_prob if not args.predef_rndaug else 0.0 for x in rnd_augs}
if args.predef_rndaug:
    args.p[args.predef_rndaug] = args.rnd_aug_prob

args.max_randaug_cnt = int(sys.argv[7])

if isclose(args.rnd_aug_prob, 0) or args.max_randaug_cnt == 0:
    args.rnd_aug_prob = 0
    args.max_randug_cnt = 0

args.mixup_p = float(sys.argv[8])

args.mixup_alpha = float(sys.argv[9])

if isclose(args.mixup_p, 0):
    args.mixup_alpha = 0.0

print('''
      Iteration={}
      Model={}
      Dataset={}
      ChannelAug={}
      Predef_Rndaug={}
      RndAugP={}
      MaxRndAugCnt={}
      MixupP={}
      Mixup_Alpha={}
      '''.format(args.iteration,
                 args.model_type,
                 args.data_name,
                 args.use_channel,
                 args.predef_rndaug,
                 args.rnd_aug_prob,
                 args.max_randaug_cnt,
                 args.mixup_p,
                 args.mixup_alpha))

if args.wavelet_filtering :
    
    if args.windowsize%2==1:
        N_ds = int(torch.log2(torch.tensor(args.windowsize-1)).floor()) - 2
    else:
        N_ds = int(torch.log2(torch.tensor(args.windowsize)).floor()) - 2

    args.f_in            =  args.number_wavelet_filtering*N_ds+1
else:
    args.f_in            =  1


args.filter_scaling_factor = 0.25

exp = Exp(args)

exp.train()
