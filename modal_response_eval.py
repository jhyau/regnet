import argparse
import math
import os
import random
import shutil
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from Recorder import Recorder
from data_utils import RegnetLoader, get_TSN_Data_set
from logger import RegnetLogger
from criterion import RegnetLoss
from model import Regnet, Modal_Response_Net
# from test import test_checkpoint
from contextlib import redirect_stdout
from config import _C as config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

def prepare_dataloaders(args):
    # Get data, data loaders and collate function ready
    #trainset = RegnetLoader(config.training_files, include_landmarks=args.include_landmarks)
    #valset = RegnetLoader(config.test_files, include_landmarks=args.include_landmarks)
    if config.train_visual_feature_extractor:
        print("Getting the images to be stacked...")
        trainset = get_TSN_Data_set(args, 'train')
        valset = get_TSN_Data_set(args, 'eval') 
    else:
        trainset = RegnetLoader(config.training_files, include_landmarks=config.include_landmarks, pairing_loss=config.pairing_loss)
        valset = RegnetLoader(config.test_files, include_landmarks=config.include_landmarks, pairing_loss=config.pairing_loss)

    # Handle the tuple of tuples loaded from RegnetLoader when pairing loss is used within parse_batch in the model
    # Originally, num_workers is set to 4 (seems to go out of memory when loading raw images)
    train_loader = DataLoader(trainset, num_workers=0, shuffle=True,
                              batch_size=config.batch_size, pin_memory=False,
                              drop_last=True)
    test_loader = DataLoader(valset, num_workers=0, shuffle=False,
                             batch_size=config.batch_size, pin_memory=False)
    print("Check number of test examples: ", len(valset))
    print("Check number of test loader examples: ", len(test_loader))
    assert(len(valset) > 0)
    assert(len(test_loader) > 0)
    return train_loader, test_loader


def test_model(args, visualization=True):
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    model = Modal_Response_Net()
    print("Initialized model")

    # Reconstruction loss
    if config.loss_type == "MSE":
        loss_fn = nn.MSELoss()
    elif config.loss_type == "L1Loss":
        loss_fn = nn.L1Loss()
    else:
        print("ERROR LOSS TYPE!")
    criterion = loss_fn
    #criterion = RegnetLoss(config.loss_type)
    print("Initialized loss")

    print("Preparing data...")
    train_loader, test_loader = prepare_dataloaders(args)
 
    model.load_checkpoint(config.checkpoint_path)
    print(f"Finished loading the model checkpoint")

    model.setup()
    model.eval()

    last_slash = config.checkpoint_path.rindex('/')
    eval_path = os.path.join(config.checkpoint_path[:last_slash+1], args.eval_output_dir)
    print(f"Save eval path: {eval_path}")
    os.makedirs(eval_path, exist_ok=True)
    reduced_loss_ = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            model.parse_batch(batch)
            model.forward()

            targets = model.gt_raw_freqs
            targets.requires_grad = False

            if visualization:
                for j in range(len(targets)):
                    name = model.video_name[j].split('/')[-1]
                    #print(name)
                    # Save the predicted frequency
                    np.save(os.path.join(eval_path, name+".npy"), model.decoder_output[j].data.cpu().numpy())
            try:
                loss = criterion(model.decoder_output, targets)
            except Exception as ex:
                print(str(ex))
                continue
            reduced_loss = loss.item()
            reduced_loss_.append(reduced_loss)
            if not math.isnan(reduced_loss):
                print("Test loss iter:{} {:.6f} ".format(i, reduced_loss))
    return np.mean(reduced_loss_) # Return the average of loss over test set


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default=config.optical_flow_dir)
    parser.add_argument('-e', '--eval_output_dir', type=str, default='eval_output', help="Output dir for evaluation results")
    parser.add_argument('-m', '--modality', type=str, choices=['RGB', 'RGB_landmarks', 'Flow'])
    parser.add_argument('-t', '--test_list', type=str)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--crop_fusion_type', type=str, default='avg',
                        choices=['avg', 'max', 'topk'])
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--flow_prefix', type=str, default='')
    #parser.add_argument('--extra_upsampling', action='store_true', help='include flag to add extra upsampling layers in the decoder and discriminator to match 44100 audio sample rate')
    #parser.add_argument('--include_landmarks', action='store_true', help='Include flag to concatenate skeletal landmark features to the feature vector')
    parser.add_argument('-c', '--config_file', type=str, default='',
                        help='file for configuration')
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    print("Using args: ", args)

    if args.config_file:
        config.merge_from_file(args.config_file)

    config.merge_from_list(args.opts)
    # config.freeze()

    os.makedirs(config.save_dir, exist_ok=True)
    with open(os.path.join(config.save_dir, 'opts.yml'), 'w') as f:
        with redirect_stdout(f):
            print(config.dump())
    f.close()

    torch.backends.cudnn.enabled = config.cudnn_enabled
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    print("Dynamic Loss Scaling:", config.dynamic_loss_scaling)
    print("cuDNN Enabled:", config.cudnn_enabled)
    print("cuDNN Benchmark:", config.cudnn_benchmark)

    print("Config being used: \n", config)
    test_model(args)
