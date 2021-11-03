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
        args.test_list = './filelists/asmr_by_material_1hr_train.txt' 
        trainset = get_TSN_Data_set(args)
        args.test_list = './filelists/asmr_by_material_1hr_test.txt'
        valset = get_TSN_Data_set(args) 
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
    print("Check number of train examples: ", len(trainset))
    print("Check number of train loader examples: ", len(train_loader))
    print("first example: ", list(train_loader)[0])
    assert(len(trainset) > 0)
    assert(len(train_loader) > 0)
    return train_loader, test_loader


def test_model(model, criterion, test_loader, epoch, logger, visualization=False):
    model.eval()
    reduced_loss_ = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            model.parse_batch(batch)
            model.forward()
            if visualization:
                for j in range(len(model.decoder_output)):
                    plt.figure(figsize=(8, 9))
                    plt.subplot(311)
                    plt.imshow(model.real_B[j].data.cpu().numpy(), 
                                    aspect='auto', origin='lower')
                    plt.title(model.video_name[j]+"_ground_truth")
                    plt.subplot(312)
                    plt.imshow(model.fake_B[j].data.cpu().numpy(), 
                                    aspect='auto', origin='lower')
                    plt.title(model.video_name[j]+"_predict")
                    plt.subplot(313)
                    plt.imshow(model.fake_B_postnet[j].data.cpu().numpy(), 
                                    aspect='auto', origin='lower')
                    plt.title(model.video_name[j]+"_postnet")
                    plt.tight_layout()
                    viz_dir = os.path.join(config.save_dir, "viz", f'epoch_{epoch:05d}')
                    os.makedirs(viz_dir, exist_ok=True)
                    plt.savefig(os.path.join(viz_dir, model.video_name[j]+".jpg"))
                    plt.close()

            targets = model.gt_raw_freqs
            targets.requires_grad = False
            loss = criterion(model.decoder_output, targets)
            #loss = criterion((model.fake_B, model.fake_B_postnet), model.real_B)
            reduced_loss = loss.item()
            reduced_loss_.append(reduced_loss)
            if not math.isnan(reduced_loss):
                print("Test loss epoch:{} iter:{} {:.6f} ".format(epoch, i, reduced_loss))
        logger.log_testing(np.mean(reduced_loss_), epoch)
    model.train()
    return np.mean(reduced_loss_) # Return the average of loss over test set


def train(args):
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    # Include the extra_upsampling parameter
    #model = Regnet(extra_upsampling=args.extra_upsampling, adversarial_loss=args.adversarial_loss)
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

    logger = RegnetLogger(os.path.join(config.save_dir, 'logs'), exclude_D_r_f=config.exclude_D_r_f, exclude_gan_loss=config.exclude_gan_loss)

    print("Preparing data...")
    train_loader, test_loader = prepare_dataloaders(args)

    # Keep track of the lowest test evaluation loss achieved
    lowest_test_loss = np.inf
    do_not_delete = []

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0

    if config.checkpoint_path != '':
        model.load_checkpoint(config.checkpoint_path)
        iteration = model.iteration
        iteration += 1  # next iteration is iteration + 1
        epoch_offset = max(0, int(iteration / len(train_loader)))
    config.epoch_count = epoch_offset
    model.setup()

    print(f"Epoch offset: {epoch_offset} for epochs: {config.epochs}")

    model.train()
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in tqdm(range(epoch_offset, config.epochs)):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            print(f"index: {i}, num items in batch: {len(batch)}")

            start = time.perf_counter()
            model.zero_grad()

            # Use a different parse_batch and forward function if using pairing loss
            model.parse_batch(batch)
            
            model.optimize_parameters()
            learning_rate = model.optimizer.param_groups[0]['lr']

            targets = model.gt_raw_freqs
            targets.requires_grad = False
            loss = criterion(model.decoder_output, targets)
            reduced_loss = loss.item()

            if not math.isnan(reduced_loss):
                duration = time.perf_counter() - start
                print("epoch:{} iter:{} loss:{:.6f} L1:{:.6f}  real-fake:{:.6f} time:{:.2f}s/it".format(
                        epoch, i, reduced_loss, model.loss_L1, (model.gt_raw_freqs - model.decoder_output).mean(), duration))
                logger.log_training(model, reduced_loss, learning_rate, duration, iteration)

            iteration += 1
        if epoch % config.num_epoch_save != 0:
            test_loss = test_model(model, criterion, test_loader, epoch, logger)
            if test_loss < lowest_test_loss:
                lowest_test_loss = test_loss
                print('Lower test loss! At epoch: ', epoch)
                do_not_delete.clear() # Remove the previously saved do not delete checkpoint, since we got a lower test loss
                model.save_checkpoint(config.save_dir, iteration, do_not_delete=do_not_delete, save_current=True)

        if epoch % config.num_epoch_save == 0:
            print("evaluation and save model")
            test_loss = test_model(model, criterion, test_loader, epoch, logger, visualization=False)
            if test_loss < lowest_test_loss:
                lowest_test_loss = test_loss
                print('Lower test loss! At epoch: ', epoch)
                do_not_delete.clear()
                model.save_checkpoint(config.save_dir, iteration, do_not_delete=do_not_delete, save_current=True)
            else:
                # If test loss is not the minimum seen so far
                model.save_checkpoint(config.save_dir, iteration, do_not_delete=do_not_delete)

        model.update_learning_rate()
    model_path = model.save_checkpoint(config.save_dir, iteration, do_not_delete=do_not_delete)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str)
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

    recorder = Recorder(config.save_dir, config.exclude_dirs)

    torch.backends.cudnn.enabled = config.cudnn_enabled
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    print("Dynamic Loss Scaling:", config.dynamic_loss_scaling)
    print("cuDNN Enabled:", config.cudnn_enabled)
    print("cuDNN Benchmark:", config.cudnn_benchmark)
    
    print("Config being used: \n", config)
    train(args)
