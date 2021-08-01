import argparse
import math
import os
import random
import shutil
import time

import torch
from torch.utils.data import DataLoader
import numpy as np
from Recorder import Recorder
from data_utils import RegnetLoader
from logger import RegnetLogger
from criterion import RegnetLoss
from model import Regnet
# from test import test_checkpoint
from contextlib import redirect_stdout
from config import _C as config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

def prepare_dataloaders():
    # Get data, data loaders and collate function ready
    trainset = RegnetLoader(config.training_files)
    valset = RegnetLoader(config.test_files)

    train_loader = DataLoader(trainset, num_workers=4, shuffle=True,
                              batch_size=config.batch_size, pin_memory=False,
                              drop_last=True)
    test_loader = DataLoader(valset, num_workers=4, shuffle=False,
                             batch_size=config.batch_size, pin_memory=False)
    print("Check number of train examples: ", len(trainset))
    print("Check number of train loader examples: ", len(train_loader))
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
                for j in range(len(model.fake_B)):
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
            loss = criterion((model.fake_B, model.fake_B_postnet), model.real_B)
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
    model = Regnet(extra_upsampling=args.extra_upsampling, adversarial_loss=args.adversarial_loss)

    criterion = RegnetLoss(config.loss_type)

    logger = RegnetLogger(os.path.join(config.save_dir, 'logs'))

    train_loader, test_loader = prepare_dataloaders()

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
            start = time.perf_counter()
            model.zero_grad()
            model.parse_batch(batch)
            model.optimize_parameters()
            learning_rate = model.optimizers[0].param_groups[0]['lr']
            loss = criterion((model.fake_B, model.fake_B_postnet), model.real_B)
            reduced_loss = loss.item()

            if not math.isnan(reduced_loss):
                duration = time.perf_counter() - start
                print("epoch:{} iter:{} loss:{:.6f} G:{:.6f} D:{:.6f} D_r-f:{:.6f} G_s:{:.6f} time:{:.2f}s/it".format(
                    epoch, i, reduced_loss, model.loss_G, model.loss_D, (model.pred_real - model.pred_fake).mean(), model.loss_G_silence, duration))
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
            test_loss = test_model(model, criterion, test_loader, epoch, logger, visualization=True)
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
    parser.add_argument('--extra_upsampling', action='store_true', help='include flag to add extra upsampling layers in the decoder and discriminator to match 44100 audio sample rate')
    parser.add_argument('--no_adversarial_loss', dest='adversarial_loss', action='store_false', help='include this flag to set adversarial loss to False, so GAN loss will not be used')
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
