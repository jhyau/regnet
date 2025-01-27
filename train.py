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
from data_utils import RegnetLoader, get_TSN_Data_set
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

def prepare_dataloaders(args):
    # Get data, data loaders and collate function ready
    #trainset = RegnetLoader(config.training_files, include_landmarks=args.include_landmarks)
    #valset = RegnetLoader(config.test_files, include_landmarks=args.include_landmarks)

    if config.train_visual_feature_extractor:
        print("Getting the images to be stacked...")
        args.input_dir = config.optical_flow_dir
        args.test_list = config.training_files #'./filelists/asmr_by_material_1hr_train.txt'
        trainset = get_TSN_Data_set(args)
        args.test_list = config.test_files  #'./filelists/asmr_by_material_1hr_test.txt'
        valset = get_TSN_Data_set(args)
    else:
        trainset = RegnetLoader(config.training_files, include_landmarks=config.include_landmarks, pairing_loss=config.pairing_loss)
        valset = RegnetLoader(config.test_files, include_landmarks=config.include_landmarks, pairing_loss=config.pairing_loss)

    # Handle the tuple of tuples loaded from RegnetLoader when pairing loss is used within parse_batch in the model
    train_loader = DataLoader(trainset, num_workers=args.workers, shuffle=True,
                              batch_size=config.batch_size, pin_memory=False,
                              drop_last=True)
    test_loader = DataLoader(valset, num_workers=args.workers, shuffle=False,
                             batch_size=config.batch_size, pin_memory=False)
    print("Check number of train examples: ", len(trainset))
    print("Check number of train loader examples: ", len(train_loader))
    print("Check number of val examples: ", len(valset))
    print("Check number of val loader examples: ", len(test_loader))
    assert(len(trainset) > 0)
    assert(len(train_loader) > 0)
    return train_loader, test_loader


def test_model(model, criterion, test_loader, epoch, logger, visualization=False):
    model.eval()
    reduced_loss_ = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Check if using pairing loss
            if config.pairing_loss:
                model.parse_batch_pairing_loss(batch)
                model.forward_pairing_loss()

                if visualization:
                    for j in range(len(model.fake_B_cen)):
                        # Center aligned
                        plt.figure(figsize=(8, 9))
                        plt.subplot(911)
                        plt.imshow(model.real_B_cen[j].data.cpu().numpy(),
                                    aspect='auto', origin='lower')
                        plt.title(model.video_name[j]+"center_ground_truth")
                        plt.subplot(912)
                        plt.imshow(model.fake_B_cen[j].data.cpu().numpy(),
                                    aspect='auto', origin='lower')
                        plt.title(model.video_name[j]+"center_predict")
                        plt.subplot(913)
                        plt.imshow(model.fake_B_cen_postnet[j].data.cpu().numpy(),
                                    aspect='auto', origin='lower')
                        plt.title(model.video_name[j]+"center_postnet")

                        # Back aligned
                        plt.subplot(914)
                        plt.imshow(model.real_B_back[j].data.cpu().numpy(),
                                    aspect='auto', origin='lower')
                        plt.title(model.video_name[j]+"back_ground_truth")
                        plt.subplot(915)
                        plt.imshow(model.fake_B_back[j].data.cpu().numpy(),
                                    aspect='auto', origin='lower')
                        plt.title(model.video_name[j]+"back_predict")
                        plt.subplot(916)
                        plt.imshow(model.fake_B_back_postnet[j].data.cpu().numpy(),
                                    aspect='auto', origin='lower')
                        plt.title(model.video_name[j]+"back_postnet")

                        # Forward aligned
                        plt.subplot(917)
                        plt.imshow(model.real_B_for[j].data.cpu().numpy(),
                                    aspect='auto', origin='lower')
                        plt.title(model.video_name[j]+"for_ground_truth")
                        plt.subplot(918)
                        plt.imshow(model.fake_B_for[j].data.cpu().numpy(),
                                    aspect='auto', origin='lower')
                        plt.title(model.video_name[j]+"for_predict")
                        plt.subplot(919)
                        plt.imshow(model.fake_B_for_postnet[j].data.cpu().numpy(),
                                    aspect='auto', origin='lower')
                        plt.title(model.video_name[j]+"for_postnet")
                        
                        plt.tight_layout()
                        viz_dir = os.path.join(config.save_dir, "viz", f'epoch_{epoch:05d}')
                        os.makedirs(viz_dir, exist_ok=True)
                        plt.savefig(os.path.join(viz_dir, model.video_name[j]+".jpg"))
                        plt.close()

                        # Save the three generated audio samples batches
                        eval_dir = os.path.join(config.save_dir, "mel_spec_eval", f'epoch_{epoch:05d}')
                        os.makedirs(eval_dir, exist_ok=True)
                        np.save(os.path.join(eval_dir, model.video_name[j]+"_center.npy"), model.fake_B_cen[j].data.cpu().numpy())
                        np.save(os.path.join(eval_dir, model.video_name[j]+"_forward.npy"), model.fake_B_for[j].data.cpu().numpy())
                        np.save(os.path.join(eval_dir, model.video_name[j]+"_back.npy"), model.fake_B_back[j].data.cpu().numpy())

                # Calculate the loss of the three examples
                loss_cen = criterion((model.fake_B_cen, model.fake_B_cen_postnet), model.real_B_cen)
                loss_back = criterion((model.fake_B_back, model.fake_B_back_postnet), model.real_B_back)
                loss_for = criterion((model.fake_B_for, model.fake_B_for_postnet), model.real_B_for)
                loss = (1.0/3) * loss_cen + (1.0/3) * loss_back + (1.0/3)*loss_for
            else:
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

                        # Save the eval generated audio mel specs
                        eval_dir = os.path.join(config.save_dir, "mel_spec_eval", f'epoch_{epoch:05d}')
                        os.makedirs(eval_dir, exist_ok=True)
                        np.save(os.path.join(eval_dir, model.video_name[j]+".npy"), model.fake_B[j].data.cpu().numpy())
                loss = criterion((model.fake_B, model.fake_B_postnet), model.real_B)
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
    model = Regnet(extra_upsampling=config.extra_upsampling,adversarial_loss=args.adversarial_loss)

    criterion = RegnetLoss(config.loss_type)

    logger = RegnetLogger(os.path.join(config.save_dir, 'logs'), exclude_D_r_f=config.exclude_D_r_f, exclude_gan_loss=config.exclude_gan_loss, modal_losses=config.load_modal_data)

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
            start = time.perf_counter()
            model.zero_grad()

            # Use a different parse_batch and forward function if using pairing loss
            if config.pairing_loss:
                model.parse_batch_pairing_loss(batch)
            elif config.train_visual_feature_extractor:
                model.parse_batch_train_visual_feat_extractor(batch)
            else:
                model.parse_batch(batch)
            
            model.optimize_parameters()
            learning_rate = model.optimizers[0].param_groups[0]['lr']

            if config.pairing_loss:
                # Need to take into account all 3 aligned examples
                loss_cen = criterion((model.fake_B_cen, model.fake_B_cen_postnet), model.real_B_cen)
                loss_back = criterion((model.fake_B_back, model.fake_B_back_postnet), model.real_B_back)
                loss_for = criterion((model.fake_B_for, model.fake_B_for_postnet), model.real_B_for)
                loss = (1.0/3) * loss_cen + (1.0/3) * loss_back + (1.0/3)*loss_for 
            else:
                loss = criterion((model.fake_B, model.fake_B_postnet), model.real_B)
            reduced_loss = loss.item()

            if not math.isnan(reduced_loss):
                duration = time.perf_counter() - start
                if config.pairing_loss:
                    # Remove the discriminator real - fake mean loss
                    if not config.wo_G_GAN:
                        print("epoch:{} iter:{} loss:{:.6f} G:{:.6f} temporal:{:.6f} D:{:.6f} G_s:{:.6f} time:{:.2f}s/it".format(
                        epoch, i, reduced_loss, model.loss_G, model.loss_temporal, model.loss_D, model.loss_G_silence, duration))
                    else:
                        print("epoch:{} iter:{} loss:{:.6f} G:{:.6f} temporal:{:.6f} G_s:{:.6f} time:{:.2f}s/it".format(
                        epoch, i, reduced_loss, model.loss_G, model.loss_temporal, model.loss_G_silence, duration))
                else:
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
    # Args for BN-Inception set up for training feature extractor
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
    parser.add_argument('--no_adversarial_loss', dest='adversarial_loss', action='store_false', help='include this flag to set adversarial loss to False, so GAN loss will not be used')
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
