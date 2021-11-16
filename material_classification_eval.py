import argparse
import math
import os
import random
import shutil
import time

import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from Recorder import Recorder
from data_utils import RegnetLoader, get_TSN_Data_set
from logger import RegnetLogger
from criterion import RegnetLoss
from model import Regnet, Modal_Response_Net, MaterialClassificationNet
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
    test_loader = DataLoader(valset, num_workers=0, shuffle=True,
                             batch_size=config.batch_size, pin_memory=False)
    print("Check number of train examples: ", len(trainset))
    print("Check number of train loader examples: ", len(train_loader))
    assert(len(trainset) > 0)
    assert(len(train_loader) > 0)
    return train_loader, test_loader


def test_model(args, visualization=True):
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    # Classification loss: CrossEntropy
    if config.loss_type == 'cross_entropy':
        print("Using cross entropy loss")
        loss_fn = nn.CrossEntropyLoss()
    elif config.loss_type == 'nll':
        print("Using negative log likelihood loss")
        loss_fn = nn.NLLLoss()
    else:
        raise("Error, unknown loss!")
    criterion = loss_fn
    print("Initialized loss")

    #logger = RegnetLogger(os.path.join(config.save_dir, 'logs'), exclude_D_r_f=config.exclude_D_r_f, exclude_gan_loss=config.exclude_gan_loss)

    print("Preparing data...")
    train_loader, test_loader = prepare_dataloaders(args)

    if args.eval_train:
        loader = train_loader
    else:
        loader = test_loader

    # Need the number of classes when initialize model
    model = MaterialClassificationNet(len(train_loader.dataset.classes))
    if config.checkpoint_path != '':
        model.load_checkpoint(config.checkpoint_path)
    else:
        raise("Need checkpoint path")
    print("Loaded model")

    model.setup()
    model.eval()
    reduced_loss_ = []
    with torch.no_grad():
        # For confusion matrix
        y_true_total = []
        y_pred_total = []

        for i, batch in enumerate(loader):
            model.parse_batch(batch)
            model.forward()

            targets = model.labels
            targets.requires_grad = False
            loss = criterion(model.output, targets)

            if visualization:
                # Take the max along the classes dimension, where the index of the max is the predicted class
                values, indices = torch.max(model.output, dim=1)
                preds = indices.tolist()
                y_pred_total += preds

                # Get the labels
                y_true_total += model.labels.tolist()

            reduced_loss = loss.item()
            reduced_loss_.append(reduced_loss)
            
            # Get the percentage
            #print(f"prediction: \n{y_pred}\n and labels: \n{y_true}")
            correct = 0
            assert(len(preds) == len(model.labels.tolist()))

            for j in range(len(model.labels.tolist())):
                if preds[j] == model.labels.tolist()[j]:
                    correct += 1

            correct_percent = correct / len(model.labels.tolist())
            print(f"Percent correctly classified for iter {i}: {correct_percent}")

            if not math.isnan(reduced_loss):
                print("Test loss iter:{} {:.6f} ".format(i, reduced_loss))

        # Total correct
        total_correct = 0
        for k in range(len(y_true_total)):
            if y_true_total[k] == y_pred_total[k]:
                total_correct += 1
        print(f"Total correct percent: {total_correct / len(y_true_total)}")

        if visualization:
            viz_dir = os.path.join(config.save_dir, "inference_viz")
            os.makedirs(viz_dir, exist_ok=True)

            # Set axes label size to be smaller
            plt.rcParams.update({'font.size': 7})

            # Create confusion matrix
            # transformed labels
            lab = test_loader.dataset.le.transform(test_loader.dataset.le.classes_)
            matrix = confusion_matrix(y_true_total, y_pred_total, labels=lab, normalize="all")
            disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=test_loader.dataset.le.classes_)
            disp.plot(include_values=False, xticks_rotation="vertical")
            #plt.xticks(rotation=90)
            plt.tight_layout()
            if not args.eval_train:
                plt.title(f'test inference on test (normalized on all)')
            else:
                plt.title(f'inference on train (normalized on all)')
            plt.show()
            if not args.eval_train:
                plt.savefig(os.path.join(viz_dir, f'test_inference_normalize_all.jpg'))
            else:
                plt.savefig(os.path.join(viz_dir, f'train_inference_normalize_all.jpg'))

            # Normalized over true
            matrix = confusion_matrix(y_true_total, y_pred_total, labels=lab, normalize="true")
            disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=test_loader.dataset.le.classes_)
            disp.plot(include_values=False, xticks_rotation="vertical")
            #plt.xticks(rotation=90)
            plt.tight_layout()
            if not args.eval_train:
                plt.title(f'test inference on test (normalized on true labels)')
            else:
                plt.title(f'inference on train (normalized on true labels)')
            plt.show()
            if not args.eval_train:
                plt.savefig(os.path.join(viz_dir, f'test_inference_normalize_true.jpg'))
            else:
                plt.savefig(os.path.join(viz_dir, f'train_inference_normalize_true.jpg'))

            # Normalized over pred
            matrix = confusion_matrix(y_true_total, y_pred_total, labels=lab, normalize="pred")
            disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=test_loader.dataset.le.classes_)
            disp.plot(include_values=False, xticks_rotation="vertical")
            #plt.xticks(rotation=90)
            plt.tight_layout()
            if not args.eval_train:
                plt.title(f'test inference on test (normalized on predictions)')
            else:
                plt.title(f'inference on train (normalized on predictions)')
            plt.show()
            if not args.eval_train:
                plt.savefig(os.path.join(viz_dir, f'test_inference_normalize_pred.jpg'))
            else:
                plt.savefig(os.path.join(viz_dir, f'train_inference_normalize_pred.jpg'))
            plt.close()

        #logger.log_testing(np.mean(reduced_loss_), epoch)
    #model.train()
    return np.mean(reduced_loss_) # Return the average of loss over test set


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_train', action='store_true', help='Run inference on train set instead of test')
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--crop_fusion_type', type=str, default='avg',
                        choices=['avg', 'max', 'topk'])
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--flow_prefix', type=str, default='')
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
    test_model(args)
