import collections
import os
import pickle
import random
import math
import numpy as np
import torch
import torch.utils.data
from config import _C as config

import aligner

# named tuple to make code more portable.
PairingLossPoint = collections.namedtuple('PairingLossPoint', ['feature', 'mel', 'video_id', 'label'])

class RegnetLoader(torch.utils.data.Dataset):
    """
    loads image, flow feature, mel-spectrogramsfiles
    """

    def __init__(self, list_file, max_sample=-1, include_landmarks=True, pairing_loss=True,
                 randomize_samples=True):
        self.video_samples = config.video_samples
        self.audio_samples = config.audio_samples
        self.mel_samples = config.mel_samples
        self.include_landmarks = include_landmarks
        self.randomize_samples = randomize_samples
        self.pairing_loss = pairing_loss
        self.num_misalign_frames = config.num_misalign_frames
        self.temporal_misalign_pos_samples = config.temporal_misalign_pos_samples
        self.temporal_misalign_neg_samples_per_pos = config.temporal_misalign_neg_samples_per_pos
        self.allow_temporal_misalign_overlap = config.allow_temporal_misalign_overlap
        self.reduced_video_samples = config.reduced_video_samples
        self.reduced_mel_samples = config.reduced_mel_samples
        self.video_fps = config.video_fps
        self.mel_to_vid = config.mel_to_vid
        self.mis_overlap = config.mis_overlap
        # This list is used when generating *non* randomized samples as a substitute
        # for aligner.
        self._non_rand_sample_idx = [(0, self.num_misalign_frames),
                                     (self.num_misalign_frames, 2*self.num_misalign_frames),
                                     (2*self.num_misalign_frames, 0)]

        with open(list_file, encoding='utf-8') as f:
            self.video_ids = [line.strip() for line in f]
        #print("Video IDs of dataset: ", self.video_ids)

    def get_feature_mel_pair(self, video_id):
        im_path = os.path.join(config.rgb_feature_dir, video_id+".pkl")
        flow_path = os.path.join(config.flow_feature_dir, video_id+".pkl")
        mel_path = os.path.join(config.mel_dir, video_id+"_mel.npy")
        im = self.get_im(im_path)
        flow = self.get_flow(flow_path)
        mel = self.get_mel(mel_path)
        output = []

        if self.pairing_loss:

            window_size = self.reduced_video_samples
            num_frames, _ = im.shape
            for i in range(self.temporal_misalign_pos_samples):
                if self.randomize_samples:
                    start_idx, mis_idx = aligner.get_misaligned_starts(num_frames, window_size, max_overlap=self.mis_overlap)
                else:
                    start_idx, mis_idx = self._non_rand_sample_idx[i]

                im_sub = self.get_feature_subset(im, start_idx, start_idx+window_size)
                flow_sub = self.get_feature_subset(flow, start_idx, start_idx+window_size)
                mel_sub = self.get_mel_subset(mel, start_idx, start_idx+window_size)
                
                im_mis = self.get_feature_subset(im, mis_idx, mis_idx+window_size)
                flow_mis = self.get_feature_subset(flow, mis_idx, mis_idx+window_size)
                mel_mis = self.get_mel_subset(mel, mis_idx, mis_idx+window_size)

                feat_sub = torch.FloatTensor(np.concatenate((im_sub, flow_sub), 1).astype(np.float32))
                feat_mis = torch.FloatTensor(np.concatenate((im_mis, flow_mis), 1).astype(np.float32))

                output.append(PairingLossPoint(feat_sub, mel_sub, video_id, 1))
                output.append(PairingLossPoint(feat_mis, mel_sub, video_id, 0))
                output.append(PairingLossPoint(feat_sub, mel_mis, video_id, 0))

            # Return a tuple of examples (each a tuple as well)
            
            return tuple(output)
    

        if self.include_landmarks:
            assert(config.landmark_feature_dir is not None)
            land_path = os.path.join(config.landmark_feature_dir, video_id+".pkl")

            # Landmark features are same dims as RGB feature
            land = self.get_land(land_path)

            # Since we concatenate landmark features as well, the visual dim will change to 3072 (parameter in config file)
            # Concatenate landmark features at the end of the feature vector, so RGB+optical flow+landmarks
            feature = np.concatenate((im, flow, land), 1)
        else:
            feature = np.concatenate((im, flow), 1) # Visual dim=2048
        feature = torch.FloatTensor(feature.astype(np.float32))
        return (feature, mel, video_id)

    def get_mel(self, filename):
        melspec = np.load(filename)
        #print(f"melspec shape: {melspec.shape}, num  of mel samples: {self.mel_samples}")
        if melspec.shape[1] < self.mel_samples:
            melspec_padded = np.zeros((melspec.shape[0], self.mel_samples))
            melspec_padded[:, 0:melspec.shape[1]] = melspec
        else:
            melspec_padded = melspec[:, 0:self.mel_samples]
        melspec_padded = torch.from_numpy(melspec_padded).float()
        return melspec_padded

    def get_im(self, im_path):
        with open(im_path, 'rb') as f:
            im = pickle.load(f, encoding='bytes')
        f.close()
        #print(f"img shape: {im.shape}, num of video samples: {self.video_samples}")
        if im.shape[0] < self.video_samples:
            im_padded = np.zeros((self.video_samples, im.shape[1]))
            im_padded[0:im.shape[0], :] = im
        else:
            im_padded = im[0:self.video_samples, :]
        assert im_padded.shape[0] == self.video_samples
        return im_padded

    def get_flow(self, flow_path):
        with open(flow_path, 'rb') as f:
            flow = pickle.load(f, encoding='bytes')
        f.close()
        if flow.shape[0] < self.video_samples:
            flow_padded = np.zeros((self.video_samples, flow.shape[1]))
            flow_padded[0:flow.shape[0], :] = flow
        else:
            flow_padded = flow[0:self.video_samples, :]
        return flow_padded

    def get_land(self, land_path):
        with open(land_path, 'rb') as f:
            land = pickle.load(f, encoding='bytes')
        f.close()
        if land.shape[0] < self.video_samples:
            land_padded = np.zeros((self.video_samples, land.shape[1]))
            land_padded[0:land.shape[0], :] = land
        else:
            land_padded = land[0:self.video_samples, :]
        assert land_padded.shape[0] == self.video_samples
        return land_padded

    def get_feature_subset(self, feature, start_idx, end_idx):
        """Expects one of three possible feature vectors: im (rgb), flow (optical flow), or land (hand landmarks vector)
        Dimensions expected: (video_sample, 1024)
        """
        feat_subset = feature[start_idx:(start_idx+self.reduced_video_samples), :]
        return feat_subset

    def get_mel_subset(self, mel, start_frame, end_frame):
        """Expected dimensions for mel spectrogram: (n_mel_channels, mel_samples)"""
        # Note that for 44100 audio sampling rate, 1720 mel samples, 172 is one second
        # These videos are 21.5 fps, so the seconds are reduced_video_samples / 21.5
        mel_center_start = start_frame * self.mel_to_vid
        mel_center_end = end_frame * self.mel_to_vid 

        # Force the mel spectrogram to match mel_samples shape
        #if (mel_center_end - mel_center_start) > self.reduced_mel_samples:
        #    mel_center_end = mel_center_end - ((mel_center_end - mel_center_start) - self.reduced_mel_samples)

        #print(f"mel start: {mel_center_start} and mel end: {mel_center_end}")
        mel_center = mel[:, mel_center_start:mel_center_end]
        assert mel_center.shape[1] == self.reduced_mel_samples 
        return mel_center

    def __getitem__(self, index):
        return self.get_feature_mel_pair(self.video_ids[index])

    def __len__(self):
        return len(self.video_ids)
