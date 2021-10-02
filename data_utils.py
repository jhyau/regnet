import os
import pickle
import random
import numpy as np
import torch
import torch.utils.data
from config import _C as config


class RegnetLoader(torch.utils.data.Dataset):
    """
    loads image, flow feature, mel-spectrogramsfiles
    """

    def __init__(self, list_file, max_sample=-1, include_landmarks=True):
        self.video_samples = config.video_samples
        self.audio_samples = config.audio_samples
        self.mel_samples = config.mel_samples
        self.include_landmarks = include_landmarks
        self.pairing_loss = config.pairing_loss
        self.num_misalign_frames = config.num_misalign_frames
        self.reduced_video_samples = config.reduced_video_samples
        self.video_fps = config.video_fps

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

        if self.pairing_loss:
            # Shift forward and backward by num_misalign_frames
            # Note that the video_samples should be < 216 (or whatever is max num frames of the loaded feature vectors) to allow for shifting
            im_center = self.get_feature_subset(im, self.num_misalign_frames, self.num_misalign_frames+self.reduced_video_samples)
            #im_center = im[self.num_misalign_frames:(self.num_misalign_frames+self.reduced_video_samples), :]
            #flow_center = flow[self.num_misalign_frames:(self.num_misalign_frames+self.reduced_video_samples), :]
            flow_center = self.get_feature_subset(flow, self.num_misalign_frames, self.num_misalign_frames+self.reduced_video_samples)
            if self.include_landmarks:
                assert(config.landmark_feature_dir is not None)
                land_path = os.path.join(config.landmark_feature_dir, video_id+".pkl")

                # Landmark features are same dims as RGB feature
                land = self.get_land(land_path)
                #land_center = land[self.num_misalign_frames:(self.num_misalign_frames+self.reduced_video_samples), :]
                land_center = self.get_feature_subset(land, self.num_misalign_frames, self.num_misalign_frames+self.reduced_video_samples)
                
                # Note that for 44100 audio sampling rate, 1720 mel samples, 172 is one second
                # These videos are 21.5 fps, so the seconds are reduced_video_samples / 21.5
                mel_center_start = int(self.num_misalign_frames / self.video_fps)
                mel_center_end = int((self.num_misalign_frames+self.reduced_video_samples) / self.video_fps)
                mel_center = mel[:, mel_center_start:mel_center_end]

                # TODO: Create the examples to be loaded by the model
                feature_center = np.concatenate((im_center, flow_center, land_center), 1)

                # Center example (include label: 1 if aligned, 0 if misaligned)
                ex_center = (feature_center, mel_center, video_id, 1)

            else:
                feature_center = np.concatenate((im_center, flow_center), 1)
    

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
        """Expects one of three possible feature vectors: im (rgb), flow (optical flow), or land (hand landmarks vector)"""
        feat_subset = feature[start_idx:(start_idx+self.reduced_video_samples), :]
        return feat_subset

    def __getitem__(self, index):
        return self.get_feature_mel_pair(self.video_ids[index])

    def __len__(self):
        return len(self.video_ids)
