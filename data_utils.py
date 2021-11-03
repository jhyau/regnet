import os
import pickle
import random
import math
import numpy as np
import torch
import torchvision
from PIL import Image
from glob import glob
import torch.utils.data
from config import _C as config


class GroupScale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Scale(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        rep_std = self.std * (tensor.size()[0]//len(self.std))

        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor

class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB': # PIL Image.convert, can convert to different modes ("L", "RGB", "HSV", etc.)
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)

class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


def get_TSN_Data_set(args):
    if args.modality == 'RGB':
        image_tmpl="img_{:05d}.jpg"
    elif args.modality == 'RGB_landmarks':
        image_tmpl="img_{:05d}_landmarks.jpg"
    else:
        image_tmpl=args.flow_prefix+"flow_{}_{:05d}.jpg"

    print("Using input size: ", args.input_size)
    cropping = torchvision.transforms.Compose([
        GroupScale((args.input_size, args.input_size)),
    ])

    #data_loader = torch.utils.data.DataLoader(
    #        TSNDataSet(args.input_dir, args.test_list,
    #                modality=args.modality,
    #                #image_tmpl="img_{:05d}.jpg" if args.modality == 'RGB' else args.flow_prefix+"flow_{}_{:05d}.jpg",
    #                image_tmpl=image_tmpl,
    #                transform=torchvision.transforms.Compose([
    #                    cropping, Stack(roll=True),
    #                    ToTorchFormatTensor(div=False),
    #                    GroupNormalize(net.input_mean, net.input_std),
    #                ])),
    #        batch_size=1, shuffle=False,
    #        num_workers=1, pin_memory=True)
    print(f"Input dir: {args.input_dir}")
    print(f"Test list: {args.test_list}")

    # BN-Inception mean and std
    input_mean = [104, 117, 128]
    input_std = [1]
    #input_mean = [0.485, 0.456, 0.406]
    #input_std = [0.229, 0.224, 0.225]

    return TSNDataSet(args.input_dir, args.test_list,
                    modality=args.modality,
                    image_tmpl="img_{:05d}.jpg" if args.modality == 'RGB' else args.flow_prefix+"flow_{}_{:05d}.jpg",
                    #image_tmpl=image_tmpl,
                    transform=torchvision.transforms.Compose([
                        cropping, Stack(roll=True),
                        ToTorchFormatTensor(div=False),
                        GroupNormalize(input_mean, input_std),
                    ]))


class TSNDataSet(torch.utils.data.Dataset):
    def __init__(self, root_path, list_file, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None):

        self.root_path = root_path
        self.list_file = list_file
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        with open(list_file) as f:
            self.video_list = [line.strip() for line in f]
            print(f"Num videos in dataset: {len(self.video_list)}")
        f.close()

    def _load_image(self, directory, idx):
        #print(f"modality: {self.modality} and template: {self.image_tmpl}")
        if self.modality == 'RGB' or self.modality == 'RGB_landmarks':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')
            return [x_img, y_img]

    def _get_modal_feature(self, video_name):
        modal_path = os.path.join(config.modal_features_dir, video_name+"_"+config.load_modal_data_type+".npy")
        #print(f"Get the gt vector: {modal_path}")
        feat = np.load(modal_path)
        assert feat.shape[-1] == config.n_modal_frequencies
        return feat

    def __getitem__(self, index):
        video_path = os.path.join(self.root_path, self.video_list[index])
        images_rgb = list()
        images_flow = list()
        # Need to fix the regex for this one :(
        #num_frames = len(glob(os.path.join(video_path, "img*.jpg")))
        num_frames_rgb = len(glob(os.path.join(video_path, "img*.jpg"))) - len(glob(os.path.join(video_path, "img*_landmarks.jpg")))
        #elif self.modality == 'Flow':
        num_frames_flow = len(glob(os.path.join(video_path, "flow_x*.jpg")))

        assert(num_frames_rgb == num_frames_flow)

        # Don't use all frames if video_samples is different
        if num_frames_rgb != config.video_samples:
            num_frames_rgb = config.video_samples
            num_frames_flow = config.video_samples

        # Get RGB images first
        self.modality = 'RGB'
        self.image_tmpl="img_{:05d}.jpg"
        for ind in (np.arange(num_frames_rgb)+1):
            images_rgb.extend(self._load_image(video_path, ind))

        # Get flow images
        flow_prefix = ''
        self.modality = 'Flow'
        self.image_tmpl = flow_prefix+"flow_{}_{:05d}.jpg"
        for ind in (np.arange(num_frames_flow)+1):
            images_flow.extend(self._load_image(video_path, ind))

        process_data_rgb = self.transform(images_rgb)
        process_data_flow = self.transform(images_flow)

        # Get the corresponding ground truth frequency
        feat = self._get_modal_feature(self.video_list[index])
        return (process_data_rgb, process_data_flow, feat, video_path)

    def __len__(self):
        return len(self.video_list)



class RegnetLoader(torch.utils.data.Dataset):
    """
    loads image, flow feature, mel-spectrogramsfiles
    """

    def __init__(self, list_file, max_sample=-1, include_landmarks=True, pairing_loss=True):
        self.video_samples = config.video_samples
        self.audio_samples = config.audio_samples
        self.mel_samples = config.mel_samples
        self.include_landmarks = include_landmarks
        self.pairing_loss = pairing_loss
        self.num_misalign_frames = config.num_misalign_frames
        self.reduced_video_samples = config.reduced_video_samples
        self.reduced_mel_samples = config.reduced_mel_samples
        self.video_fps = config.video_fps

        # Load modal response info (freqs, gains, dampings) to act as ground truth
        self.load_modal_data = config.load_modal_data
        self.load_modal_data_type = config.load_modal_data_type
        self.n_modal_frequencies = config.n_modal_frequencies

        with open(list_file, encoding='utf-8') as f:
            self.video_ids = [line.strip() for line in f]
        #print("Video IDs of dataset: ", self.video_ids)


    def get_feature_modal_response(self, video_id):
        im_path = os.path.join(config.rgb_feature_dir, video_id+".pkl")
        flow_path = os.path.join(config.flow_feature_dir, video_id+".pkl")
        modal_feat = os.path.join(config.modal_features_dir, video_id+"_"+self.load_modal_data_type+".npy") 

        im = self.get_im(im_path)
        flow = self.get_flow(flow_path)
        feats = self.get_modal_feature(modal_feat)
        feature = np.concatenate((im, flow), 1) # Visual dim=2048
        feature = torch.FloatTensor(feature.astype(np.float32))
        return (feature, feats, video_id)


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

            # Note that for 44100 audio sampling rate, 1720 mel samples, 172 is one second
            # These videos are 21.5 fps, so the seconds are reduced_video_samples / 21.5
            #mel_center_start = int(self.num_misalign_frames / self.video_fps)
            #mel_center_end = int((self.num_misalign_frames+self.reduced_video_samples) / self.video_fps)
            #mel_center = mel[:, mel_center_start:mel_center_end]
            mel_center = self.get_mel_subset(mel, self.num_misalign_frames, self.num_misalign_frames+self.reduced_video_samples)

            # Calculate the backward and forward shifts
            im_back = self.get_feature_subset(im, 0, self.reduced_video_samples)
            flow_back = self.get_feature_subset(flow, 0, self.reduced_video_samples)
            mel_back = self.get_mel_subset(mel, 0, self.reduced_video_samples)

            im_for = self.get_feature_subset(im, 2*self.num_misalign_frames, 2*self.num_misalign_frames+self.reduced_video_samples)
            flow_for = self.get_feature_subset(flow, 2*self.num_misalign_frames, 2*self.num_misalign_frames+self.reduced_video_samples)
            mel_for = self.get_mel_subset(mel, 2*self.num_misalign_frames, 2*self.num_misalign_frames+self.reduced_video_samples)

            if self.include_landmarks:
                assert(config.landmark_feature_dir is not None)
                land_path = os.path.join(config.landmark_feature_dir, video_id+".pkl")

                # Landmark features are same dims as RGB feature
                land = self.get_land(land_path)
                #land_center = land[self.num_misalign_frames:(self.num_misalign_frames+self.reduced_video_samples), :]
                land_center = self.get_feature_subset(land, self.num_misalign_frames, self.num_misalign_frames+self.reduced_video_samples)
                land_back = self.get_feature_subset(land, 0, self.reduced_video_samples)
                land_for = self.get_feature_subset(land, 2*self.num_misalign_frames, 2*self.num_misalign_frames+self.reduced_video_samples)

                # Create the examples to be loaded by the model
                feature_center = np.concatenate((im_center, flow_center, land_center), 1)
                feature_back = np.concatenate((im_back, flow_back, land_back), 1)
                feature_for = np.concatenate((im_for, flow_for, land_for), 1)
            else:
                feature_center = np.concatenate((im_center, flow_center), 1)
                feature_back = np.concatenate((im_back, flow_back), 1)
                feature_for = np.concatenate((im_for, flow_for), 1)

            feature_center = torch.FloatTensor(feature_center.astype(np.float32))
            feature_back = torch.FloatTensor(feature_back.astype(np.float32))
            feature_for = torch.FloatTensor(feature_for.astype(np.float32))

            # Return the examples for pairing loss
            # Center example (include label: 1 if aligned, 0 if misaligned)
            ex_center = (feature_center, mel_center, video_id, 1)
            ex_center_mis_back = (feature_center, mel_back, video_id, 0)
            ex_center_mis_for = (feature_center, mel_for, video_id, 0)

            # Backward examples
            ex_back = (feature_back, mel_back, video_id, 1)
            ex_back_mis_cen = (feature_back, mel_center, video_id, 0)
            ex_back_mis_for = (feature_back, mel_for, video_id, 0)

            # Forward examples
            ex_for = (feature_for, mel_for, video_id, 1)
            ex_for_mis_cen = (feature_for, mel_center, video_id, 0)
            ex_for_mis_back = (feature_for, mel_back, video_id, 0)

            assert(np.array_equal(feature_back[self.num_misalign_frames:, :], feature_center[:-self.num_misalign_frames, :]))
            
            mel_center_start = math.floor((self.num_misalign_frames / self.video_fps) * (self.mel_samples/self.audio_samples))
            assert(np.array_equal(mel_back[:, mel_center_start:], mel_center[:, :-mel_center_start])) 
            assert(np.array_equal(mel_center[:, mel_center_start:], mel_for[:, :-mel_center_start]))
            assert(np.array_equal(mel_back[:, int(2*mel_center_start):], mel_for[:, :-int(2*mel_center_start)]))

            # Return a tuple of examples (each a tuple as well)
            return (ex_center, ex_center_mis_back, ex_center_mis_for,
                    ex_back, ex_back_mis_cen, ex_back_mis_for,
                    ex_for, ex_for_mis_cen, ex_for_mis_back)
    

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

    def get_modal_feature(self, modal_path):
        feat = np.load(modal_path)
        assert feat.shape[-1] == self.n_modal_frequencies
        return feat

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
        one_second = self.mel_samples / self.audio_samples
        mel_center_start = math.floor((start_frame / self.video_fps) * one_second)
        mel_center_end = math.floor(((start_frame+self.reduced_video_samples) / self.video_fps) * one_second)

        # Force the mel spectrogram to match mel_samples shape
        #if (mel_center_end - mel_center_start) > self.reduced_mel_samples:
        #    mel_center_end = mel_center_end - ((mel_center_end - mel_center_start) - self.reduced_mel_samples)

        #print(f"mel start: {mel_center_start} and mel end: {mel_center_end}")
        mel_center = mel[:, mel_center_start:mel_center_end]
        return mel_center

    def __getitem__(self, index):
        if self.load_modal_data:
            return get_feature_modal_response(self.video_ids[index])
        else:
            return self.get_feature_mel_pair(self.video_ids[index])

    def __len__(self):
        return len(self.video_ids)
