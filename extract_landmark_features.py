import sys, os
import torch
import argparse
import pickle as pkl
import numpy as np
import torch.nn.parallel
import torchvision
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset
from mlp.models import Perceptron



class ToTorchFormatTensor(object):
    def __init__(self, div=True):
        self.div = div

    def __call__(self, tensor_group):
        return [torch.from_numpy(tensor) for tensor in tensor_group]


class Flatten(object):
    def __init__(self, start_dim=0, end_dim=-1):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def __call__(self, tensor_group):
        return [torch.flatten(tensor, start_dim=self.start_dim, end_dim=self.end_dim) for tensor in tensor_group]


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, tensor_group):
        #return torch.cat(img_group, axis=2)
        return torch.stack(tensor_group, dim=0)


class LandmarksDataset(Dataset):
    def __init__(self, root_path, list_file,
                 landmarks_tmpl='img_{:05d}.pt', transform=None):

        self.root_path = root_path
        self.list_file = list_file
        self.landmarks_tmpl = landmarks_tmpl
        self.transform = transform
        with open(list_file) as f:
            self.video_list = [line.strip() for line in f]
        f.close()

    def __getitem__(self, index):
        video_path = os.path.join(self.root_path, self.video_list[index])
        landmarks = list()
        num_frames = len(glob(os.path.join(video_path, "img*.pt"))) - len(glob(os.path.join(video_path, "img*_landmarks.pt")))

        for ind in (np.arange(num_frames)+1):
            # Store each landmark coordinate array into the list
            # use extend to add elements within an iterable (e.g. another list) into the list
            data = torch.load(os.path.join(video_path, self.landmarks_tmpl.format(ind)))
            arr = data['landmarks']
            landmarks.append(arr)
        process_data = self.transform(landmarks)
        return process_data, video_path

    def __len__(self):
        return len(self.video_list)



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Extract feature vectors from landmark coordinates to same dims as RGB feature vectors")
    parser.add_argument('-i', '--input_dir', type=str)
    parser.add_argument('-o', '--output_dir', type=str)
    parser.add_argument('-t', '--test_list', type=str)
    parser.add_argument('--input_size', type=int, default=126) # 42 * 3, need to flatten landmark coordinates first
    parser.add_argument('--output_size', type=int, default=1024) # RGB feature vectors are of dims (num frames in video, feature size)
    args = parser.parse_args()

    print('args for extract features: ', args)

    # Initialize feature extraction model
    net = Perceptron(args.input_size, args.output_size)

    # Initialize dataset of raw landmarks
    data_loader = torch.utils.data.DataLoader(
            LandmarksDataset(args.input_dir, args.test_list,
                transform=torchvision.transforms.Compose([
                    ToTorchFormatTensor(),
                    Flatten(),
                    Stack(),
                ])
            )
    )

    print("Number of examples in data loader: ", len(data_loader))
    net = torch.nn.DataParallel(net).cuda()
    net.eval()

    for i, (data, video_path) in enumerate(tqdm(data_loader)):
        os.makedirs(args.output_dir, exist_ok=True)
        ft_path = os.path.join(args.output_dir, video_path[0].split(os.sep)[-1]+".pkl")
        # Check shape of input
        input_var = torch.squeeze(data).float()
        print(f'Input shape: {input_var.shape}')
        rst = np.squeeze(net(input_var).data.cpu().numpy().copy())
        
        # Check output shape
        print(f"Output shape: {rst.shape}")
        pkl.dump(rst, open(ft_path, "wb"))
    
    print("Done with feature extraction for landmarks")
