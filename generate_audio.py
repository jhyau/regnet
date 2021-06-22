# coding: utf-8
import argparse
import sys, os
import torch
from torch.utils.data import DataLoader
import numpy as np
import librosa
from data_utils import RegnetLoader
from model import Regnet
from config import _C as config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from wavenet_vocoder import builder
# waveglow vocoder imports
import json
sys.path.append('./waveglow/')
sys.path.append('./waveglow/tacotron2/')
from scipy.io.wavfile import write
from waveglow.denoiser import Denoiser
from waveglow.mel2samp import files_to_list, load_wav_to_torch, MAX_WAV_VALUE
from waveglow.train import load_checkpoint
from waveglow.glow import WaveGlow, WaveGlowLoss

def build_wavenet(checkpoint_path=None, device='cuda:0'):
    model = builder.wavenet(
        out_channels=30,
        layers=24,
        stacks=4,
        residual_channels=512,
        gate_channels=512,
        skip_out_channels=256,
        cin_channels=80, # If n_mel_samples is not 80 anymore, need to change this part to match
        gin_channels=-1,
        weight_normalization=True,
        n_speakers=None,
        dropout=0.05,
        kernel_size=3,
        upsample_conditional_features=True,
        upsample_scales=[4, 4, 4, 4],
        freq_axis_kernel_size=3,
        scalar_input=True,
    )

    model = model.to(device)
    if checkpoint_path:
        print("Load WaveNet checkpoint from {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.make_generation_fast_()

    return model

def gen_waveform(model, save_path, c, device, args):
    initial_input = torch.zeros(1, 1, 1).to(device)
    if c.shape[1] != config.n_mel_channels:
        c = np.swapaxes(c, 0, 1)
    length = c.shape[0] * 256  # default: 860 * 256 = 220160, where mel_samples=860 and n_mel_channels=80. c is shape (860, 80) usually for 10 second prediction
    print(f"first dim in c shape: {c.shape}, length of the waveform to be generated: {length}")
    c = torch.FloatTensor(c.T).unsqueeze(0).to(device)
    print(f"actual shape for input melspectrogram: {c.shape}")
    with torch.no_grad():
        y_hat = model.incremental_forward(
            initial_input, c=c, g=None, T=length, tqdm=tqdm, softmax=True, quantize=True,
            log_scale_min=np.log(1e-14))
    waveform = y_hat.view(-1).cpu().data.numpy()
    librosa.output.write_wav(save_path, waveform, sr=args.sampling_rate) # default sr=22050

def gen_waveform_waveglow(args, save_path, c, device):
    # Set up the waveglow config
    #with open(args.waveglow_config, 'rb') as f:
    #    config = json.load(f)
    #waveglow_config = config["waveglow_config"]
    #train_config = config["train_config"]
    # Load model from checkpoint, then eval
    #model = WaveGlow(**waveglow_config).cuda()
    #optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    #waveglow, optimizer, iteration = load_checkpoint(args.waveglow_path, model, optimizer)
    waveglow = torch.load(args.waveglow_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()

    print(f"loaded mel spectrogram original shape: {c.shape}")
    #if c.shape[1] != config.n_mel_channels:
    #    c = np.swapaxes(c, 0, 1)
    #length = c.shape[0] * 256  # default: 860 * 256 = 220160, where mel_samples=860 and n_mel_channels=80. c is shape (860, 80) usually for 10 second prediction
    #print(f"first dim in c shape: {c.shape}, length of the waveform to be generated: {length}")
    #c = torch.FloatTensor(c.T).unsqueeze(0).to(device)
    #print(f"first dim in c shape: {c.shape}, length of the waveform to be generated: {length}")

    # install apex if want to use amp
    if args.is_fp16:
        print('using apex for waveglow')
        from apex import amp
        waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")
        #c = c.half()

    if args.denoiser_strength > 0:
        #denoiser = Denoiser(waveglow).cuda()
        denoiser = Denoiser(waveglow).to(device)

    mel = torch.autograd.Variable(c.to(device))
    mel = torch.unsqueeze(mel, 0)
    mel = mel.half() if args.is_fp16 else mel
    print(f"waveglow actual input melspectrogram shape: {mel.shape}")

    with torch.no_grad():
        audio = waveglow.infer(mel, sigma=args.sigma)
        if args.denoiser_strength  > 0:
            audio = denoiser(audio, args.denoiser_strength)
        #audio = audio * MAX_WAV_VALUE
    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio = audio.astype('int16')
    write(save_path, args.sampling_rate, audio)
    print(save_path)


def get_mel(filename):
        melspec = np.load(filename)
        #print(f"melspec shape: {melspec.shape}, num  of mel samples: {self.mel_samples}")
        if melspec.shape[1] < config.mel_samples:
            melspec_padded = np.zeros((melspec.shape[0], config.mel_samples))
            melspec_padded[:, 0:melspec.shape[1]] = melspec
        else:
            melspec_padded = melspec[:, 0:config.mel_samples]
        melspec_padded = torch.from_numpy(melspec_padded).float()
        return melspec_padded


def generate_audio(args, config):
    #valset = RegnetLoader(config.test_files)
    #test_loader = DataLoader(valset, num_workers=4, shuffle=False,
    #                         batch_size=config.batch_size, pin_memory=False)

    # Load the audio
    with open(config.test_files, encoding='utf-8') as f:
        video_ids = [line.strip() for line in f]

    test_loader = []
    for id in video_ids:
        mel = get_mel(os.path.join(config.mel_dir, id+"_mel.npy"))
        test_loader.append((id, mel))

    # Set up device and wavenet
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wavenet_model = build_wavenet(config.wavenet_path, device)
    for i,(id, mel_spec) in enumerate(test_loader):
        save_path = os.path.join(config.save_dir, id+"_gen_audio.wav")
        # Check the save dir exists
        if not os.path.exists(config.save_dir):
            os.mkdir(config.save_dir)
        print("Saving file: ", save_path)
        if args.vocoder == 'wavenet':
            gen_waveform(wavenet_model, save_path, mel_spec, device, args)
        else:
            gen_waveform_waveglow(args, save_path, mel_spec, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates the audio listed in the test files of the config')
    parser.add_argument('--vocoder', type=str, default='waveglow', help='Vocoder to use. waveglow or wavenet')
    parser.add_argument('-c', '--config_file', type=str, default='',
                        help='file for configuration')
    parser.add_argument('--waveglow_path', type=str, default='./pretrained_waveglow/published_ver/waveglow_256channels_universal_v5.pt', help='The path to waveglow checkpoint to load')
    parser.add_argument('--waveglow_config', type=str, default='./pretrained_waveglow/config.json', help='Config file for waveglow vocoder to load')
    parser.add_argument('--sigma', default=6.0, type=float)
    parser.add_argument('--sampling_rate', default=22050, type=int)
    parser.add_argument('--denoiser_strength', default=0.0, type=float, help='Removes model bias. Start with 0.1 and adjust')
    parser.add_argument('--is_fp16', action='store_true', help='Use the apex library to do mixed precision for waveglow')
    parser.add_argument('--gt', action='store_true', help='Use this flag to only generate sound for the ground truth')
    parser.add_argument('--extra_upsampling', action='store_true', help='include this flag to add extra upsampling layers to decoder and discriminator to match 44100 audio sample rate')
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        config.merge_from_file(args.config_file)
 
    config.merge_from_list(args.opts)
    # config.freeze()
    print("Config file settings: \n", config)
    print("args settings: \n", args)

    torch.backends.cudnn.enabled = config.cudnn_enabled
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    print("Dynamic Loss Scaling:", config.dynamic_loss_scaling)
    print("cuDNN Enabled:", config.cudnn_enabled)
    print("cuDNN Benchmark:", config.cudnn_benchmark)

    generate_audio(args, config)
