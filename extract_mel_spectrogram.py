import numpy as np
import os, sys
import librosa
import argparse
import os.path as P
from multiprocessing import Pool
from functools import partial
from glob import glob
sys.path.append('./waveglow')
sys.path.append('./waveglow/tacotron2')
print(sys.path)

from waveglow.mel2samp import files_to_list, load_wav_to_torch, TacotronSTFT, MAX_WAV_VALUE
import torch
import torch.utils.data
from scipy.io.wavfile import read
#mel_basis = librosa.filters.mel(22050, n_fft=1024, fmin=125, fmax=7600, n_mels=80)

def get_spectrogram(audio_path, save_dir, length, mel_basis, mel_samples, args):
    wav, _ = librosa.load(audio_path, sr=None)
    print(f'shape of audio from librosa.load: {wav.shape}')
    y = np.zeros(length)
    if wav.shape[0] < length:
        y[:len(wav)] = wav
    else:
        y = wav[:length]
    spectrogram = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    mel_spec = np.dot(mel_basis, spectrogram)    
    mel_spec = 20 * np.log10(np.maximum(1e-5, mel_spec)) - 20
    mel_spec = np.clip((mel_spec + 100) / 100, 0, 1.0)
    print(f'mel_spec space: {mel_spec.shape}')

    #mel_spec = mel_spec[:, :860]
    #assert(mel_samples == 1720)
    save_dir = save_dir + '_orig_method'
    mel_spec = mel_spec[:, :mel_samples]
    print(f'wavenet melspectrogram shape: {mel_spec.shape}')
    os.makedirs(save_dir, exist_ok=True)
    audio_name = os.path.basename(audio_path).split('.')[0]
    np.save(P.join(save_dir, audio_name + "_mel.npy"), mel_spec)
    np.save(P.join(save_dir, audio_name + "_audio.npy"), y)


def get_spectrogram_waveglow(audio_path, save_dir, length, mel_samples, args):
    """
    Convert audio to mel spectrograms following waveglow's mel2samp functions
    """
    print("Getting mel spectrograms for waveglow...")
    audio, sr = load_wav_to_torch(audio_path)
    print(f'shape of audio from scipy.load: {audio.shape}')

    if len(audio.size()) >= 2:
        audio = audio[:, 0] # (n_samples, n_channels)

    y = np.zeros(length)
    if audio.shape[0] < length:
        y[:len(audio)] = audio
    else:
        y = audio[:length]

    # Get the mel spectrogram
    audio_norm = audio / MAX_WAV_VALUE
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)

    # Set up STFT to get mel spectrograms
    stft = TacotronSTFT(filter_length=args.filter_length, hop_length=args.hop_length, win_length=args.win_length, sampling_rate=args.sampling_rate, mel_fmin=args.fmin, mel_fmax=args.fmax)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    print(f'Output melspec shape original: {melspec.size()}')
    melspec = melspec[:, :mel_samples]
    melspec = melspec.detach().cpu().numpy() # Convert torch tensor to numpy
    print(f'shape of melspec final: {melspec.shape}')

    # Separate waveglow mel specs for now
    #save_dir = save_dir + '_waveglow_mels'
    os.makedirs(save_dir, exist_ok=True)
    audio_name = os.path.basename(audio_path).split('.')[0]
    np.save(P.join(save_dir, audio_name + "_mel.npy"), melspec)
    np.save(P.join(save_dir, audio_name + "_audio.npy"), y)

if __name__ == '__main__':
    """
    "data_config": {
        "training_files": "train_files_tapping_wooden.txt",
        "validation_files": "test_files_tapping_wooden.txt",
        "segment_length": 32000,
        "sampling_rate": 44100,
        "filter_length": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "mel_fmin": 0.0,
        "mel_fmax": 16000.0
    },
    "dist_config": {
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54321"
    },

    "waveglow_config": {
        "n_mel_channels": 80,
        "n_flows": 12,
        "n_group": 8,
        "n_early_every": 4,
        "n_early_size": 2,
        "WN_config": {
            "n_layers": 8,
            "n_channels": 256,
            "kernel_size": 3
        }
    }
    """

    paser = argparse.ArgumentParser()
    paser.add_argument("-i", "--input_dir", default="data/features/dog/audio_10s_22050hz")
    paser.add_argument("-o", "--output_dir", default="data/features/dog/melspec_10s_22050hz")
    paser.add_argument("--sampling_rate", type=int, default=44100, help='Audio sampling rate') # default is 22050
    paser.add_argument("--n_fft", type=int, default=1024, help='Number of FFT components')
    paser.add_argument("--fmin", type=float, default=0.0, help="lowest frequency (in Hz)") # original regnet default (125)
    paser.add_argument("--fmax", type=float, default=16000.0, help="highest frequency in Hz") # original regnet default (7600)
    paser.add_argument("--n_mels", type=int, default=80, help='number of mel bands to generate')
    paser.add_argument("--mel_samples", type=int, default=1720, help='Number of mel spectrogram audio samples') # In config file, default is 860
    paser.add_argument("--vocoder", type=str, default='waveglow', help='Vocoder that will be generating audio (waveglow or wavenet)')
    paser.add_argument("--filter_length", type=int, default=1024, help='filter length for Tacotron STFT')
    paser.add_argument("--hop_length", type=int, default=256, help='Hop length for STFT')
    paser.add_argument("--win_length", type=int, default=1024, help='Win length for STFT')
    # Maybe need to change length here to match 15 seconds (22050 * 15)? Default is 10 seconds
    paser.add_argument("-l", "--length", default=441000) # Default is 10 seconds with 22050 audio sample rate, so 220500
    paser.add_argument("-n", '--num_worker', type=int, default=32)
    args = paser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    length = int(args.length)
    print("args for extract mel spectrograms: ", args)

    # Check that the input directory exists
    if not os.path.isdir(input_dir):
        raise Exception("Provided input file for extracting mel spectrograms does not exist")

    audio_paths = glob(P.join(input_dir, "*.wav"))
    audio_paths.sort()

    mel_basis = librosa.filters.mel(args.sampling_rate, n_fft=args.n_fft, fmin=args.fmin, fmax=args.fmax, n_mels=args.n_mels)

    with Pool(args.num_worker) as p:
        if args.vocoder == 'waveglow':
            p.map(partial(get_spectrogram_waveglow, save_dir=output_dir, length=length, mel_samples=args.mel_samples, args=args), audio_paths)
        else:
            p.map(partial(get_spectrogram, save_dir=output_dir, length=length, mel_basis=mel_basis, mel_samples=args.mel_samples, args=args), audio_paths)

    print("extracting mel spectrograms worker pool has closed")
