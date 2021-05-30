import numpy as np
import os
import librosa
import argparse
import os.path as P
from multiprocessing import Pool
from functools import partial
from glob import glob

#mel_basis = librosa.filters.mel(22050, n_fft=1024, fmin=125, fmax=7600, n_mels=80)

def get_spectrogram(audio_path, save_dir, length, mel_basis, mel_samples):
    wav, _ = librosa.load(audio_path, sr=None)
    y = np.zeros(length)
    if wav.shape[0] < length:
        y[:len(wav)] = wav
    else:
        y = wav[:length]
    spectrogram = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    mel_spec = np.dot(mel_basis, spectrogram)    
    mel_spec = 20 * np.log10(np.maximum(1e-5, mel_spec)) - 20
    mel_spec = np.clip((mel_spec + 100) / 100, 0, 1.0)
    
    #mel_spec = mel_spec[:, :860]
    #assert(mel_samples == 1720)
    mel_spec = mel_spec[:, :mel_samples]
    os.makedirs(save_dir, exist_ok=True)
    audio_name = os.path.basename(audio_path).split('.')[0]
    np.save(P.join(save_dir, audio_name + "_mel.npy"), mel_spec)
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
    paser.add_argument("--fmin", type=float, default=125, help="lowest frequency (in Hz)") # Use original regnet default, not waveglow
    paser.add_argument("--fmax", type=float, default=7600, help="highest frequency in Hz") # Use original regnet default, not waveglow
    paser.add_argument("--n_mels", type=int, default=80, help='number of mel bands to generate')
    paser.add_argument("--mel_samples", type=int, default=1720, help='Number of mel spectrogram audio samples') # In config file, default is 860
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
        p.map(partial(get_spectrogram, save_dir=output_dir, length=length, mel_basis=mel_basis, mel_samples=args.mel_samples), audio_paths)
