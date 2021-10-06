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
from waveglow.mel2samp import files_to_list, MAX_WAV_VALUE
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
    print('generating with wavenet...')
    initial_input = torch.zeros(1, 1, 1).to(device)
    if c.shape[1] != config.n_mel_channels:
        c = np.swapaxes(c, 0, 1)
    length = c.shape[0] * 256  # default: 860 * 256 = 220160, where mel_samples=860 and n_mel_channels=80. c is shape (860, 80) usually for 10 second prediction
    print(f"first dim in c shape: {c.shape}, length of the waveform to be generated: {length}")
    c = torch.FloatTensor(c.T).unsqueeze(0).to(device)
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
    print("generate with waveglow..")
    waveglow = torch.load(args.waveglow_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()

    if c.shape[1] != config.n_mel_channels:
        c = np.swapaxes(c, 0, 1)
    length = c.shape[0] * 256  # default: 860 * 256 = 220160, where mel_samples=860 and n_mel_channels=80. c is shape (860, 80) usually for 10 second prediction
    print(f"first dim in c shape: {c.shape}, length of the waveform to be generated: {length}")
    #c = torch.FloatTensor(c.T).unsqueeze(0).to(device)

    # install apex if want to use amp
    if args.is_fp16:
        print("using apex for waveglow sound generation...")
        from apex import amp
        waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")
        #c = c.half()

    if args.denoiser_strength > 0:
        denoiser = Denoiser(waveglow).cuda()

    mel = torch.autograd.Variable(c)
    mel = torch.unsqueeze(mel, 0)
    #c = torch.FloatTensor(c.T).unsqueeze(0).to(device)
    mel = mel.half() if args.is_fp16 else mel

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

def test_model(args, config):
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    # add extra_upsampling parameter
    model = Regnet(extra_upsampling=config.extra_upsampling)
    valset = RegnetLoader(config.test_files, include_landmarks=config.include_landmarks, pairing_loss=config.pairing_loss)
    test_loader = DataLoader(valset, num_workers=4, shuffle=False,
                             batch_size=config.batch_size, pin_memory=False)
    if config.checkpoint_path != '':
        model.load_checkpoint(config.checkpoint_path)
    model.setup()
    model.eval()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wavenet_model = build_wavenet(config.wavenet_path, device) 
    with torch.no_grad():
        os.makedirs(config.save_dir, exist_ok=True)
        with open(os.path.join(config.save_dir, 'mel_files.txt'), 'w') as file:
            for i, batch in enumerate(test_loader):
                #if config.pairing_loss:
                #    model.parse_batch_pairing_loss()
                #    model.forward_pairing_loss()
                model.parse_batch(batch)
                model.forward()            
                for j in range(len(model.fake_B)):
                    plt.figure(figsize=(8, 9))
                    plt.subplot(311)
                    # model.real_B is the ground truth spectrogram
                    print(f"ground truth spec size: {model.real_B[j].data.cpu().numpy().shape}")
                    print(f"ground truth max: {np.max(model.real_B[j].data.cpu().numpy())} and min: {np.min(model.real_B[j].data.cpu().numpy())}")
                    plt.imshow(model.real_B[j].data.cpu().numpy(), 
                                    aspect='auto', origin='lower')
                    plt.title(model.video_name[j]+"_ground_truth")
                    plt.subplot(312)
                    # model.fake_B is the generator's prediction
                    print(f"prediction spec size: {model.fake_B[j].data.cpu().numpy().shape}")
                    print(f'prediction max: {np.max(model.fake_B[j].data.cpu().numpy())} and min: {np.min(model.fake_B[j].data.cpu().numpy())}')
                    plt.imshow(model.fake_B[j].data.cpu().numpy(), 
                                    aspect='auto', origin='lower')
                    plt.title(model.video_name[j]+"_predict")
                    plt.subplot(313)
                    # model.fake_B_postnet is the generator's postnet prediction
                    plt.imshow(model.fake_B_postnet[j].data.cpu().numpy(), 
                                    aspect='auto', origin='lower')
                    plt.title(model.video_name[j]+"_postnet")
                    plt.xlabel('Time')
                    plt.tight_layout()

                    # Make sure save directory exists
                    os.makedirs(config.save_dir, exist_ok=True)
                    plt.savefig(os.path.join(config.save_dir, model.video_name[j]+".jpg"))
                    plt.close()

                    # Save a zoomed-in plot so time dim is stretched out
                    # Assuming the sample is 10 seconds, split to parts of 2 seconds
                    num_plots = args.num_plots
                    step_size = config.mel_samples / num_plots
                    for step in range(num_plots):
                        #print(f"Starting time dim: {step*step_size}, ending time dim: {step_size*(step+1)}")
                        plt.figure(figsize=(8,9))
                        #extent = [step*step_size, step_size*(step+1), min(model.real_B[j].data.cpu().numpy().any()), max(model.real_B[j].data.cpu().numpy().any())]
                        extent = [step*step_size, step_size*(step+1), 0, model.real_B[j].data.cpu().numpy().shape[0]] # For imshow of (m,n) array, image is m rows and n columns
                        plt.subplot(311)
                        if step == num_plots-1:
                            plt.imshow(model.real_B[j].data.cpu().numpy()[:,int(step*step_size):],
                                    aspect='auto', origin='lower', extent=extent)
                        else:
                            plt.imshow(model.real_B[j].data.cpu().numpy()[:,int(step*step_size):int(step_size*(step+1))],
                                    aspect='auto', origin='lower', extent=extent)
                        plt.title(model.video_name[j]+"_ground_truth_part"+str(step))

                        plt.subplot(312)
                        #extent = [step*step_size, step_size*(step+1), min(model.fake_B[j].data.cpu().numpy().any()), max(model.fake_B[j].data.cpu().numpy().any())]
                        extent = [step*step_size, step_size*(step+1), 0, model.fake_B[j].data.cpu().numpy().shape[0]]
                        if step == num_plots-1:
                            plt.imshow(model.fake_B[j].data.cpu().numpy()[:,int(step*step_size):],
                                    aspect='auto', origin='lower', extent=extent)
                        else:
                            plt.imshow(model.fake_B[j].data.cpu().numpy()[:,int(step*step_size):int(step_size*(step+1))],
                                    aspect='auto', origin='lower', extent=extent)
                        plt.title(model.video_name[j]+"_predict_part"+str(step))

                        plt.subplot(313)
                        #extent = [step*step_size, step_size*(step+1), min(model.fake_B_postnet[j].data.cpu().numpy().any()), max(model.fake_B_postnet[j].data.cpu().numpy().any())]
                        extent = [step*step_size, step_size*(step+1), 0, model.fake_B_postnet[j].data.cpu().numpy().shape[0]]
                        if step == num_plots-1:
                            plt.imshow(model.fake_B_postnet[j].data.cpu().numpy()[:,int(step*step_size):],
                                    aspect='auto', origin='lower', extent=extent)
                        else:
                            plt.imshow(model.fake_B_postnet[j].data.cpu().numpy()[:,int(step*step_size):int(step_size*(step+1))],
                                    aspect='auto', origin='lower', extent=extent)
                        plt.title(model.video_name[j]+"_postnet_part"+str(step))
                        plt.xlabel('Time')
                        plt.tight_layout()
                        plt.savefig(os.path.join(config.save_dir, model.video_name[j]+f"_part{step}_of_{num_plots}.jpg"))
                        plt.close()

                    file.write('../'+os.path.join(config.save_dir, model.video_name[j]+".npy \n"))
                    file.write('../'+os.path.join(config.save_dir, model.video_name[j]+"_gt.npy \n"))
                    
                    # Saving the model prediction mel spec as numpy file
                    np.save(os.path.join(config.save_dir, model.video_name[j]+".npy"), 
                              model.fake_B[j].data.cpu().numpy())
                    # Save ground truth as well
                    np.save(os.path.join(config.save_dir, model.video_name[j]+"_gt.npy"),
                            model.real_B[j].data.cpu().numpy())
                    # Save postnet prediction
                    #np.save(os.path.join(config.save_dir, model.video_name[j]+"_postnet.npy"),
                    #        model.fake_B_postnet[j].data.cpu().numpy())
                    # Using the prediction mel spectrogram to generate sound
                    if args.gt:
                        print("using ground truth melspectrograms for vocoder inference...")
                        mel_spec = model.real_B[j].data.cpu().numpy()
                        save_path = os.path.join(config.save_dir, model.video_name[j]+"_gt.wav")
                    else:
                        mel_spec = model.fake_B[j].data.cpu().numpy()
                        save_path = os.path.join(config.save_dir, model.video_name[j]+".wav")

                    if args.gt_and_pred:
                        mel_spec_gt = model.real_B[j].data.cpu().numpy()
                        save_path_gt = os.path.join(config.save_dir, model.video_name[j]+"_gt.wav")
                        mel_spec_pred = model.fake_B[j].data.cpu().numpy()
                        save_path_pred = os.path.join(config.save_dir, model.video_name[j]+".wav")

                        if args.vocoder == 'wavenet':
                            gen_waveform(wavenet_model, save_path_gt, mel_spec_gt, device, args)
                            gen_waveform(wavenet_model, save_path_pred, mel_spec_pred, device, args)
                        else:
                            print('For waveglow, run inference separately')
                            #gen_waveform_waveglow(args, save_path_gt, mel_spec_gt, device)
                            #gen_waveform_waveglow(args, save_path_pred, mel_spec_pred, device)
                    else:
                        if args.gt:
                            print("using ground truth melspectrograms for vocoder inference...")
                            mel_spec = model.real_B[j].data.cpu().numpy()
                            save_path = os.path.join(config.save_dir, model.video_name[j]+"_gt.wav")
                        else:
                            mel_spec = model.fake_B[j].data.cpu().numpy()
                            save_path = os.path.join(config.save_dir, model.video_name[j]+".wav")

                        if args.vocoder == 'wavenet':
                            gen_waveform(wavenet_model, save_path, mel_spec, device, args)
                        else:
                            print("Not immediately generating audio with waveglow")
                            #gen_waveform_waveglow(args, save_path, mel_spec, device)
    model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocoder', type=str, default='waveglow', help='Vocoder to use. waveglow or wavenet')
    parser.add_argument('-c', '--config_file', type=str, default='',
                        help='file for configuration')
    parser.add_argument('--waveglow_path', type=str, default='./pretrained_waveglow/published_ver/waveglow_256channels_universal_v5.pt', help='The path to waveglow checkpoint to load. Currently default is set to Waveglow published weights')
    parser.add_argument('--waveglow_config', type=str, default='./pretrained_waveglow/config.json', help='Config file for waveglow vocoder to load')
    parser.add_argument('--sigma', default=2.0, type=float)
    parser.add_argument('--sampling_rate', default=22050, type=int)
    parser.add_argument('--denoiser_strength', default=0.0, type=float, help='Removes model bias. Start with 0.1 and adjust')
    parser.add_argument('--is_fp16', action='store_true', help='Use the apex library to do mixed precision for waveglow')
    parser.add_argument('--num_plots', default=35, type=int, help='How many smaller plots to split the time dimension of the mel spectrogram plots')
    parser.add_argument('--gt', action='store_true', help='generate only ground truth audio')
    parser.add_argument('--gt_and_pred', action='store_true', help='generate both ground truth and prediction audio')
    #parser.add_argument('--extra_upsampling', action='store_true', help='include this flag to add extra upsampling layers to decoder and discriminator to match 44100 audio sample rate')
    #parser.add_argument('--include_landmarks', action='store_true', help='Include flag to include landmarks in features')
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

    test_model(args, config)
