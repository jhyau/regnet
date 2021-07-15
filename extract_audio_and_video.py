import os
from glob import glob
import numpy as np
import os.path as P
import argparse
from multiprocessing import Pool
from functools import partial

def execCmd(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text

def pipline(video_path, output_dir, fps, sr, duration_target):
    video_name = os.path.basename(video_path)
    audio_name = video_name.replace(".mp4", ".wav")

    # Extract Original Audio
    ori_audio_dir = P.join(output_dir, "audio_ori")
    os.makedirs(ori_audio_dir, exist_ok=True)
    os.system(f"ffmpeg -i {video_path} -loglevel error -f wav -vn -y {P.join(ori_audio_dir, audio_name)}")

    # Cut Video According to Audio
    align_video_dir = P.join(output_dir, "videos_algin")
    os.makedirs(align_video_dir, exist_ok=True)
    duration = execCmd(f"ffmpeg -i {P.join(ori_audio_dir, audio_name)}  2>&1 | grep 'Duration' | cut -d ' ' -f 4 | sed s/,//")
    duration = duration.replace('\n', "")
    os.system("ffmpeg -ss 0 -t {} -i {} -loglevel error -c:v libx264 -c:a aac -strict experimental -b:a 98k -y {}".format(
            duration, video_path, P.join(align_video_dir, video_name)))

    # Repeat Video
    repeat_video_dir = P.join(output_dir, "videos_repeat")
    os.makedirs(repeat_video_dir, exist_ok=True)
    hour, min, sec = [float(_) for _ in duration.split(":")]
    duration_second = 3600*hour + 60*min + sec
    n_repeat = duration_target//duration_second + 1
    os.system("ffmpeg -stream_loop {} -i {} -loglevel error -c copy -fflags +genpts -y {}".format(n_repeat, 
            P.join(align_video_dir, video_name), P.join(repeat_video_dir, video_name)))

    # Cut Video
    cut_video_dir = P.join(output_dir, f"videos_{duration_target}s")
    os.makedirs(cut_video_dir, exist_ok=True)
    os.system("ffmpeg -ss 0 -t {} -i {} -loglevel error -c:v libx264 -c:a aac -strict experimental -b:a 98k -y {}".format(duration_target, 
            P.join(repeat_video_dir, video_name), P.join(cut_video_dir, video_name)))

    # Extract Audio
    cut_audio_dir = P.join(output_dir, f"audio_{duration_target}s")
    os.makedirs(cut_audio_dir, exist_ok=True)
    os.system("ffmpeg -i {} -loglevel error -f wav -vn -y {}".format(
            P.join(cut_video_dir, video_name), P.join(cut_audio_dir, audio_name)))

    # change audio sample rate
    sr_audio_dir = P.join(output_dir, f"audio_{duration_target}s_{sr}hz")
    os.makedirs(sr_audio_dir, exist_ok=True)
    os.system("ffmpeg -i {} -loglevel error -ac 1 -ab 16k -ar {} -y {}".format(
            P.join(cut_audio_dir, audio_name), sr, P.join(sr_audio_dir, audio_name)))

    # change video fps
    fps_audio_dir = P.join(output_dir, f"videos_{duration_target}s_{fps}fps")
    os.makedirs(fps_audio_dir, exist_ok=True)
    os.system("ffmpeg -y -i {} -loglevel error -r {} -c:v libx264 -strict -2 {}".format(
            P.join(cut_video_dir, video_name), fps, P.join(fps_audio_dir, video_name)))

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
    paser.add_argument("-i", "--input_dir", default="data/VAS/dog/videos")
    paser.add_argument("-o", "--output_dir", default="data/features/dog")
    paser.add_argument("-d", "--duration", type=int, default=10)
    paser.add_argument("-a", '--audio_sample_rate', default='44100') # originally 22050
    paser.add_argument("-v", '--video_fps', default='21.5') # For ASMR videos, original is 30 fps
    paser.add_argument("-n", '--num_worker', type=int, default=32)
    args = paser.parse_args()
    print(f'args for extracting audio and video: {args}')
    input_dir = args.input_dir
    output_dir = args.output_dir
    duration_target = args.duration
    sr = args.audio_sample_rate
    fps = args.video_fps
    
    video_paths = glob(P.join(input_dir, "*.mp4"))
    video_paths.sort()

    with Pool(args.num_worker) as p:
        p.map(partial(pipline, output_dir=output_dir, 
        sr=sr, fps=fps, duration_target=duration_target), video_paths)

    # Notify when done
    print("Worker pool is done and closed")
