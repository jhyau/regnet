import sys, os
import argparse
from glob import glob

# ffmpeg -i v.mp4 -i a.wav -c:v copy -map 0:v:0 -map 1:a:0 new.mp4

parser = argparse.ArgumentParser('This file will replace the audio, given a video')
parser.add_argument('job_type', type=str, help='bulk (provide text file of videos to replace audio for), dir (replace video with all audio found in a directory), or single video')
parser.add_argument('vid_path', type=str, help='Path to the video. If in bulk, then to directory of the videos')
parser.add_argument('audio_path', type=str, help='Path to the audio to replace in the video. If in bulk, then to directory of audio files')
parser.add_argument('output_path', type=str, help='Path to output directory')
parser.add_argument('--save_title', type=str, default=None, help='Specify path/title to save the video. Defaults to None and will name new video normally')
parser.add_argument('--file', type=str, default=None, help='path to file of videos to replace (e.g. filelists/asmr_both_vids_test.txt) where each video name is listed per line')
args = parser.parse_args()

print('Replacing video with audio based on args: ', args)

if not os.path.isdir(args.output_path):
    os.path.mkdirs(args.output_path, exist_ok=True)

# given list of videos, assumes that the audioes to replace will be the vocoder ground truth output as well as the vocoder regnet's prediction
if args.job_type == 'bulk':
    if args.file is None:
        print('You need to provide a file of videos for bulk job of replacing audio')
        sys.exit(1)
    # Get the audio from both regnet prediction and waveglow ground truth
    # Assumes audio path is the directory of regnet and vocoder inference (_synthesis.wav and _gt_synthesis.wav)
    with open(args.file, 'r') as f:
        line = f.readline()
        while (line):
            vid = os.path.join(args.vid_path, line.strip()+'.mp4')
            audio_pred = os.path.join(args.audio_path, line.strip()+'_synthesis.wav')
            audio_gt = os.path.join(args.audio_path, line.strip()+'_gt_synthesis.wav')
            pred_vid = os.path.join(args.output_path, audio_pred.split('/')[-1].split('.')[0])
            gt_vid = os.path.join(args.output_path, audio_gt.split('/')[-1].split('.')[0])
            print('Saving video with prediction audio: ', pred_vid)
            print('Saving video with gt audio: ', gt_vid)

            os.system(f"ffmpeg -i {vid} -i {audio_pred} -c:v copy -map 0:v:0 -map 1:a:0 {pred_vid}.mp4")
            os.system(f"ffmpeg -i {vid} -i {audio_gt} -c:v copy -map 0:v:0 -map 1:a:0 {gt_vid}.mp4")
            line = f.readline()
elif args.job_type == 'single':
    if args.save_title is None:
        name = args.audio_path.split('/')[-1].split('.')[0]
        args.save_title = name
    new_vid_name = os.path.join(args.output_path, args.save_title)
    print('Saving new video: ', new_vid_name)
    os.system(f"ffmpeg -i {args.vid_path} -i {args.audio_path} -c:v copy -map 0:v:0 -map 1:a:0 {new_vid_name}.mp4")
else:
    # Find all audio in the given audio directory and replace the audio for the given video with each audio found
    wav_list = glob(os.path.join(args.audio_path, "*.wav"))
    for wav in wav_list:
        save_title = wav.split('/')[-1].split('.')[0]
        vid_path = os.path.join(args.output_path, save_title+'.mp4')
        os.system(f"ffmpeg -i {args.vid_path} -i {wav} -c:v copy -map 0:v:0 -map 1:a:0 {vid_path}")

print("Done replacing the audio in video")
