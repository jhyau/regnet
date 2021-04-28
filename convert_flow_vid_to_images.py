import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", type=str, help='directory with flow videos')
parser.add_argument("output_dir", type=str, help='directory to output flow video images')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

videos = os.listdir(args.input_dir)
for vid in tqdm(videos):
    print(f"Processing video {vid}...")
    os.system(f"ffmpeg -i {os.path.join(args.input_dir,vid)} -vf fps=30 {args.output_dir}/%04d.jpg")

print("Completed extracting optical flow as image frames from flow video")
