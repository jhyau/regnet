import os, sys
import argparse
import random
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('audio_path', type=str, help="Path to where all the audio material types are. Directory of directories")
parser.add_argument('--val_split', type=float, default=0.2, help='percent split for validation set')
parser.add_argument('--val_num', type=int, default=None, help='Actual number of samples to be moved to val, will be used instead of val split is this is provided.')
args = parser.parse_args()
print(args)

# Get list of all dirs
dirs = os.listdir(args.audio_path)
for dir in dirs:
    print(f"Dealing with {dir}")

    # Get list of all files
    files = os.listdir(os.path.join(args.audio_path, dir))
    num_to_split = int(len(files) * args.val_split)
    os.makedirs(os.path.join(os.path.join(args.audio_path, dir), 'val'), exist_ok=True)
    val_path = os.path.join(os.path.join(args.audio_path, dir), 'val')
    print(f"validation path: {os.path.join(os.path.join(args.audio_path, dir), 'val')}")

    if num_to_split > 0 or args.val_num is not None:
        if args.val_num is not None:
            num_to_split = args.val_num
        print("Moving files to validation...")
        val_files = random.sample(files, num_to_split)
        for file in val_files:
            shutil.move(os.path.join(os.path.join(args.audio_path, dir), file), val_path)
            #shutil.move(os.path.join(os.path.join(os.path.join(args.audio_path, dir), 'train'), file), os.path.join(os.path.join(args.audio_path, dir), 'val'))

    # Move the rest to train
    os.makedirs(os.path.join(os.path.join(args.audio_path, dir), 'train'), exist_ok=True)
    print(f"Train dir: {os.path.join(os.path.join(args.audio_path, dir), 'train')}")
    src = os.path.join(os.path.join(args.audio_path, dir), '*.wav')
    dest = os.path.join(os.path.join(args.audio_path, dir), 'train')
    os.system(f"mv {src} {dest}")
