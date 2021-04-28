import os
import os.path as P
from glob import glob
import argparse

if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument("-i", "--input_dir", default="dataset/VEGAS/dog/videos")
    paser.add_argument("-o", "--output_dir", default="filelists")
    paser.add_argument("-p", "--prefix", default="dog", choices=["dog", "fireworks", "baby", "drum", "gun", "sneeze", "cough", "hammer", "ASMR_1_Hr", "ASMR_3_Hrs"])
    args = paser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    video_paths = glob(P.join(input_dir, "*.mp4"))
    if args.prefix == 'ASMR_1_Hr':
        video_paths.sort(key=lambda x: int(x.split('_')[-1][-1].split(".")[0]))
    else:
        video_paths.sort(key=lambda x: int(x.split('_')[-1][-1].split(".")[0]))
    
    if args.prefix in ["dog", "fireworks", "baby", "drum"]:
        n_test = 128
    elif args.prefix in ["ASMR_1_Hr"]:
        n_test = 2
    elif args.prefix in ["ASMR_3_Hrs"]:
        n_test = 4
    else:
        n_test = 64

    with open(P.join(output_dir, args.prefix+"_train.txt"), 'w') as f:
        for video_path in video_paths[:-n_test]:
            f.write(f"{os.path.basename(video_path).split('.')[0]}\n")
    f.close()

    with open(P.join(output_dir, args.prefix+"_test.txt"), 'w') as f:
        for video_path in video_paths[-n_test:]:
            f.write(f"{os.path.basename(video_path).split('.')[0]}\n")
    f.close()
    print("Done splitting train and test")
