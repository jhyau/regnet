import cv2
from glob import glob
import os
import os.path as P
import argparse
from multiprocessing import Pool
from functools import partial
import numpy as np

def cal_for_frames(video_path, output_dir, width, height):
    save_dir = P.join(output_dir, P.basename(video_path).split('.')[0])
    os.makedirs(save_dir, exist_ok=True)
    video = cv2.VideoCapture(video_path)
    _, prev = video.read()
    num = 1
    prev = cv2.resize(prev, (width, height))
    cv2.imwrite(P.join(save_dir, f"img_{num:05d}.jpg"), prev)
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    while video.isOpened():
        reg, curr = video.read()
        if not reg:
            break
        curr = cv2.resize(curr, (width, height))
        cv2.imwrite(P.join(save_dir, f"img_{num:05d}.jpg"), curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        flow = compute_TVL1(prev, curr)
        cv2.imwrite(P.join(save_dir, f"flow_x_{num:05d}.jpg"), flow[:, :, 0])
        cv2.imwrite(P.join(save_dir, f"flow_y_{num:05d}.jpg"), flow[:, :, 1])
        prev = curr
        num += 1
    if num < 215:
        print(video_path)
    
def compute_TVL1(prev, curr, bound=20):
    """Compute the TV-L1 optical flow."""
    #TVL1 = cv2.DualTVL1OpticalFlow_create()
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    flow[flow>bound]=bound
    flow[flow<-bound]=-bound
    flow-=-bound
    flow*=(255/float(2*bound))
    return flow

def check_video(video_paths, output_dir, input_dir):
    # Only keep video paths that aren't done yet
    if not os.path.exists(output_dir):
        return video_paths

    directories = os.listdir(output_dir)
    done_vids = []
    # Expect 216 images for each video (21.5 fps)
    for dirs in directories:
        rgbs = len(glob(os.path.join(dirs, "img*.jpg"))) - len(glob(os.path.join(dirs, "img*_landmarks.jpg")))
        flows_x = len(glob(os.path.join(dirs, "flow_x*.jpg")))
        flows_y = len(glob(os.path.join(dirs, "flow_y*.jpg")))
        
        if (flows_x == flows_y) and (flows_x == rgbs) and (flows_y == rgbs) and (rgbs == 216):
            done_vids.append(os.path.join(input_dir, os.path.join(dirs, '.mp4')))
    return done_vids

if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument("-i", "--input_dir", default="data/features/dog/videos_10s_21.5fps")
    paser.add_argument("-o", "--output_dir", default="data/features/dog/OF_10s_21.5fps")
    paser.add_argument("-w", "--width", type=int, default=340)
    paser.add_argument("-g", "--height", type=int, default=256)
    paser.add_argument("-n", '--num_worker', type=int, default=16)

    args = paser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    width = args.width
    height = args.height

    print("args for extracting rgb and optical flow features: ", args)

    video_paths = glob(P.join(input_dir, "*.mp4"))
    print("All videos: ", video_paths)

    # Check if some directories are already done
    done_vids = check_video(video_paths, args.output_dir, args.input_dir)
    video_paths = video_paths - done_vids
    print("finished vids: ", don_vids)
    print("Remaining videos: ", video_paths)

    video_paths.sort()
    with Pool(args.num_worker) as p:
        p.map(partial(cal_for_frames, output_dir=output_dir, 
                    width=width, height=height), video_paths)

    print("Optical flow extraction completed")
