import sys, os
import numpy as np
import mediapipe as mp
import cv2
import argparse
from PIL import Image, ImageDraw
from glob import glob
from tqdm import tqdm
import torch

IMG_WIDTH = 340
IMG_HEIGHT = 256

# Landmarks
HAND_LANDMARKS = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
        'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
        'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']

# Argparser
parser = argparse.ArgumentParser()
parser.add_argument('base_image', type=str, help='Path to base image to plot saved landmark coordinates on')
parser.add_argument('landmark', type=str, help='Path to landmarks coordinates file (.pt)')
parser.add_argument('--save_path', type=str, default='.', help='Path to save the plotting verification. Defaults to current working directory')
#parser.add_argument('--no_plot', action='store_true', help='Include this flag to only output the landmarks without plotting')
args = parser.parse_args()

print(f'Args: {args}')


# Get the skeletal hand landmarks (keys: 'landmarks', 'img_width', 'img_height')
#landmarks = np.load('data/features/ASMR/asmr_both_vids/ASMR_Addictive_Tapping_1_Hr_No_Talking_skeletal/output-landmarks.npz', allow_pickle=True)
landmarks = torch.load(args.landmark)

# Load the image
#img = cv2.imread(args.base_image)
img = Image.open(args.base_image)

# Load the first two RGB frames of the video
#frame1 = cv2.imread('data/features/ASMR/asmr_both_vids/OF_10s_21.5fps/ASMR_Addictive_Tapping_1_Hr_No_Talking-1-of-365/img_00001.jpg')
#frame2 = cv2.imread('data/features/ASMR/asmr_both_vids/OF_10s_21.5fps/ASMR_Addictive_Tapping_1_Hr_No_Talking-1-of-365/img_00002.jpg')

#frame1 = Image.open('data/features/ASMR/asmr_both_vids/OF_10s_21.5fps/ASMR_Addictive_Tapping_1_Hr_No_Talking-1-of-365/img_00008.jpg')
#frame2 = Image.open('data/features/ASMR/asmr_both_vids/OF_10s_21.5fps/ASMR_Addictive_Tapping_1_Hr_No_Talking-1-of-365/img_00009.jpg')

# Get the frames' landmark coordinates, dims of (42, 3)
print(f"Landmark shape: {landmarks['landmarks'].shape}")
#print(f"Number of saved frames from the landmarks: {landmarks['arr_0'].shape}")
#frame1_raw = landmarks['arr_0'][29]
#frame2_raw = landmarks['arr_0'][30]

print(f'width: {landmarks["img_width"]} and height: {landmarks["img_height"]}')
#print(f'Total number of frames in video: {num_frames}')
#print(f'FPS: {frame_rate}')

# Get the coordinates for the plot
frame1_points = []
for i in range(landmarks['landmarks'].shape[0]):
    frame1_points.append((landmarks['landmarks'][i,0]*landmarks['img_width'], landmarks['landmarks'][i,1]*landmarks['img_height']))


# Plot the coordinates
draw = ImageDraw.Draw(img)
draw.point(frame1_points)
name = args.base_image.split('/')[-1].split('.')[0]
img.save(os.path.join(args.save_path, name+'_verify.png'), "PNG")

#frame2_points=[]
#for i in range(frame2_raw.shape[0]):
#    frame2_points.append((frame2_raw[i,0]*img_width, frame2_raw[i,1]*img_height))

#while video.isOpened():
#    reg, curr = video.read()
#    if frame_count == 29:
#        curr = Image.fromarray(curr)
        #with Image.open('data/features/ASMR/asmr_both_vids/OF_10s_21.5fps/ASMR_Addictive_Tapping_1_Hr_No_Talking-1-of-365/img_00030.jpg') as im:
#        draw = ImageDraw.Draw(curr)
#        draw.point(frame1_points)
#        curr.save('frame30_landmarks.png', "PNG")

#    if frame_count == 30:
#        curr = Image.fromarray(curr)
        #with Image.open('data/features/ASMR/asmr_both_vids/OF_10s_21.5fps/ASMR_Addictive_Tapping_1_Hr_No_Talking-1-of-365/img_00031.jpg') as im:
#        draw = ImageDraw.Draw(curr)
#        draw.point(frame2_points)
#        curr.save('frame31_landmarks.png', "PNG")
#        break
                                                                                                                                                                                   101,104       96%

