import os, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw

IMG_WIDTH = 340
IMG_HEIGHT = 256

# Get the skeletal hand landmarks
landmarks = np.load('data/features/ASMR/asmr_both_vids/ASMR_Addictive_Tapping_1_Hr_No_Talking_skeletal/output-landmarks.npz', allow_pickle=True)

# Load the first two RGB frames of the video
#frame1 = cv2.imread('data/features/ASMR/asmr_both_vids/OF_10s_21.5fps/ASMR_Addictive_Tapping_1_Hr_No_Talking-1-of-365/img_00001.jpg')
#frame2 = cv2.imread('data/features/ASMR/asmr_both_vids/OF_10s_21.5fps/ASMR_Addictive_Tapping_1_Hr_No_Talking-1-of-365/img_00002.jpg')

#frame1 = Image.open('data/features/ASMR/asmr_both_vids/OF_10s_21.5fps/ASMR_Addictive_Tapping_1_Hr_No_Talking-1-of-365/img_00008.jpg')
#frame2 = Image.open('data/features/ASMR/asmr_both_vids/OF_10s_21.5fps/ASMR_Addictive_Tapping_1_Hr_No_Talking-1-of-365/img_00009.jpg')

# Get the frames' landmark coordinates, dims of (42, 3)
print(f"Number of saved frames from the landmarks: {landmarks['arr_0'].shape}")
frame1_raw = landmarks['arr_0'][29]
frame2_raw = landmarks['arr_0'][30]

# Get the original image frame, before resizing
#video = cv2.VideoCapture('data/features/ASMR/asmr_both_vids/videos_10s/ASMR_Addictive_Tapping_1_Hr_No_Talking-1-of-365.mp4')
video = cv2.VideoCapture('data/ASMR/orig_full_asmr_videos/ASMR_Addictive_Tapping_1_Hr_No_Talking.mp4')
frame_count = 0
frame_rate = video.get(cv2.CAP_PROP_FPS)
img_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
img_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT) # ID: 7
#img_width = video.get(3)
#img_height = video.get(4)
print(f'width: {img_width} and height: {img_height}')
print(f'Total number of frames in video: {num_frames}')
print(f'FPS: {frame_rate}')

frame1_points = []
for i in range(frame1_raw.shape[0]):
    frame1_points.append((frame1_raw[i,0]*img_width, frame1_raw[i,1]*img_height))

frame2_points=[]
for i in range(frame2_raw.shape[0]):
    frame2_points.append((frame2_raw[i,0]*img_width, frame2_raw[i,1]*img_height))

while video.isOpened():
    reg, curr = video.read()
    if frame_count == 29:
        curr = Image.fromarray(curr)
        #with Image.open('data/features/ASMR/asmr_both_vids/OF_10s_21.5fps/ASMR_Addictive_Tapping_1_Hr_No_Talking-1-of-365/img_00030.jpg') as im:
        draw = ImageDraw.Draw(curr)
        draw.point(frame1_points)
        curr.save('frame30_landmarks.png', "PNG")

    if frame_count == 30:
        curr = Image.fromarray(curr)
        #with Image.open('data/features/ASMR/asmr_both_vids/OF_10s_21.5fps/ASMR_Addictive_Tapping_1_Hr_No_Talking-1-of-365/img_00031.jpg') as im:
        draw = ImageDraw.Draw(curr)
        draw.point(frame2_points)
        curr.save('frame31_landmarks.png', "PNG")
        break
    frame_count += 1

print("Done")
