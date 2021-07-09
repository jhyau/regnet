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

frame1 = Image.open('data/features/ASMR/asmr_both_vids/OF_10s_21.5fps/ASMR_Addictive_Tapping_1_Hr_No_Talking-1-of-365/img_00008.jpg')
frame2 = Image.open('data/features/ASMR/asmr_both_vids/OF_10s_21.5fps/ASMR_Addictive_Tapping_1_Hr_No_Talking-1-of-365/img_00009.jpg')

# Get the frames' landmark coordinates, dims of (42, 3)
frame1_raw = landmarks['arr_0'][29]
frame2_raw = landmarks['arr_0'][30]


frame1_points = []
for i in range(frame1_raw.shape[0]):
    frame1_points.append((frame1_raw[i,0]*IMG_WIDTH, frame1_raw[i,1]*IMG_HEIGHT))

with Image.open('data/features/ASMR/asmr_both_vids/OF_10s_21.5fps/ASMR_Addictive_Tapping_1_Hr_No_Talking-1-of-365/img_00030.jpg') as im:
    draw = ImageDraw.Draw(im)
    draw.point(frame1_points)
    im.save('frame30_landmarks.png', "PNG")

frame2_points=[]
for i in range(frame2_raw.shape[0]):
    frame2_points.append((frame2_raw[i,0]*IMG_WIDTH, frame2_raw[i,1]*IMG_HEIGHT))

with Image.open('data/features/ASMR/asmr_both_vids/OF_10s_21.5fps/ASMR_Addictive_Tapping_1_Hr_No_Talking-1-of-365/img_00031.jpg') as im:
    draw = ImageDraw.Draw(im)
    draw.point(frame2_points)
    im.save('frame31_landmarks.png', "PNG")

print("Done")
