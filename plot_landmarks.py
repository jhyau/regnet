import os, sys
import numpy as np
import mediapipe as mp
import cv2
import argparse
from PIL import Image, ImageDraw
from glob import glob
from tqdm import tqdm

IMG_WIDTH = 340
IMG_HEIGHT = 256


# Argparser
parser = argparse.ArgumentParser()
parser.add_argument('dir_path', type=str, help='Path to the main directory of optical flow extractions')
parser.add_argument('--min_conf', default=0.5, type=float, help='Minimum confidence for the hand landmarks detection. Defaults to 0.5')
args = parser.parse_args()

print(f'Args: {args}')

# Set up solution for tracking hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For static images, get the images
videos = os.listdir(args.dir_path)
for video in tqdm(videos):
    # Get the images/frames from each video
    print("Getting landmarks for video: ", video)
    #all_images = os.listdir(f"data/features/ASMR/asmr_both_vids/OF_10s_21.5fps/{video}")
    rgb_images = glob(os.path.join(args.dir_path, f"{video}/img*.jpg"))
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=args.min_conf) as hands:
        for idx, file in enumerate(rgb_images):
            # Read an image, flip it around y-axis for correct handedness output
            image = cv2.flip(cv2.imread(file), 1)

            # Make sure loaded image is RGB after getting results of tracking hands
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Draw hand landmarks on the image
            #print("Working on image: ", file)
            name = file.split("/")[-1].split('.')[0] + '_landmarks.jpg'
            #print(f"Handedness: {results.multi_handedness}")
            #if not results.multi_hand_landmarks:
                # No results detected (or below min confidence probably)
                #continue
            
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    #print(f"Hand landmarks: {hand_landmarks}")
                    # Draw on the image
                    mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.imwrite(os.path.join(f'data/features/ASMR/asmr_both_vids/OF_10s_21.5fps/{video}/', name), cv2.flip(annotated_image, 1))

# Get the skeletal hand landmarks
#landmarks = np.load('data/features/ASMR/asmr_both_vids/ASMR_Addictive_Tapping_1_Hr_No_Talking_skeletal/output-landmarks.npz', allow_pickle=True)

# Load the first two RGB frames of the video
#frame1 = cv2.imread('data/features/ASMR/asmr_both_vids/OF_10s_21.5fps/ASMR_Addictive_Tapping_1_Hr_No_Talking-1-of-365/img_00001.jpg')
#frame2 = cv2.imread('data/features/ASMR/asmr_both_vids/OF_10s_21.5fps/ASMR_Addictive_Tapping_1_Hr_No_Talking-1-of-365/img_00002.jpg')

#frame1 = Image.open('data/features/ASMR/asmr_both_vids/OF_10s_21.5fps/ASMR_Addictive_Tapping_1_Hr_No_Talking-1-of-365/img_00008.jpg')
#frame2 = Image.open('data/features/ASMR/asmr_both_vids/OF_10s_21.5fps/ASMR_Addictive_Tapping_1_Hr_No_Talking-1-of-365/img_00009.jpg')

# Get the frames' landmark coordinates, dims of (42, 3)
#print(f"Number of saved frames from the landmarks: {landmarks['arr_0'].shape}")
#frame1_raw = landmarks['arr_0'][29]
#frame2_raw = landmarks['arr_0'][30]

# Get the original image frame, before resizing
#video = cv2.VideoCapture('data/features/ASMR/asmr_both_vids/videos_10s/ASMR_Addictive_Tapping_1_Hr_No_Talking-1-of-365.mp4')
#video = cv2.VideoCapture('data/ASMR/orig_full_asmr_videos/ASMR_Addictive_Tapping_1_Hr_No_Talking.mp4')
#frame_count = 0
#frame_rate = video.get(cv2.CAP_PROP_FPS)
#img_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
#img_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
#num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT) # ID: 7
#img_width = video.get(3)
#img_height = video.get(4)
#print(f'width: {img_width} and height: {img_height}')
#print(f'Total number of frames in video: {num_frames}')
#print(f'FPS: {frame_rate}')

#frame1_points = []
#for i in range(frame1_raw.shape[0]):
#    frame1_points.append((frame1_raw[i,0]*img_width, frame1_raw[i,1]*img_height))

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
#    frame_count += 1

print("Done")
