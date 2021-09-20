import os, sys
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
parser.add_argument('dir_path', type=str, help='Path to the main directory of videos')
parser.add_argument('output_path', type=str, help="Path to save outputted frames")
parser.add_argument('--specific_vid', type=str, default=None, help="Specify path to a specific video, if only want to plot for one video")
parser.add_argument('--min_detect_conf', default=0.5, type=float, help='Minimum confidence for the hand landmarks detection. Defaults to 0.5')
parser.add_argument('--min_tracking_conf', default=0.5, type=float, help="Minimum confidence for tracking hands in video. Defaults to 0.5")
parser.add_argument('--no_plot', action='store_true', help='Include this flag to only output the landmarks without plotting')
parser.add_argument('--no_coords', action='store_true', help="Include this flag to not save coordinates")
parser.add_argument('--stats', action='store_true', help="Include this flag to get statistics about the data set for landmarks")
args = parser.parse_args()

print(f'Args: {args}')

# Set up solution for tracking hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
#mp_drawing_styles = mp.solutions.drawing_styles

# For each video, count number of frames that have less than 2 hands detected
err_hands = {}
no_hands = {} # Count when no hands are detected
three_hands = {}
total_num_frames = 0


def handle_multiple_detected_hands(hand_detection_results):
    """Determines which two detected hands should be returned to have the coordinates saved"""
    num_results = len(hand_detection_results.multi_handedness)
    verify = len(hand_detection_results.multi_hand_landmarks)
    assert(num_results == verify)

    # If there is only one hand detected
    if len(hand_detection_results.multi_hand_landmarks) == 1:
        handedness = hand_detection_results.multi_handedness[0].classification[0].label
        results = []
        results.append((handedness, hand_detection_results.multi_hand_landmarks[0]))
        return results

    left_scores = {'max': 0, 'min': float('inf')}
    right_scores = {'max': 0, 'min': float("inf")}
    results = [] # Returns a list of tuples (handedness, hand_landmarks). Should have only 2 elements
    for i in range(num_results):
        handedness = hand_detection_results.multi_handedness[i].classification[0].label
        score = hand_detection_results.multi_handedness[i].classification[0].score
        if handedness == "Left" and score > left_scores['max']:
            left_scores['max'] = score
            left = hand_detection_results.multi_hand_landmarks[i]
        
        if handedness == "Right" and score > right_scores['max']:
            right_scores['max'] = score
            right = hand_detection_results.multi_hand_landmarks[i]
        
        if handedness == "Right" and score < right_scores['min']:
            right_scores['min'] = score
            left_backup = hand_detection_results.multi_hand_landmarks[i]
        
        if handedness == "Left" and score < left_scores['min']:
            left_scores['min'] = score
            right_backup = hand_detection_results.multi_hand_landmarks[i]
    
    # Choose which two landmarks should be used
    #print("left scores: ", left_scores)
    #print("right scores: ", right_scores)
    if left_scores['max'] > 0:
        results.append(('Left', left))
    else:
        results.append(("Left", left_backup))

    if right_scores['max'] > 0:
        results.append(("Right", right))
    else:
        results.append(("Right", right_backup))
    return results


# For static images, get the images
videos = os.listdir(args.dir_path)
for video in tqdm(videos):
    if args.specific_vid is not None and args.specific_vid+'.mp4' != video:
        continue

    # Get the images/frames from each video
    print("Getting landmarks for video: ", video)
    cap = cv2.VideoCapture(os.path.join(args.dir_path, video))
    print(f"Frame rate: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"Num frames in video: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
    
    # Regnet optical flow extraction steps
    _, prev = cap.read()
    num = 1
    prev = cv2.resize(prev, (340, 256))
    #cv2.imwrite(P.join(save_dir, f"img_{num:05d}.jpg"), prev)
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=args.min_detect_conf,
            min_tracking_confidence=args.min_tracking_conf) as hands:

        err_hands[video] = 0
        no_hands[video] = 0
        three_hands[video] = 0

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame")
                # If loading a video, use 'break' instead of 'continue'.
                break
            
            image = cv2.resize(image, (340, 256))
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Set up dictionary to save the landmarks and image width, height
            coordinates = {}
            total_num_frames += 1

            # 3 coordinates, for 21 landmarks each hand, so total of 42 landmarks
            coordinates['landmarks'] = np.zeros((42,3)) # Left hand should be top 21 rows, right bottom 21 rows

            # Draw hand landmarks on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            #print("Working on image: ", file)
            #title = file.split("/")[-1].split('.')[0]
            #name = file.split("/")[-1].split('.')[0] + '_landmarks_vid.jpg'
            #print(f"Handedness: {results.multi_handedness}")
            
            image_height, image_width, _ = image.shape

            # Save the image height and width
            coordinates['img_height'] = image_height
            coordinates['img_width'] = image_width

            #annotated_image = image.copy()
            if results.multi_hand_landmarks:
                # Iterating through each hand
                #print(f'Num hands: {len(results.multi_hand_landmarks)}')
                if len(results.multi_hand_landmarks) < 2:
                    err_hands[video] += 1

                if len(results.multi_hand_landmarks) > 2:
                    #print("LOL what is happening: ", video)
                    #print("num hands: ", len(results.multi_hand_landmarks))
                    #print(f"Hand landmarks: {results.multi_hand_landmarks}")
                    #print(f"Handedness: {results.multi_handedness}")
                    # Identify which is the extra hand to exclude
                    #import pdb; pdb.set_trace()
                    three_hands[video] += 1

                results_to_use = handle_multiple_detected_hands(results)
                #print("Number of usable results: ", len(results_to_use))

                if args.stats:
                    continue

                #for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                for i, (handedness, hand_landmarks) in enumerate(results_to_use):
                    #print(f"Hand landmarks: {hand_landmarks}")
                    # Match the left hand to first 21 rows and right hand to lower 21 rows
                    # results.multi_handedness[0].classification[0].label
                    #handedness = results.multi_handedness[i].classification[0].label
                    if handedness == "Left":
                        row = 0
                    else:
                        row = 1

                    if not args.stats and not args.no_coords:
                        for j,key in enumerate(HAND_LANDMARKS): # 21 landmarks per hand
                            # Save the landmarks of the image in object file
                            obj = getattr(mp_hands.HandLandmark, key)
                            index = j + (row * 21)
                            coordinates['landmarks'][index,:] = np.array([hand_landmarks.landmark[obj].x, hand_landmarks.landmark[obj].y, hand_landmarks.landmark[obj].z])

                    if not args.no_plot and not args.stats:
                        # Draw on the image
                        #mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                no_hands[video] += 1

            # Saving plotted images
            if not args.no_plot and not args.stats:
                #cv2.imwrite(os.path.join(f'data/features/ASMR/asmr_both_vids/OF_10s_21.5fps/{video}/', name), cv2.flip(annotated_image, 1))
                if not os.path.isdir(args.output_path):
                    os.makedirs(args.output_path, exist_ok=True)

                if args.specific_vid is None:
                    title = video.split(".")[0]
                    os.makedirs(os.path.join(args.output_path, f"{title}/"), exist_ok=True)
                    #title = file.split("/")[-1].split('.')[0]
                    #name = file.split("/")[-1].split('.')[0] + '_landmarks_vid.jpg'
                    cv2.imwrite(os.path.join(os.path.join(args.output_path, f'{title}/'), f"img_{num:05d}_landmarks_vid.jpg"), cv2.flip(image, 1))
                else:
                    cv2.imwrite(os.path.join(args.output_path, f'img_{num:05d}_landmarks_vid.jpg'), cv2.flip(image, 1))
            
            # Saving the coordinate info
            if not args.stats and not args.no_coords:
                if not os.path.isdir(args.output_path):
                    os.makedirs(args.output_path, exist_ok=True)

                if args.specific_vid is not None:
                    print(f"Saving to: " + os.path.join(args.output_path, f'img_{num:05d}_vid.pt'))
                    torch.save(coordinates, os.path.join(args.output_path, f'img_{num:05d}_vid.pt'))
                else:
                    title = video.split(".")[0]
                    os.makedirs(os.path.join(args.output_path, f"{title}/"), exist_ok=True)
                    print(f"Saving to: " + os.path.join(os.path.join(args.output_path, f'{title}/'), f'img_{num:05d}_vid.pt'))
                    torch.save(coordinates, os.path.join(os.path.join(args.output_path, f'{title}/'), f'img_{num:05d}_vid.pt'))
            num += 1
    cap.release()

# Print out stats
no_hand = 0
one_hand = 0
multi_hand = 0
for key in no_hands:
    no_hand += no_hands[key]

for key in err_hands:
    one_hand += err_hands[key]

for key in three_hands:
    multi_hand += three_hands[key]
print(f"Frames without any hands detected: {no_hand} out of {total_num_frames}")
print(f"Frames with only one hand detected: {one_hand} out of {total_num_frames}")
print(f"Frames with more than two hands detected: {multi_hand} out of {total_num_frames}")

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
