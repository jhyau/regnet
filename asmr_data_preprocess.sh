
#soundlist=" "dog" "fireworks" "drum" "baby" "gun" "sneeze" "cough" "hammer" "
soundtype="ASMR_3_Hrs_fake_audio"
dir_name="fake_audio_Ultimate_Tapping_ASMR_3_Hours_subset"
duration="15"
#for soundtype in $soundlist 
#do

# Data preprocessing. We will first pad all videos to 10s and change video FPS and audio
# sampling rate.
python extract_audio_and_video.py \
-i data/ASMR/${dir_name} \
-o data/features/ASMR/${dir_name}

# Generating RGB frame and optical flow. This script uses CPU to calculate optical flow,
# which may take a very long time. We strongly recommend you to refer to TSN repository 
# (https://github.com/yjxiong/temporal-segment-networks) to speed up this process.
#python extract_rgb_flow.py \
#-i data/features/ASMR/${dir_name}/videos_${duration}s_21.5fps \
#-o data/features/ASMR/${dir_name}/OF_${duration}s_21.5fps

#Split training/testing list
python gen_list.py \
-i data/ASMR/${dir_name}  \
-o filelists --prefix ${soundtype}

#Extract Mel-spectrogram from audio
python extract_mel_spectrogram.py \
-i data/features/ASMR/${dir_name}/audio_${duration}s_22050hz \
-o data/features/ASMR/${dir_name}/melspec_${duration}s_22050hz

#Extract RGB feature
CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
-t filelists/${soundtype}_train.txt \
-m RGB \
-i data/features/ASMR/${dir_name}/OF_${duration}s_21.5fps \
-o data/features/ASMR/${dir_name}/feature_rgb_bninception_dim1024_21.5fps

CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
-t filelists/${soundtype}_test.txt \
-m RGB \
-i data/features/ASMR/${dir_name}/OF_${duration}s_21.5fps \
-o data/features/ASMR/${dir_name}/feature_rgb_bninception_dim1024_21.5fps

#Extract optical flow feature
CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
-t filelists/${soundtype}_train.txt \
-m Flow \
-i data/features/ASMR/${dir_name}/OF_${duration}s_21.5fps \
-o data/features/ASMR/${dir_name}/feature_flow_bninception_dim1024_21.5fps

CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
-t filelists/${soundtype}_test.txt \
-m Flow \
-i data/features/ASMR/${dir_name}/OF_${duration}s_21.5fps \
-o data/features/ASMR/${dir_name}/feature_flow_bninception_dim1024_21.5fps

#done
