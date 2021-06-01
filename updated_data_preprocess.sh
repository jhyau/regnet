
#soundlist=" "dog" "fireworks" "drum" "baby" "gun" "sneeze" "cough" "hammer" "
soundtype="dog"
dir_name="dog"
duration="10"
duration_num=10
mel_duration_num=$((44100 * duration_num))
echo $mel_duration_num
audio_sample_rate=44100
num_mel_samples=1720
echo $duration
echo $audio_sample_rate
#for soundtype in $soundlist 
#do

# Data preprocessing. We will first pad all videos to 10s and change video FPS and audio
# sampling rate.
python extract_audio_and_video.py \
-i data/VAS/${dir_name}/videos \
-o data/features/${dir_name}_${audio_sample_rate} \
-d ${duration_num} \
-a ${audio_sample_rate}

# Generating RGB frame and optical flow. This script uses CPU to calculate optical flow,
# which may take a very long time. We strongly recommend you to refer to TSN repository 
# (https://github.com/yjxiong/temporal-segment-networks) to speed up this process.
#python extract_rgb_flow.py \
#-i data/features/${dir_name}/videos_${duration}s_21.5fps \
#-o data/features/${dir_name}/OF_${duration}s_21.5fps

#Split training/testing list
#python gen_list.py \
#-i data/features/${dir_name}  \
#-o filelists --prefix ${soundtype}

#Extract Mel-spectrogram from audio
python extract_mel_spectrogram.py \
-i data/features/${dir_name}_${audio_sample_rate}/audio_${duration}s_${audio_sample_rate}hz \
-o data/features/${dir_name}_${audio_sample_rate}/melspec_${duration}s_${audio_sample_rate}hz \
-l $mel_duration_num \
--mel_samples ${num_mel_samples}

#Extract RGB feature
#CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
#-t filelists/${soundtype}_train.txt \
#-m RGB \
#-i data/features/${dir_name}/OF_${duration}s_21.5fps \
#-o data/features/${dir_name}/feature_rgb_bninception_dim1024_21.5fps

#CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
#-t filelists/${soundtype}_test.txt \
#-m RGB \
#-i data/features/${dir_name}/OF_${duration}s_21.5fps \
#-o data/features/${dir_name}/feature_rgb_bninception_dim1024_21.5fps

#Extract optical flow feature
#CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
#-t filelists/${soundtype}_train.txt \
#-m Flow \
#-i data/features/${dir_name}/OF_${duration}s_21.5fps \
#-o data/features/${dir_name}/feature_flow_bninception_dim1024_21.5fps

#CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
#-t filelists/${soundtype}_test.txt \
#-m Flow \
#-i data/features/${dir_name}/OF_${duration}s_21.5fps \
#-o data/features/${dir_name}/feature_flow_bninception_dim1024_21.5fps

#done
