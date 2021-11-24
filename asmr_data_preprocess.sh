
#soundlist=" "dog" "fireworks" "drum" "baby" "gun" "sneeze" "cough" "hammer" "
soundtype="asmr_both_vids"
dir_name="asmr_both_vids1"
duration="10"
duration_num=10
mel_duration_num=$((44100 * duration_num))
echo $mel_duration_num
audio_sample_rate=44100
video_fps=21.5
filter_length=1024
hop_length=256
win_length=1024
fmin=0.0
fmax=16000.0
vocoder="waveglow"
num_mel_samples=1720
# Need a '-' here as well.
speed_suffix=-1.0
echo $duration
echo $audio_sample_rate
echo $video_fps
#for soundtype in $soundlist 
#do

# Data preprocessing. We will first pad all videos to 10s and change video FPS and audio
# sampling rate.
python extract_audio_and_video.py \
-i "data/ASMR/ASMR_Addictive_Tapping_1_Hrx1.0/" \
-o data/features/ASMR/${dir_name} \
-d ${duration_num} \
-a ${audio_sample_rate} \
-v ${video_fps} || exit
# -i data/ASMR/${dir_name} \

# Generating RGB frame and optical flow. This script uses CPU to calculate optical flow,
# which may take a very long time. We strongly recommend you to refer to TSN repository 
# (https://github.com/yjxiong/temporal-segment-networks) to speed up this process.
#python extract_rgb_flow.py \
#-i data/features/ASMR/${dir_name}/videos_${duration}s_${video_fps}fps \
#-o data/features/ASMR/${dir_name}/OF_${duration}s_${video_fps}fps || exit

#Split training/testing list
#python gen_list.py \
#-i data/ASMR/${dir_name}  \
#-o filelists --prefix ${soundtype}

#Extract Mel-spectrogram from audio
#python extract_mel_spectrogram.py \
#-i data/features/ASMR/${dir_name}/audio_${duration}s_${audio_sample_rate}hz \
#-o data/features/ASMR/${dir_name}/melspec_${duration}s_${audio_sample_rate}hz \
#-l $mel_duration_num \
#--vocoder ${vocoder} \
#--sampling_rate $audio_sample_rate \
#--fmin $fmin \
#--fmax $fmax \
#--filter_length $filter_length \
#--hop_length $hop_length \
#--win_length $win_length \
#--mel_samples $num_mel_samples || exit


#Extract RGB feature
#CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
#-t filelists/${soundtype}_train${speed_suffix}.txt \
#-m RGB \
#-i data/features/ASMR/${dir_name}/OF_${duration}s_${video_fps}fps \
#-o data/features/ASMR/${dir_name}/feature_rgb_bninception_dim1024_${video_fps}fps

#CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
#-t filelists/${soundtype}_test${speed_suffix}.txt \
#-m RGB \
#-i data/features/ASMR/${dir_name}/OF_${duration}s_${video_fps}fps \
#-o data/features/ASMR/${dir_name}/feature_rgb_bninception_dim1024_${video_fps}fps || exit

#CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
#-t filelists/${soundtype}_true_test${speed_suffix}.txt \
#-m RGB \
#-i data/features/ASMR/${dir_name}/OF_${duration}s_${video_fps}fps \
#-o data/features/ASMR/${dir_name}/feature_rgb_bninception_dim1024_${video_fps}fps

#Extract RGB+landmarks feature
#CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
#-t filelists/${soundtype}_train${speed_suffix}.txt \
#-m RGB_landmarks \
#-i data/features/ASMR/${dir_name}/OF_${duration}s_${video_fps}fps \
#-o data/features/ASMR/${dir_name}/feature_rgb_landmarks_bninception_dim1024_${video_fps}fps

#CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
#-t filelists/${soundtype}_test${speed_suffix}.txt \
#-m RGB_landmarks \
#-i data/features/ASMR/${dir_name}/OF_${duration}s_${video_fps}fps \
#-o data/features/ASMR/${dir_name}/feature_rgb_landmarks_bninception_dim1024_${video_fps}fps

#CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
#-t filelists/${soundtype}_true_test${speed_suffix}.txt \
#-m RGB_landmarks \
#-i data/features/ASMR/${dir_name}/OF_${duration}s_${video_fps}fps \
#-o data/features/ASMR/${dir_name}/feature_rgb_bninception_dim1024_${video_fps}fps


#Extract optical flow feature
#CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
#-t filelists/${soundtype}_train${speed_suffix}.txt \
#-m Flow \
#-i data/features/ASMR/${dir_name}/OF_${duration}s_${video_fps}fps \
#-o data/features/ASMR/${dir_name}/feature_flow_bninception_dim1024_${video_fps}fps

#CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
#-t filelists/${soundtype}_test${speed_suffix}.txt \
#-m Flow \
#-i data/features/ASMR/${dir_name}/OF_${duration}s_${video_fps}fps \
#-o data/features/ASMR/${dir_name}/feature_flow_bninception_dim1024_${video_fps}fps || exit

#CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
#-t filelists/${soundtype}_true_test${speed_suffix}.txt \
#-m Flow \
#-i data/features/ASMR/${dir_name}/OF_${duration}s_${video_fps}fps \
#-o data/features/ASMR/${dir_name}/feature_flow_bninception_dim1024_${video_fps}fps

#done
