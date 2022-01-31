from yacs.config import CfgNode as CN
import math

_C  =  CN()
_C.epochs = 2 # Use this one for tiny experiments
#_C.epochs = 1000
_C.num_epoch_save = 10
_C.seed = 123
_C.dynamic_loss_scaling = True
_C.dist_backend = "nccl"
_C.dist_url = "tcp://localhost:54321"
_C.cudnn_enabled = True
_C.cudnn_benchmark = False
# some metadata to keep track of the experiments.
_C.exp_number = -1
_C.exp_prefix = 'exp'
_C.exp_tag = 'This is the default tag. Overwrite to keep track what you are testing'
_C.save_dir = 'ckpt/dog'
_C.video_dir = "/juno/u/jyau/regnet/data/features/ASMR/asmr_both_vids/videos_10s"
# Note: |exp_number| is used to create/reuse a directory within |save_dir|. -1 will generate a new
# with the first open number.

_C.checkpoint_path = ''
_C.epoch_count = 0
_C.exclude_dirs = ['ckpt', 'data']
_C.training_files = 'filelists/asmr_materials_1mat_train.txt'  #'filelists/dog_train.txt'
_C.test_files = 'filelists/asmr_materials_1mat_test.txt'  #'filelists/dog_test.txt'
_C.rgb_feature_dir = "/juno/u/jyau/regnet/data/features/ASMR/asmr_both_vids/feature_rgb_bninception_dim1024_21.5fps"  #"data/features/dog/feature_rgb_bninception_dim1024_21.5fps"
# New: landmark feature dir
_C.landmark_feature_dir = None
_C.flow_feature_dir = "/juno/u/jyau/regnet/data/features/ASMR/asmr_both_vids/feature_flow_bninception_dim1024_21.5fps"  #"data/features/dog/feature_flow_bninception_dim1024_21.5fps"
_C.mel_dir = "/juno/u/jyau/regnet/data/features/ASMR/asmr_both_vids/melspec_10s_44100hz"  #"data/features/dog/melspec_10s_22050hz"
_C.video_samples = 215 # Number of frames in each video. First dim of the feature vectors
_C.audio_samples = 10  # Number of seconds of audio
_C.mel_samples = 1720 # Original mel samples to match 22050 audio sampling rate: 860
_C.visual_dim = 2048  # Feature vector length. 1024 (rgb) + 1024 (flow) = 2048. becomes 3072 if include landmark feature vector
_C.n_mel_channels = 80
# This might be inaccurate if they don't divide cleanly. Please make sure they do.
_C.mel_to_vid = _C.mel_samples // _C.video_samples
_C.mis_overlap = 0

# Including pairing/misalignment loss
_C.video_fps = 21.5 # Video fps. Default is 21.5, but for asmr videos, it actually is 30
_C.pairing_loss = True
_C.randomize_subset_samples = True # Toggle this to make random cuts for misalignment vs pre-defined cuts at 0, self.num_misaligned_frames and 2*num_misaligned_frames
_C.num_misalign_frames = 43
_C.reduced_video_samples = _C.video_samples // 10 * 3 # Make it a (roughly) 3s chunk.
_C.reduced_mel_samples = _C.reduced_video_samples * _C.mel_to_vid # Corresponding reduced time dimension of mel spectrogram
_C.temporal_alignment_lambda = 2.0 # Weight for temporal loss
_C.allow_temporal_misalign_overlap = True # Allow the misalignment in negative samples to overlap
_C.temporal_misalign_pos_samples = 3 # How many examples of 'aligned' snippets we want
_C.temporal_misalign_neg_samples_per_pos = 2 # How many negative examples to gen per positive example

# Use visual encoder output as input to the second network/discriminator
_C.visual_encoder_input = True

# Include extra upsampling (needed to match waveglow configs of 44100 audio sampling rate, 1720 mel samples)
_C.extra_upsampling = True

# Include landmark featuers
_C.include_landmarks = False

# Logger parameters
_C.exclude_D_r_f = True
_C.exclude_gan_loss = True

# Encoder parameters
_C.random_z_dim = 512
_C.encoder_n_lstm = 2
_C.encoder_embedding_dim = 2048
_C.encoder_kernel_size = 5
_C.encoder_n_convolutions = 3

# Modal impulse prediction parameters
_C.n_modal_frequencies = 256

# Auxiliary parameters
_C.auxiliary_type = "lstm_last"
_C.auxiliary_dim = 256
_C.auxiliary_sample_rate = 32
_C.mode_input = ""
_C.aux_zero = False

# Decoder parameters
_C.decoder_conv_dim = 1024

# Mel-post processing network parameters
_C.postnet_embedding_dim = 512
_C.postnet_kernel_size = 5
_C.postnet_n_convolutions = 5

_C.loss_type = "MSE" # Reconstruction loss type
_C.weight_decay = 1e-6
_C.grad_clip_thresh = 1.0
_C.batch_size = 32
_C.lr = 0.0002
_C.beta1 = 0.5
_C.continue_train = False
_C.lambda_Oriloss = 10000.0 # Original weight: 10000.0
_C.lambda_Silenceloss = 0
_C.niter = 100
_C.D_interval = 1
_C.wo_G_GAN = True # Flag on whether to use GAN loss(or GAN setup in our case) or not
_C.wavenet_path = ""
# waveglow checkpoint path. Used only during test.
_C.waveglow_path = "/juno/group/SoundProject/WaveGlowWeights/TrainAll/checkpoints/waveglow_152500"
