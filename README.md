# Generating Visually Aligned Sound from Videos

This is the official pytorch implementation of the TIP paper "[Generating Visually Aligned Sound from Videos][REGNET]" and the corresponding Visually Aligned Sound (VAS) dataset. 

Demo videos containing sound generation results can be found [here][demo].

![](https://github.com/PeihaoChen/regnet/blob/master/overview.png)

## Updates

- We release the pre-computed features for the testset of Dog category, together with the pre-trained RegNet. You can use them for generating dog sounds by yourself. (23/11/2020)

# Contents
----

* [Usage Guide](#usage-guide)
   * [Getting Started](#getting-started)
      * [Installation](#installation)
      * [Download Datasets](#download-datasets)
      * [Data Preprocessing](#data-preprocessing)
   * [Training RegNet](#training-regnet)
   * [Generating Sound](#generating-sound)
   * [Pre-trained RegNet](#pre-trained-regnet)
* [Other Info](#other-info)
   * [Citation](#citation)
   * [Contact](#contact)


----
# Usage Guide

## Getting Started
[[back to top](#Generating-Visually-Aligned-Sound-from-Videos)]

### Installation

Clone this repository into a directory. We refer to that directory as *`REGNET_ROOT`*.

NOTE: Since this repo has submodules, need to clone recursively to get them:

```bash
git clone --recursive git@github.com:jhyau/regnet.git
cd regnet
```

If git clone was already done without recursive, run this to get the submodules:
```bash
git submodule update --init --recursive
```

```bash
git clone https://github.com/PeihaoChen/regnet
cd regnet
```
Create a new Conda environment.
```bash
conda create -n regnet python=3.7.1
conda activate regnet
```
Install [PyTorch][pytorch] and other dependencies.
```bash
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0
conda install ffmpeg -n regnet -c conda-forge
pip install -r requirements.txt
```

Note: Can use newer torch version and updated CUDA tool kit, so suggest installing:
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```
instead of pytorch==1.2.0

Note on DDSP: Instead of installing with the usual 
```
pip install ddsp
```

Navigate to the forked DDSP repo and run
```
pip install -e .
```

Another note: After installing ddsp, if there is an error for needing a newer version of tensorflow (e.g. >=2.5), upgrade the tensorflow installation with 
```
pip install tensorflow --upgrade
```

### Download Datasets

In our paper, we collect 8 sound types (Dog, Fireworks, Drum, Baby form [VEGAS][vegas] and Gun, Sneeze, Cough, Hammer from [AudioSet][audioset]) to build our [Visually Aligned Sound (VAS)][VAS] dataset.
Please first download VAS dataset and unzip the data to *`$REGNET_ROOT/data/`*  folder.

For each sound type in AudioSet, we download all videos from Youtube and clean data on Amazon Mechanical Turk (AMT) using the same way as [VEGAS][visual_to_sound].


```bash
unzip ./data/VAS.zip -d ./data
```



### Data Preprocessing

Run `data_preprocess.sh` to preprocess data and extract RGB and optical flow features. 

Notice: The script we provided to calculate optical flow is easy to run but is resource-consuming and will take a long time. We strongly recommend you to refer to [TSN repository][TSN] and their built [docker image][TSN_docker] (our paper also uses this solution)  to speed up optical flow extraction and to restrictly reproduce the results.
```bash
source data_preprocess.sh
```


## Training RegNet

Training the RegNet from scratch. The results will be saved to `ckpt/dog`.

```bash
CUDA_VISIBLE_DEVICES=7 python train.py \
save_dir ckpt/dog \
auxiliary_dim 32 \ 
rgb_feature_dir data/features/dog/feature_rgb_bninception_dim1024_21.5fps \
flow_feature_dir data/features/dog/feature_flow_bninception_dim1024_21.5fps \
mel_dir data/features/dog/melspec_10s_22050hz \
checkpoint_path ''
```

In case that the program stops unexpectedly, you can continue training.
```bash
CUDA_VISIBLE_DEVICES=7 python train.py \
-c ckpt/dog/opts.yml \
checkpoint_path ckpt/dog/checkpoint_018081
```

Notes:
When starting own training, in addition to repalce rgb\_feature\_dir and flow\_feature\_dir adn mel\_dir, also replace training\_files, test\_files, and batch\_size.

## Generating Sound


During inference, our RegNet will generate visually aligned spectrogram, and then use [WaveNet][wavenet] as vocoder to generate waveform from spectrogram. You should first download our trained WaveNet model for different sound categories (
[Dog](https://github.com/PeihaoChen/regnet/releases/download/WaveNet_model/dog_checkpoint_step000200000_ema.pth),
[Fireworks](https://github.com/PeihaoChen/regnet/releases/download/WaveNet_model/fireworks_checkpoint_step000267000_ema.pth),
[Drum](https://github.com/PeihaoChen/regnet/releases/download/WaveNet_model/drum_checkpoint_step000160000_ema.pth),
[Baby](https://github.com/PeihaoChen/regnet/releases/download/WaveNet_model/baby_checkpoint_step000470000_ema.pth),
[Gun](https://github.com/PeihaoChen/regnet/releases/download/WaveNet_model/gun_checkpoint_step000152000_ema.pth),
[Sneeze](https://github.com/PeihaoChen/regnet/releases/download/WaveNet_model/sneeze_checkpoint_step000071000_ema.pth),
[Cough](https://github.com/PeihaoChen/regnet/releases/download/WaveNet_model/cough_checkpoint_step000079000_ema.pth),
[Hammer](https://github.com/PeihaoChen/regnet/releases/download/WaveNet_model/hammer_checkpoint_step000137000_ema.pth)
). 

The generated spectrogram and waveform will be saved at `ckpt/dog/inference_result`
```bash
CUDA_VISIBLE_DEVICES=7 python test.py \
-c ckpt/dog/opts.yml \ 
aux_zero True \ 
checkpoint_path ckpt/dog/checkpoint_041000 \ 
save_dir ckpt/dog/inference_result \
wavenet_path /path/to/wavenet_dog.pth
```

If you want to train your own WaveNet model, you can use [WaveNet repository][wavenet_repository].
```bash
git clone https://github.com/r9y9/wavenet_vocoder && cd wavenet_vocoder
git checkout 2092a64
```

## Pre-trained RegNet

You can also use our pre-trained RegNet and pre-computed features for generating visually aligned sounds.

First, download and unzip the pre-computed features ([Dog](https://github.com/PeihaoChen/regnet/releases/download/pretrained_RegNet/features_dog_testset.tar)) to `./data/features/dog` folder.
```bash
cd ./data/features/dog
tar -xvf features_dog_testset.tar # unzip
```

Second, download and unzip our pre-trained RegNet ([Dog](https://github.com/PeihaoChen/regnet/releases/download/pretrained_RegNet/RegNet_dog_checkpoint_041000.tar)) to `./ckpt/dog` folder.
```bash
cd ./ckpt/dog
tar -xvf ./ckpt/dog/RegNet_dog_checkpoint_041000.tar # unzip
```


Third, run the inference code.
```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
-c config/dog_opts.yml \ 
aux_zero True \ 
checkpoint_path ckpt/dog/checkpoint_041000 \ 
save_dir ckpt/dog/inference_result \
wavenet_path /path/to/wavenet_dog.pth
```


Running inference with waveglow audio generation
```bash
python test.py --vocoder waveglow --waveglow_path /juno/group/SoundProject/WaveGlowWeights/TrainAll/checkpoints/waveglow_152500 --sampling_rate 44100 --is_fp16 --num_plots 3 --gt -c ckpt/asmr_full_no_GAN_loss/opts.yml aux_zero True checkpoint_path ckpt/asmr_full_no_GAN_loss/checkpoint_004900 save_dir ckpt/asmr_full_no_GAN_loss/inf_val_best_waveglow_trainall
```



Replacing the audio of a given video with another audio
```bash
python replace_audio_of_video.py single data/features/ASMR/asmr_both_vids/videos_10s_21.5fps/ASMR_Addictive_Tapping_1_Hr_No_Talking-58-of-365.mp4 ckpt/asmr_full_no_GAN_loss/inf_val_best_waveglow_trainall/ASMR_Addictive_Tapping_1_Hr_No_Talking-58-of-365_synthesis.wav ckpt/asmr_full_no_GAN_loss/inf_val_best_waveglow_trainall/ 
```

Enjoy your experiments!


# Other Info
[[back to top](#Generating-Visually-Aligned-Sound-from-Videos)]

## Citation


Please cite the following paper if you feel RegNet useful to your research
```
@Article{chen2020regnet,
  author  = {Peihao Chen, Yang Zhang, Mingkui Tan, Hongdong Xiao, Deng Huang and Chuang Gan},
  title   = {Generating Visually Aligned Sound from Videos},
  journal = {TIP},
  year    = {2020},
}
```

## Contact
For any question, please file an issue or contact
```
Peihao Chen: phchencs@gmail.com
Hongdong Xiao: xiaohongdonghd@gmail.com
```

[REGNET]:https://arxiv.org/abs/2008.00820
[audioset]:https://research.google.com/audioset/index.html
[VEGAS_link]:http://bvision11.cs.unc.edu/bigpen/yipin/visual2sound_webpage/VEGAS.zip
[pytorch]:https://github.com/pytorch/pytorch
[wavenet]:https://arxiv.org/abs/1609.03499
[wavenet_repository]:https://github.com/r9y9/wavenet_vocoder
[opencv]:https://github.com/opencv/opencv
[dense_flow]:https://github.com/yjxiong/dense_flow
[VEGAS]: http://bvision11.cs.unc.edu/bigpen/yipin/visual2sound_webpage/visual2sound.html
[visual_to_sound]: https://arxiv.org/abs/1712.01393
[TSN]: https://github.com/yjxiong/temporal-segment-networks
[VAS]: https://drive.google.com/file/d/14birixmH7vwIWKxCHI0MIWCcZyohF59g/view?usp=sharing
[TSN_docker]: https://hub.docker.com/r/bitxiong/tsn/tags
[demo]: https://youtu.be/fI_h5mZG7bg
