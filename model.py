import glob
import math
import os
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.optim import lr_scheduler
import torchvision.models as models
from tsn.models import TSN
from criterion import RegnetLoss
from config import _C as config

torch.cuda.empty_cache()

def to_gpu(x, device):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(device, non_blocking=True)
    return torch.autograd.Variable(x)

class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal
        
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(config.n_mel_channels, config.postnet_embedding_dim,
                         kernel_size=config.postnet_kernel_size, stride=1,
                         padding=int((config.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(config.postnet_embedding_dim))
        )

        for i in range(1, config.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(config.postnet_embedding_dim,
                             config.postnet_embedding_dim,
                             kernel_size=config.postnet_kernel_size, stride=1,
                             padding=int((config.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(config.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(config.postnet_embedding_dim, config.n_mel_channels,
                         kernel_size=config.postnet_kernel_size, stride=1,
                         padding=int((config.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(config.n_mel_channels))
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))
        x = self.convolutions[-1](x)

        return x

class MaterialClassificationNet(nn.Module):
    """Network to learn task of classifying material given video"""
    def __init__(self, num_classes):
        super(MaterialClassificationNet, self).__init__()
        # Number of classes for classification task (returned from dataset)
        self.num_classes = num_classes
        self.config = config
        self.model_names = ["material_classification_model"]
        self.device = torch.device('cuda:0')
        self.classifier_model = init_net(EncoderClassifier(num_classes), self.device)

        # Classification loss: CrossEntropy
        if config.loss_type == 'cross_entropy':
            loss_fn = nn.CrossEntropyLoss()
        elif config.loss_type == 'nll':
            loss_fn = nn.NLLLoss()
        else:
            raise("Error, unknown loss!")
        self.criterionL1 = loss_fn.to(self.device)

        # Also include contrastive loss

        self.optimizer = torch.optim.Adam(self.classifier_model.parameters(),
                                            lr=config.lr, betas=(config.beta1, 0.999))
        self.n_iter = -1

    def parse_batch(self, batch):
        inputs, labels, video_name, frame_index = batch
        #self.inputs = (raw_rgb.to(self.device).float(), raw_flow.to(self.device).float())
        self.inputs = inputs.to(self.device).float()
        self.labels = labels.to(self.device)
        self.video_name = video_name
        self.frame_index = frame_index

    def forward(self):
        # Pass the input through the visual encoder and classifier model
        self.output = self.classifier_model(self.inputs)
        return self.output

    def backward(self):
        # Classification loss for predicted material class/type
        targets = self.labels
        targets.requires_grad = False
        self.loss_L1 = self.criterionL1(self.output, targets)

        # TODO: Include contrastive loss (Normalized Temperature-Scaled Cross-Entropy Loss from SimCLR paper)

        self.loss_L1.backward()


    def get_scheduler(self, optimizer, config):
        def lambda_rule(epoch):
            # lr_l = 1.0 - max(0, epoch + 2 - config.niter) / float(config.epochs - config.niter + 1)
            lr_l = 1.0 - max(0, epoch + 2 + config.epoch_count - config.niter) / float(config.epochs - config.niter + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        return scheduler

    def setup(self):
        self.scheduler = self.get_scheduler(self.optimizer, config)  #[self.get_scheduler(optimizer, config) for optimizer in self.optimizers]

    def load_checkpoint(self, checkpoint_path):
        for name in self.model_names:
            filepath = "{}_net{}".format(checkpoint_path, name)
            print("Loading net{} from checkpoint '{}'".format(name, filepath))
            state_dict = torch.load(filepath, map_location='cpu')
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            #net = getattr(self, 'net' + name)
            net = self.classifier_model
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            checkpoint_state = state_dict["optimizer_net{}".format(name)]
            net.load_state_dict(checkpoint_state)
            self.iteration = state_dict["iteration"]

            learning_rate = state_dict["learning_rate"]
        #for index in range(len(self.optimizers)):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

    def save_checkpoint(self, save_directory, iteration, do_not_delete=[], save_current=False):
        # do_not_delete list of model_path checkpoints that shouldn't be deleted
        lr = self.optimizer.param_groups[0]['lr']
        for name in self.model_names:
            filepath = os.path.join(save_directory, "checkpoint_{:0>6d}_net{}".format(iteration, name))
            print("Saving net{} and optimizer state at iteration {} to {}".format(
                name, iteration, filepath))
            #net = getattr(self, 'net' + name)
            net = self.classifier_model
            if torch.cuda.is_available():
                torch.save({"iteration": iteration,
                            "learning_rate": lr,
                            "optimizer_net{}".format(name): net.module.cpu().state_dict()}, filepath)
                net.to(self.device)
            else:
                torch.save({"iteration": iteration,
                            "learning_rate": lr,
                            "optimizer_net{}".format(name): net.cpu().state_dict()}, filepath)

            if save_current:
                do_not_delete.append(filepath)

            """delete old model"""
            model_list = glob.glob(os.path.join(save_directory, "checkpoint_*_*"))
            model_list.sort()
            for model_path in model_list[:-2]:
                if model_path in do_not_delete:
                    # Skip past the checkpoints that we want to keep
                    continue
                cmd = "rm {}".format(model_path)
                print(cmd)
                os.system(cmd)
        return model_list[-1][:-5]


    def update_learning_rate(self):
        #for scheduler in self.schedulers:
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def optimize_parameters(self):
        self.n_iter += 1

        # Determine which forward function to use
        self.forward()

        # update model
        #self.set_requires_grad(self.netD, False)
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()


class EncoderClassifier(nn.Module):
    # Classifier part should be conv + linear layer + relu + softmax
    # Visual encoder output (no LSTM): (batch size, video samples, encoder embedding dim)
    # After conv layer: (batch size, 1, encoder embedding dim)
    # Squeeze the result to get (batch size, encoder embedding dim)
    # Send through linear layer to get: (batch size, num classes)
    # Send this through softmax
    # But need to make sure input to linear layer is flattened
    def __init__(self, num_classes):
        super(EncoderClassifier, self).__init__()
        self.num_classes = num_classes

        if config.mode_input == "":
            self.mode_input = "vis_spec" if self.training else "vis"
        else:
            self.mode_input = config.mode_input

        # Initialize the visual encoder
        self.encoder = Encoder()

        # Classifier architecture
        conv_model = []
        # Per frame, then input is only 1 channel
        if config.per_frame:
            conv_model += [nn.Conv1d(in_channels=1, out_channels=1,
                kernel_size=5, stride=1, padding=2, dilation=1)] # third dim (L in pytorch docs) does not change
        else:
            conv_model += [nn.Conv1d(in_channels=config.video_samples, out_channels=1,
                kernel_size=5, stride=1, padding=2, dilation=1)] # third dim (L in pytorch docs) does not change
        conv_model += [nn.BatchNorm1d(1)]
        conv_model += [nn.ReLU(True)]
        self.conv_model = nn.Sequential(*conv_model)

        # Linear layer + softmax
        fc_model = []
        fc_model += [nn.Linear(config.encoder_embedding_dim, num_classes)]
        fc_model += [nn.ReLU(True)]

        if config.loss_type == 'nll':
            fc_model += [nn.LogSoftmax(dim=1)] # dim along which log softmax is computed so every slice along dim sums to 1
            # Softmax will be done automatically with nn.CrossEntropyLoss. with nn.NLLoss, need to include the log softmax layer
        self.fc_model = nn.Sequential(*fc_model)
        
    def forward(self, x):
        # Pass input through the visual encoder
        x = self.encoder(x)

        # Output from visual encoder (no LSTM for now) to feed into this network
        conv_output = self.conv_model(x)
        
        # Squeeze before sending through fully connected layer and softmax
        fc_input = torch.squeeze(conv_output)
        predictions = self.fc_model(fc_input)
        return predictions


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.random_z_dim = config.random_z_dim
        self.encoder_dim_with_z = config.visual_dim + self.random_z_dim

        convolutions = []
        for i in range(config.encoder_n_convolutions):
            conv_input_dim = self.encoder_dim_with_z if i==0 else config.encoder_embedding_dim
            conv_layer = nn.Sequential(
                ConvNorm(conv_input_dim,
                         config.encoder_embedding_dim,
                         kernel_size=config.encoder_kernel_size, stride=1,
                         padding=int((config.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(config.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.BiLSTM = nn.LSTM(config.encoder_embedding_dim,
                           int(config.encoder_embedding_dim / 4), config.encoder_n_lstm,
                           batch_first=True, bidirectional=True)
        self.BiLSTM_proj = nn.Linear(int(config.encoder_embedding_dim/2), int(config.encoder_embedding_dim/2))

    def forward(self, x):
        x = x.transpose(1, 2)
        z = torch.randn(x.shape[0], self.random_z_dim).to('cuda:0')
        z = z.view(z.size(0), z.size(1), 1).expand(z.size(0), z.size(1), x.size(2))
        x = torch.cat([x, z], 1)
        #print("encoder input earlier size: ", x.size())
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = x.transpose(1, 2)
        #print("before bilstm: ", x.size())

        # For classification, don't need to go through BiLISTM
        if config.classification or not config.use_lstm:
            return x

        x, _ = self.BiLSTM(x)
        #print("after bilstm: ", x.size())
        x = self.BiLSTM_proj(x)
        #print("after linear layer: ", x.size())
        return x


class Auxiliary_lstm_last(nn.Module):

    def __init__(self):
        super(Auxiliary_lstm_last, self).__init__()
        self.BiLSTM = nn.LSTM(config.n_mel_channels, int(config.auxiliary_dim), 2,
                           batch_first=True, bidirectional=True)
        self.BiLSTM_proj = nn.Linear(config.auxiliary_dim, config.auxiliary_dim)
        # Check if using pairing loss (would use reduced time dim)
        if config.pairing_loss:
            self.video_samples = config.reduced_video_samples
        else:
            self.video_samples = config.video_samples 

    def forward(self, x):
        x = x.transpose(1, 2)
        x, (h, c) = self.BiLSTM(x)
        x = self.BiLSTM_proj(h[-1])
        bs, c = x.shape
        #x = x.unsqueeze(1).expand(bs, 215, c)
        x = x.unsqueeze(1).expand(bs, self.video_samples, c)
        return x


class Auxiliary_lstm_sample(nn.Module):

    def __init__(self):
        super(Auxiliary_lstm_sample, self).__init__()
        self.BiLSTM = nn.LSTM(config.n_mel_channels, int(config.auxiliary_dim/2), 2,
                           batch_first=True, bidirectional=True)
        self.auxiliary_sample_rate = config.auxiliary_sample_rate

        if config.pairing_loss:
            self.mel_samples = config.reduced_mel_samples
            self.video_samples = config.reduced_video_samples
        else:
            self.video_samples = config.video_samples
            self.mel_samples = config.mel_samples

    def forward(self, x):
        x = x.transpose(1, 2)
        x, (h, c) = self.BiLSTM(x)
        bs, T, C = x.shape
        forword = x[:, :, :int(C/2)]
        backword = x[:, :, int(C/2):]

        forword_sampled = forword[:, torch.arange(0, T, self.auxiliary_sample_rate).long(), :]
        backword_sampled = backword[:, torch.arange(0, T, self.auxiliary_sample_rate).long()+1, :]
        sampled = torch.cat([forword_sampled, backword_sampled], dim=-1)
        sampled_repeat = sampled.unsqueeze(1).repeat(1, int(self.auxiliary_sample_rate/4), 1, 1).view(bs, -1, C)
        #assert sampled_repeat.shape[1] == math.ceil(860/self.auxiliary_sample_rate) * int(self.auxiliary_sample_rate/4)
        assert sampled_repeat.shape[1] == math.ceil(self.mel_samples/self.auxiliary_sample_rate) * int(self.auxiliary_sample_rate/4)
        #sampled_repeat = sampled_repeat[:, :215, :]
        sampled_repeat = sampled_repeat[:, :self.video_samples, :]
        return sampled_repeat


class Auxiliary_conv(nn.Module):

    def __init__(self):
        super(Auxiliary_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(config.n_mel_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.Conv1d(32, config.auxiliary_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(config.auxiliary_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class Modal_Impulse_Decoder(nn.Module):
    """
    Takes the visual encoder output and predicts the vectors needed to generate modal impulse:
    - gains
    - frequencies
    - dampings
    """
    def __init__(self):
        super(Modal_Impulse_Decoder, self).__init__()
        # Set up the decoder's convolution layers
        # Raw frequencies (1, 1, 256), raw gains: (1, 1, 256), raw dampings: (1, 256)
        # Visual encoder output has shape (batch size, video_samples, encoder_embedding_dim [2048])
        # If passes LSTM, then shape is (batch size, video_samples, config.encoder_embedding_dim/2 [1024])
        if config.use_lstm:
            in_chan = config.video_samples
        else:
            # Assuming per frame inputs
            assert(config.per_frame == True)
            in_chan = 1

        model = []
        model += [nn.Conv1d(in_channels=in_chan, out_channels=in_chan,
            kernel_size=5, stride=2, padding=2, dilation=1)]
        model += [nn.BatchNorm1d(in_chan)]
        model += [nn.ReLU(True)]

        if not config.use_lstm:
            # Need an extra layer of convolutions if output is directly from visual encoder and doesn't go through LSTM
            model += [nn.Conv1d(in_channels=in_chan, out_channels=in_chan,
                kernel_size=5, stride=2, padding=2, dilation=1)]
            model += [nn.BatchNorm1d(in_chan)]
            model += [nn.ReLU(True)]

        model += [nn.Conv1d(in_channels=in_chan, out_channels=1,
            kernel_size=5, stride=2, padding=2, dilation=1)]
        model += [nn.BatchNorm1d(1)]
        model += [nn.ReLU(True)]
        self.model = nn.Sequential(*model)

        #self.fc = nn.Linear(input_dim, output_dim)
        #self.relu = torch.nn.ReLU() # instead of Heaviside step fn

    def forward(self, x):
        output = self.model(x)
        #fc_output = self.fc(x)
        #output = self.relu(fc_output) # instead of Heaviside step fn
        return output


class Decoder(nn.Module):
    def __init__(self, extra_upsampling):
        super(Decoder, self).__init__()
        self.n_mel_channels = config.n_mel_channels
        # Added extra_upsampling parameter
        self.extra_upsampling = extra_upsampling
        model = []
        model += [nn.ConvTranspose1d(in_channels=config.decoder_conv_dim + config.auxiliary_dim, out_channels=int(config.decoder_conv_dim / 2),
                               kernel_size=4, stride=2, padding=1)]
        model += [nn.BatchNorm1d(int(config.decoder_conv_dim / 2))]
        model += [nn.ReLU(True)]

        model += [nn.Conv1d(in_channels=int(config.decoder_conv_dim / 2), out_channels=int(config.decoder_conv_dim / 2),
                               kernel_size=5, stride=1, padding=2)]
        model += [nn.BatchNorm1d(int(config.decoder_conv_dim / 2))]
        model += [nn.ReLU(True)]

        if extra_upsampling:
            # Adding additional transpose convolution layers to upsample more
            model += [nn.ConvTranspose1d(in_channels=int(config.decoder_conv_dim / 2), out_channels=int(config.decoder_conv_dim / 2),
                                   kernel_size=4, stride=2, padding=1)]
            model += [nn.BatchNorm1d(int(config.decoder_conv_dim / 2))]
            model += [nn.ReLU(True)]

            # Adding 1D convolution layer after transpose convolution layer
            model += [nn.Conv1d(in_channels=int(config.decoder_conv_dim / 2), out_channels=int(config.decoder_conv_dim / 2),
                                   kernel_size=5, stride=1, padding=2)]
            model += [nn.BatchNorm1d(int(config.decoder_conv_dim / 2))]
            model += [nn.ReLU(True)]
            # Done adding one additional transpose convolution layer. Adding two leads to 16*input size

        model += [nn.ConvTranspose1d(in_channels=int(config.decoder_conv_dim / 2), out_channels=self.n_mel_channels,
                               kernel_size=4, stride=2, padding=1)]
        model += [nn.BatchNorm1d(self.n_mel_channels)]
        model += [nn.ReLU(True)]

        model += [nn.Conv1d(in_channels=int(self.n_mel_channels), out_channels=self.n_mel_channels,
                               kernel_size=5, stride=1, padding=2)]

        self.model = nn.Sequential(*model)

    def forward(self, decoder_inputs):
        x = decoder_inputs.transpose(1, 2)
        #print("decode input size after transpose: ", x.size())

        x = self.model(x)
        #print("decoder output size: ", x.size())
        return x


class VisualFeatureExtractorCNN(nn.Module):
    """
    CNN Encoder to extract features from the images
    In finetuning: initialized pretrained model adn update all of the model's parameters (retraining the whole model)
    In feature extraction: Initialize pretrained model and only update the final layer weights
    It is called feature extraction because we use the pretrained CNN as a fixed feature-extractor,
    and only change the output layer.
    """
    def __init__(self, embed_dim=int(config.encoder_embedding_dim / 2), normalization='batch'):
        super(VisualFeatureExtractorCNN, self).__init__()
        model = models.resnet50(pretrained=True)
        # Pass in the pretrained resnet model (18, 34, 50, 101, 152, etc.) parameter later
        # Use pretrained ResNet18 as CNN encoder
        # self.encoder = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        resnet = model
        self.embed_dim = embed_dim
        self.device = torch.device('cuda:0')
        self.batch_size = config.batch_size
        # print("ResNet50 layers: ", list(resnet.children()))
        # print("model parameters: ", resnet.parameters())
        # Check if the requires_grad is set to true
        # for param in resnet.parameters():
        #     print(f'parmeter: {param}, and requires grad? {param.requires_grad}')

        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_dim)

        # Another way to do finetuning
        #resnet.fc = nn.Linear(resnet.fc.in_features, embed_dim)

        # Use instance norm if batch size is 1
        if normalization == 'batch':
            #self.bn = nn.BatchNorm1d(embed_dim, momentum=0.01)
            self.bn = nn.BatchNorm1d(config.video_samples, momentum=0.01)
        else:
            # InstanceNorm1d also doesn't work with 1 sample :(
            self.bn = nn.InstanceNorm1d(config.video_samples, momentum=0.01)

        # Finetune the last residual block, not just the top fc layer
 
    def forward(self, images, length):
        """Extract feature vectors from input images, but with finetuning"""
        #with torch.no_grad():
            # Pass images through resnet, use no grad if don't want to train parts of the model
            # Use no grad if only for feature extraction and don't want to finetune
        # Retrain/finetune the entire pretrained resnet50 model
        # Expect input to be similar to (64, 3, 7, 7), so expects 3 channels (1 image)

        # Shape: (batch size, 216, 3, 224, 224)
        input_var = torch.autograd.Variable(images.view(images.size(0), -1, length, images.size(2), images.size(3)))

        # Current input: (batch size, 216 * 3, 224, 224) or (batch size, 216 * 2, 224, 224)
        #num_images = int(images.shape[1] / 3)
        num_images = input_var.size(0)
        print(f"reshaped input size: {input_var.size()}")
        all_img_features = []
        for idx in range(num_images): # Iterating through batch size now, so actually num videos
            #print(f"img idx: {idx} out of {num_images}")
            #features = self.resnet(images[:, int(idx*3):int(idx*3)+3 , :, :])
            features = self.resnet(input_var[idx,:,:,:,:])
            features = features.reshape(features.size(0), -1)
            features = self.linear(features)
            all_img_features.append(features)
            # Pass the output through the new last fc layer and batchnorm
            #if config.batch_size > 1 and images.shape[0] > 1:
            #    features = self.bn(self.linear(features))
            #else:
            #    features = self.linear(features)
        final_feature = torch.stack(all_img_features, dim=0) # stack along batch size dimension
        print(f"final feature size: {final_feature.shape}")
        output = self.bn(final_feature)
        # Needs to output (batch, seq (time frames), feature)
        return output


class Frequency_Net(nn.Module):
    def __init__(self):
        super(Frequency_Net, self).__init__()
        if config.train_visual_feature_extractor:
            if config.visual_feature_extractor == "bn-inception":
                self.rgb_visual_feat_extractor = TSN("RGB", consensus_type=config.consensus_type, dropout=config.dropout, model_path=config.bn_inception_file)
                self.flow_visual_feat_extractor = TSN("Flow", consensus_type=config.consensus_type, dropout=config.dropout, model_path=config.bn_inception_file)
            elif config.visual_feature_extractor == 'resnet50':
                self.visual_feat_extractor = VisualFeatureExtractorCNN()
            else:
                raise Exception("Unknown visual feature extractor")
        self.encoder = Encoder()
        self.decoder = Modal_Impulse_Decoder()
        if config.mode_input == "":
            self.mode_input = "vis_spec" if self.training else "vis"
        else:
            self.mode_input = config.mode_input

    def forward(self, inputs):
        if self.mode_input == "vis_spec":
            vis_thr, spec_thr = 1, 1
        elif self.mode_input == "vis":
            vis_thr, spec_thr = 1, 0
        elif self.mode_input == "spec":
            vis_thr, spec_thr = 0, 1
        else:
            print(self.mode_input)
        
        # If want to train visual feature extractor, pass raw images through extractor first
        if config.train_visual_feature_extractor:
            # Input needs to be rgb and optical flow stacked images
            rgb, flow = inputs

            if config.visual_feature_extractor == 'bn-inception':
                rgb_feat_list = []
                # Iterate through 1 at a time
                for i in range(rgb.shape[0]):
                    rgb_input_var = torch.autograd.Variable(rgb[i,:,:,:].view(-1, 3, rgb.size(2), rgb.size(3)))
                    rgb_feat = torch.squeeze(self.rgb_visual_feat_extractor(rgb_input_var))
                    rgb_feat_list.append(rgb_feat)
                    #rgb_input_var.detach() # Release memory from gpu
                # Stack them up
                rgb_final_feat = torch.stack(rgb_feat_list, dim=0) # to create a batch again
                print(f"RGB feature shape: {rgb_final_feat.shape}")

                # Extract flow features
                flow_feat_list = []
                for i in range(flow.shape[0]):
                    flow_input_var = torch.autograd.Variable(flow[i,:,:,:].view(-1, 2, flow.size(2), flow.size(3)))
                    flow_feat = torch.squeeze(self.flow_visual_feat_extractor(flow_input_var))
                    flow_feat_list.append(flow_feat)
                    #flow_input_var.detach()
                flow_final_feat = torch.stack(flow_feat_list, dim=0)
                print(f"flow feature shape: {flow_final_feat.shape}")

                # Concatenate
                #feature = np.concatenate((rgb_final_feat, flow_final_feat), 1) # Visual dim=2048
                feature = torch.cat((rgb_final_feat, flow_final_feat), -1) # Concatenate along the last dimension
                inputs = torch.FloatTensor(feature.astype(np.float32))
                #inputs = self.real_A
                print(f"inputs shape before passing to encoder: {inputs.shape}")
            else:
                # If using resnet50
                rgb_feat = self.visual_feat_extractor(rgb, 3)
                inputs = rgb_feat

        # For pre-extracted features, can directly pass through the encoder
        # Pass the input through the visual encoder
        print(f"input to encoder: {inputs.shape}")
        encoder_output = self.encoder(inputs * vis_thr)
        print(f"encoder output: {encoder_output.shape}")
        decoder_output = self.decoder(encoder_output)
        print(f"decoder output shape: {decoder_output.shape}")
        assert(decoder_output.shape[-1] == config.n_modal_frequencies) # Needs to match frequency shape
        return decoder_output


class Modal_Response_Net(nn.Module):
    def __init__(self):
        super(Modal_Response_Net, self).__init__()
        self.config = config
        self.model_names = ["visual_encoder_frequency_decoder"]
        self.device = torch.device('cuda:0')
        self.freq_model = init_net(Frequency_Net(), self.device)

        # Reconstruction loss
        if config.loss_type == "MSE":
            loss_fn = nn.MSELoss()
        elif config.loss_type == "L1Loss":
            loss_fn = nn.L1Loss()
        else:
            print("ERROR LOSS TYPE!")
        self.criterionL1 = loss_fn.to(self.device)
        self.optimizer = torch.optim.Adam(self.freq_model.parameters(),
                                            lr=config.lr, betas=(config.beta1, 0.999))
        self.n_iter = -1

    def parse_batch(self, batch):
        if config.train_visual_feature_extractor:
            raw_rgb, raw_flow, raw_freqs, video_name = batch
            #self.inputs = (raw_rgb.to(self.device).float(), raw_flow.to(self.device).float())
            self.inputs = (raw_rgb.to(self.device).float(), raw_flow.float())
        else:
            input, raw_freqs, video_name, frame_index = batch
            self.inputs = input.to(self.device).float()
            self.frame_index = frame_index
        self.gt_raw_freqs = raw_freqs.to(self.device).float()
        self.video_name = video_name

    def forward(self):
        # Pass the input through the visual encoder and frequency decoder
        self.decoder_output = self.freq_model(self.inputs)
        return self.decoder_output

    def backward(self):
        # Reconstruction loss for the predicted frequency
        targets = self.gt_raw_freqs
        targets.requires_grad = False
        self.loss_L1 = self.criterionL1(self.decoder_output, targets)

        # loss_G_GAN is adversarial loss, the other two term (loss_G_L1 and loss_G_silence) are reconstruction loss
        #self.loss_G = self.loss_G_GAN + self.loss_G_L1 * self.config.lambda_Oriloss + self.loss_G_silence * self.config.lambda_Silenceloss

        self.loss_L1.backward()


    def get_scheduler(self, optimizer, config):
        def lambda_rule(epoch):
            # lr_l = 1.0 - max(0, epoch + 2 - config.niter) / float(config.epochs - config.niter + 1)
            lr_l = 1.0 - max(0, epoch + 2 + config.epoch_count - config.niter) / float(config.epochs - config.niter + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        return scheduler

    def setup(self):
        self.scheduler = self.get_scheduler(self.optimizer, config)  #[self.get_scheduler(optimizer, config) for optimizer in self.optimizers]

    def load_checkpoint(self, checkpoint_path):
        for name in self.model_names:
            filepath = "{}_net{}".format(checkpoint_path, name)
            print("Loading net{} from checkpoint '{}'".format(name, filepath))
            state_dict = torch.load(filepath, map_location='cpu')
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            #net = getattr(self, 'net' + name)
            net = self.freq_model
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            checkpoint_state = state_dict["optimizer_net{}".format(name)]
            net.load_state_dict(checkpoint_state)
            self.iteration = state_dict["iteration"]

            learning_rate = state_dict["learning_rate"]
        #for index in range(len(self.optimizers)):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

    def save_checkpoint(self, save_directory, iteration, do_not_delete=[], save_current=False):
        # do_not_delete list of model_path checkpoints that shouldn't be deleted
        lr = self.optimizer.param_groups[0]['lr']
        for name in self.model_names:
            filepath = os.path.join(save_directory, "checkpoint_{:0>6d}_net{}".format(iteration, name))
            print("Saving net{} and optimizer state at iteration {} to {}".format(
                name, iteration, filepath))
            #net = getattr(self, 'net' + name)
            net = self.freq_model
            if torch.cuda.is_available():
                torch.save({"iteration": iteration,
                            "learning_rate": lr,
                            "optimizer_net{}".format(name): net.module.cpu().state_dict()}, filepath)
                net.to(self.device)
            else:
                torch.save({"iteration": iteration,
                            "learning_rate": lr,
                            "optimizer_net{}".format(name): net.cpu().state_dict()}, filepath)

            if save_current:
                do_not_delete.append(filepath)

            """delete old model"""
            model_list = glob.glob(os.path.join(save_directory, "checkpoint_*_*"))
            model_list.sort()
            for model_path in model_list[:-2]:
                if model_path in do_not_delete:
                    # Skip past the checkpoints that we want to keep
                    continue
                cmd = "rm {}".format(model_path)
                print(cmd)
                os.system(cmd)
        return model_list[-1][:-5]


    def update_learning_rate(self):
        #for scheduler in self.schedulers:
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def optimize_parameters(self):
        self.n_iter += 1

        # Determine which forward function to use
        if config.pairing_loss:
            self.forward_pairing_loss()
        else:
            self.forward()

        # update model
        #self.set_requires_grad(self.netD, False)
        self.optimizer.zero_grad()

        if config.pairing_loss:
            self.backward_G_pairing_loss()
        else:
            self.backward()
        self.optimizer.step()


class Regnet_G(nn.Module):
    def __init__(self, extra_upsampling):
        super(Regnet_G, self).__init__()
        auxiliary_class = None
        if config.auxiliary_type == "lstm_last":
            auxiliary_class = Auxiliary_lstm_last
        elif config.auxiliary_type == "lstm_sample":
            auxiliary_class = Auxiliary_lstm_sample
        elif config.auxiliary_type == "conv":
            auxiliary_class = Auxiliary_conv
        self.n_mel_channels = config.n_mel_channels
        # Added extra_upsampling boolean
        self.extra_upsampling = extra_upsampling
        self.encoder = Encoder()
        self.auxiliary = auxiliary_class()
        self.decoder = Decoder(extra_upsampling)
        self.postnet = Postnet()
        if config.mode_input == "":
            self.mode_input = "vis_spec" if self.training else "vis"
        else:
            self.mode_input = config.mode_input
        self.aux_zero = config.aux_zero

        # If wanting to train/finetune the feature extractor
        if config.train_visual_feature_extractor:
            self.rgb_visual_feat_extractor = TSN("RGB", consensus_type=config.consensus_type, dropout=config.dropout, model_path=config.bn_inception_file)
            self.flow_visual_feat_extractor = TSN("Flow", consensus_type=config.consensus_type, dropout=config.dropout, model_path=config.bn_inception_file)

    def forward(self, inputs, real_B):
        if self.mode_input == "vis_spec":
            vis_thr, spec_thr = 1, 1
        elif self.mode_input == "vis":
            vis_thr, spec_thr = 1, 0
        elif self.mode_input == "spec":
            vis_thr, spec_thr = 0, 1
        else:
            print(self.mode_input)
        
        if config.train_visual_feature_extractor:
            rgb, flow = inputs

            # Extract RGB feautures
            print(f"input rgb shape: {rgb.shape} and flow shape: {flow.shape}")
            rgb_feat_list = []
            # Iterate through 1 at a time
            for i in range(rgb.shape[0]):
                print("rgb feat index: ", i)
                rgb_input_var = torch.autograd.Variable(rgb[i,:,:,:].view(-1, 3, rgb.size(2), rgb.size(3)))
                rgb_feat = torch.squeeze(self.rgb_visual_feat_extractor(rgb_input_var))
                rgb_feat_list.append(rgb_feat)
                rgb_input_var.detach() # Release memory from gpu
            # Stack them up
            rgb_final_feat = torch.stack(rgb_feat_list, dim=0) # to create a batch again
            print(f"RGB feature shape: {rgb_final_feat.shape}")

            # Extract flow features
            flow_feat_list = []
            for i in range(flow.shape[0]):
                flow_input_var = torch.autograd.Variable(flow[i,:,:,:].view(-1, 2, flow.size(2), flow.size(3)))
                flow_feat = torch.squeeze(self.flow_visual_feat_extractor(flow_input_var))
                flow_feat_list.append(flow_feat)
                flow_input_var.detach()
            flow_final_feat = torch.stack(flow_feat_list, dim=0)
            print(f"flow feature shape: {flow_final_feat.shape}")

            # Concatenate
            #feature = np.concatenate((rgb_final_feat, flow_final_feat), 1) # Visual dim=2048
            feature = torch.cat((rgb_final_feat, flow_final_feat), -1) # Concatenate along the last dimension
            self.real_A = torch.FloatTensor(feature.astype(np.float32))
            inputs = self.real_A
            print(f"inputs shape before passing to encoder: {inputs.shape}")

        #print(f"Mode input for the generator: {self.mode_input}")
        encoder_output_init = self.encoder(inputs * vis_thr)
        gt_auxilitary = self.auxiliary(real_B * spec_thr)
        #print(f'encoder output size: {encoder_output.size()} and gt size: {gt_auxilitary.size()}')
        if self.aux_zero:
            gt_auxilitary = gt_auxilitary * 0
            #print(f"Ground truth spectrogram set to zero: {gt_auxilitary}")
        encoder_output = torch.cat([encoder_output_init, gt_auxilitary], dim=2)
        #print("after concatenation to feed to decoder: ", encoder_output.size())
        mel_output_decoder = self.decoder(encoder_output)
        mel_output_postnet = self.postnet(mel_output_decoder)
        #print(f'decoder mel output shape: {mel_output_decoder.size()} and postnet output shape: {mel_output_postnet.size()}')
        mel_output = mel_output_decoder + mel_output_postnet
        self.gt_auxilitary = gt_auxilitary
        #print(f'mel output size: {mel_output.shape}, gt_aux shape: {gt_auxilitary.shape}')
        return mel_output, mel_output_decoder, encoder_output_init


class Regnet_D(nn.Module):
    def __init__(self, extra_upsampling, visual_encoder_input):
        super(Regnet_D, self).__init__()

        if not visual_encoder_input:
            input_size = config.visual_dim
        else:
            input_size = int(config.encoder_embedding_dim / 2)

        if extra_upsampling:
            self.feature_conv = nn.Sequential(
                # To pass in the visual encoder output instead of frame features
                nn.ConvTranspose1d(input_size, config.decoder_conv_dim,
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(config.decoder_conv_dim),
                nn.LeakyReLU(0.2, True),
                # Add an additional upsampling layer to match 1720 output
                nn.ConvTranspose1d(config.decoder_conv_dim, config.decoder_conv_dim,
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(config.decoder_conv_dim),
                nn.LeakyReLU(0.2, True),
                # End of adding additional upsampling layer
                nn.ConvTranspose1d(config.decoder_conv_dim, 64,
                                   kernel_size=4, stride=2, padding=1),
            )
        else:
            self.feature_conv = nn.Sequential(
                nn.ConvTranspose1d(config.visual_dim, config.decoder_conv_dim,
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(config.decoder_conv_dim),
                nn.LeakyReLU(0.2, True),
                # Add an additional upsampling layer to match 1720 output
                #nn.ConvTranspose1d(config.decoder_conv_dim, config.decoder_conv_dim,
                #                   kernel_size=4, stride=2, padding=1),
                #nn.BatchNorm1d(config.decoder_conv_dim),
                #nn.LeakyReLU(0.2, True),
                # End of adding additional upsampling layer
                nn.ConvTranspose1d(config.decoder_conv_dim, 64,
                                   kernel_size=4, stride=2, padding=1),
            )

        self.mel_conv = nn.ConvTranspose1d(config.n_mel_channels, 64,
                               kernel_size=1, stride=1)

        sequence = [
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(512, 1024, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(1024, 1, kernel_size=4, stride=1, padding=1),
        ]
        self.down_sampling = nn.Sequential(*sequence)  # receptive field = 34

    def forward(self, *inputs):
        feature, mel = inputs
        #print(f"discriminator feature shape: {feature.size()}, mel shape: {mel.size()}")
        feature_conv = self.feature_conv(feature.transpose(1, 2))
        mel_conv = self.mel_conv(mel)
        #print(f"discriminator after transform feature shape: {feature_conv.size()}, mel shape: {mel_conv.size()}")
        input_cat = torch.cat((feature_conv, mel_conv), 1)
        #print("input concatenated feature+mel shape: ", input_cat.shape)
        out = self.down_sampling(input_cat)
        #print(f"discriminator output final shape: {out.shape}")
        
        if not config.pairing_loss:
            print("Using sigmoid for discriminator output...")
            out = nn.Sigmoid()(out) # Needed if using BCELoss, but sigmoid is already included in BCELossWithLogits
        return out


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=False, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class TemporalAlignmentLoss(nn.Module):
    """Discriminate betweeen aligned or misaligned input/target
    'real' for aligned and 'fake' for misaligned
    """
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(TemporalAlignmentLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.BCEWithLogitsLoss() #sigmoid + BCE loss

    def __call__(self, input, target):
        """
        Input should be discriminator output. Discriminate between aligned (real) or not aligned (fake)
        Instead of before, which was to discriminate between 'real' (ground truth mel) and 'fake' (generator mel)
        """
        # TODO: Remove later? To reduce to one score instead of expanding target to match...
        # Default target shape is (batch size, ), but discriminator output is (batch size, 1, sequence length<214>)
        #target = torch.unsqueeze(target, 1)
        #target = target.expand_as(input)

        # Take the average across the row of discriminator output to create one score per example
        input = torch.squeeze(torch.mean(input, -1))
        #print("misalignment loss input shape: ", input.shape)
        #print(f"averaged discriminator output values: {input}")
        return self.loss(input, target)
        


def init_net(net, device, init_type='normal', init_gain=0.02):
    assert (torch.cuda.is_available())
    net.to(device)
    net = torch.nn.DataParallel(net, range(torch.cuda.device_count()))
    init_weights(net, init_type, gain=init_gain)
    return net


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class Regnet(nn.Module):
    def __init__(self, extra_upsampling=False, adversarial_loss=True):
        super(Regnet, self).__init__()
        self.config = config
        self.n_mel_channels = config.n_mel_channels
        self.model_names = ['G', 'D']
        self.device = torch.device('cuda:0')
        self.netG = init_net(Regnet_G(extra_upsampling), self.device)
        self.netD = init_net(Regnet_D(extra_upsampling, visual_encoder_input=config.visual_encoder_input), self.device)

        # Set to pairing loss
        if config.pairing_loss:
            self.criterionGAN = TemporalAlignmentLoss().to(self.device)
        else:
            self.criterionGAN = GANLoss().to(self.device) # Adversarial loss
        self.criterionL1 = RegnetLoss(config.loss_type).to(self.device)

        self.optimizers = []
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=config.lr, betas=(config.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=config.lr, betas=(config.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        self.D_interval = config.D_interval
        self.n_iter = -1
        self.wo_G_GAN = config.wo_G_GAN # Set this parameter to True to exclude adversarial loss in the generator
        # add a parameter to indicate when to use extra upsampling layers in the docoder and discriminator (for 44100 audio sample rate)
        self.extra_upsampling = extra_upsampling
        # add parameter to indicate whether or not to use adversarial loss when training
        self.adversarial_loss = adversarial_loss
        if not self.adversarial_loss:
            if not self.wo_G_GAN:
                print("Adversarial loss should not be used, but is being used for generator. Setting to false")
                self.wo_G_GAN = True

    def parse_batch(self, batch):
        input, mel, video_name = batch
        self.real_A = input.to(self.device).float()
        self.real_B = mel.to(self.device).float()
        self.video_name = video_name
        #print(f'input size: {self.real_A.shape}, real mel-spec size: {self.real_B.shape}')

    def parse_batch_train_visual_feat_extractor(self, batch):
        raw_rgb, raw_flow, mel, video_name = batch
        self.inputs = (raw_rgb.to(self.device).float(), raw_flow.float())
        self.real_B = mel.to(self.device).float()
        self.video_name = video_name

    def parse_batch_pairing_loss(self, batch):
        """Use this function to parse in the batch of examples based on pairing loss"""
        # There are 9 different sets of examples for pairing loss: 6 misaligned, 3 aligned
        cen, cen_mis_back, cen_mis_for, back, back_mis_cen, back_mis_for, forward, forward_mis_cen, forward_mis_back = batch

        # Set all the input video features and audio for the model
        self.video_name = cen[2]
        assert(self.video_name == back[2])
        assert(self.video_name == forward_mis_cen[2])
        assert(self.video_name == back_mis_for[2])

        # A is video feature vector, B is mel spec audio
        # Align: video is center, audio is center
        self.real_A_cen = cen[0].to(self.device).float()
        self.real_B_cen = cen[1].to(self.device).float()
        self.cen_label = cen[-1].to(self.device).float()

        # Misalign: video is center, audio is shifted back
        self.real_A_cen_mis_back = cen_mis_back[0].to(self.device).float()
        self.real_B_cen_mis_back = cen_mis_back[1].to(self.device).float()
        self.cen_mis_back_label = cen_mis_back[-1].to(self.device).float()

        # Misalign: video is center, audio is shifted forward
        self.real_A_cen_mis_for = cen_mis_for[0].to(self.device).float()
        self.real_B_cen_mis_for = cen_mis_for[1].to(self.device).float()
        self.cen_mis_for_label = cen_mis_for[-1].to(self.device).float()

        # Align: video is back, audio is back
        self.real_A_back = back[0].to(self.device).float()
        self.real_B_back = back[1].to(self.device).float()
        self.back_label = back[-1].to(self.device).float()

        # Misalign: video is back, audio is center
        self.real_A_back_mis_cen = back_mis_cen[0].to(self.device).float()
        self.real_B_back_mis_cen = back_mis_cen[1].to(self.device).float()
        self.back_mis_cen_label = back_mis_cen[-1].to(self.device).float()

        # Misalign: video is back, audio is forward
        self.real_A_back_mis_for = back_mis_for[0].to(self.device).float()
        self.real_B_back_mis_for = back_mis_for[1].to(self.device).float()
        self.back_mis_for_label = back_mis_for[-1].to(self.device).float()

        # Align: video is forward, audio is forward
        self.real_A_for = forward[0].to(self.device).float()
        self.real_B_for = forward[1].to(self.device).float()
        self.for_label = forward[-1].to(self.device).float()

        # Misalign: video is forward, audio is center
        self.real_A_for_mis_cen = forward_mis_cen[0].to(self.device).float()
        self.real_B_for_mis_cen = forward_mis_cen[1].to(self.device).float()
        self.for_mis_cen_label = forward_mis_cen[-1].to(self.device).float()

        # Misalign: video is forward, audio is back
        self.real_A_for_mis_back = forward_mis_back[0].to(self.device).float()
        self.real_B_for_mis_back = forward_mis_back[1].to(self.device).float()
        self.for_mis_back_label = forward_mis_back[-1].to(self.device).float()


    def forward(self):
        if config.train_visual_feature_extractor:
            self.fake_B, self.fake_B_postnet, self.encoder_output = self.netG(self.inputs, self.real_B)
        else:
            self.fake_B, self.fake_B_postnet, self.encoder_output = self.netG(self.real_A, self.real_B)
        #print(f'audio prediction size: {self.fake_B.shape}, postnet output shape: {self.fake_B_postnet.shape}')

    def forward_pairing_loss(self):
        # Have the generator predict for the three aligned examples
        #print(f"input sample shape: {self.real_A_cen.shape} and {self.real_B_cen.shape}")
        self.fake_B_cen, self.fake_B_cen_postnet, self.fake_cen_encoder_output = self.netG(self.real_A_cen, self.real_B_cen)
        self.fake_B_back, self.fake_B_back_postnet, self.fake_back_encoder_output = self.netG(self.real_A_back, self.real_B_back)
        self.fake_B_for, self.fake_B_for_postnet, self.fake_for_encoder_output = self.netG(self.real_A_for, self.real_B_for)

        #print(f"Output generated shape: {self.fake_B_cen.shape} and {self.fake_B_cen_postnet.shape}") 

    def get_scheduler(self, optimizer, config):
        def lambda_rule(epoch):
            # lr_l = 1.0 - max(0, epoch + 2 - config.niter) / float(config.epochs - config.niter + 1)
            lr_l = 1.0 - max(0, epoch + 2 + config.epoch_count - config.niter) / float(config.epochs - config.niter + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        return scheduler

    def setup(self):
        self.schedulers = [self.get_scheduler(optimizer, config) for optimizer in self.optimizers]

    def load_checkpoint(self, checkpoint_path):
        for name in self.model_names:
            filepath = "{}_net{}".format(checkpoint_path, name)
            print("Loading net{} from checkpoint '{}'".format(name, filepath))
            state_dict = torch.load(filepath, map_location='cpu')
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            net = getattr(self, 'net' + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            checkpoint_state = state_dict["optimizer_net{}".format(name)]
            net.load_state_dict(checkpoint_state)
            self.iteration = state_dict["iteration"]

            learning_rate = state_dict["learning_rate"]
        for index in range(len(self.optimizers)):
            for param_group in self.optimizers[index].param_groups:
                param_group['lr'] = learning_rate

    def save_checkpoint(self, save_directory, iteration, do_not_delete=[], save_current=False):
        # do_not_delete list of model_path checkpoints that shouldn't be deleted
        lr = self.optimizers[0].param_groups[0]['lr']
        for name in self.model_names:
            filepath = os.path.join(save_directory, "checkpoint_{:0>6d}_net{}".format(iteration, name))
            print("Saving net{} and optimizer state at iteration {} to {}".format(
                name, iteration, filepath))
            net = getattr(self, 'net' + name)
            if torch.cuda.is_available():
                torch.save({"iteration": iteration,
                            "learning_rate": lr,
                            "optimizer_net{}".format(name): net.module.cpu().state_dict()}, filepath)
                net.to(self.device)
            else:
                torch.save({"iteration": iteration,
                            "learning_rate": lr,
                            "optimizer_net{}".format(name): net.cpu().state_dict()}, filepath)

            if save_current:
                do_not_delete.append(filepath)

            """delete old model"""
            model_list = glob.glob(os.path.join(save_directory, "checkpoint_*_*"))
            model_list.sort()
            for model_path in model_list[:-2]:
                if model_path in do_not_delete:
                    # Skip past the checkpoints that we want to keep
                    continue
                cmd = "rm {}".format(model_path)
                print(cmd)
                os.system(cmd)
        return model_list[-1][:-5]


    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(self.real_A.detach(), self.fake_B.detach())
        self.pred_fake = pred_fake.data.cpu()
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        pred_real = self.netD(self.real_A, self.real_B)
        self.pred_real = pred_real.data.cpu()
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        if not self.adversarial_loss:
            # If adversarial loss is set to False, set adversarial loss to 0
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0
        else:
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()
        # Discriminator uses purely adversarial loss

    def backward_D_pairing_loss(self):
        """Discriminator backprop using pairing loss"""
        # Using visual encoder output as input
        # Use all 9 examples, starting with the 6 misaligned "fake" examples
        # Need to detach to prevent backproping through generator with these fake examples
        pred_fake_center_mis_back = self.netD(self.fake_cen_encoder_output.detach(), self.real_B_cen_mis_back.detach())
        #self.pred_fake_center_mis_back = pred_fake_center_mis_back.data.cpu()
        #print("discriminator output shape: ", self.pred_fake_center_mis_back.shape)
        #print("Actual output: ", self.pred_fake_center_mis_back)
        self.loss_D_fake_center_mis_back = self.criterionGAN(pred_fake_center_mis_back, self.cen_mis_back_label)
        #print("pairing loss: ", self.loss_D_fake_center_mis_back)
       
        pred_fake_center_mis_for = self.netD(self.fake_cen_encoder_output.detach(), self.real_B_cen_mis_for.detach())
        #self.pred_fake_center_mis_for = pred_fake_center_mis_for.data.cpu()
        self.loss_D_fake_center_mis_for = self.criterionGAN(pred_fake_center_mis_for, self.cen_mis_for_label)

        pred_fake_back_mis_cen = self.netD(self.fake_back_encoder_output.detach(), self.real_B_back_mis_cen.detach())
        #self.pred_fake_back_mis_cen = pred_fake_back_mis_cen.data.cpu()
        self.loss_D_fake_back_mis_cen = self.criterionGAN(pred_fake_back_mis_cen, self.back_mis_cen_label)

        pred_fake_back_mis_for = self.netD(self.fake_back_encoder_output.detach(), self.real_B_back_mis_for.detach())
        #self.pred_fake_back_mis_for = pred_fake_back_mis_for.data.cpu()
        self.loss_D_fake_back_mis_for = self.criterionGAN(pred_fake_back_mis_for, self.back_mis_for_label)

        pred_fake_for_mis_cen = self.netD(self.fake_for_encoder_output.detach(), self.real_B_for_mis_cen.detach())
        #self.pred_fake_for_mis_cen = pred_fake_for_mis_cen.data.cpu()
        self.loss_D_fake_for_mis_cen = self.criterionGAN(pred_fake_for_mis_cen, self.for_mis_cen_label)

        pred_fake_for_mis_back = self.netD(self.fake_for_encoder_output.detach(), self.real_B_for_mis_back.detach())
        #self.pred_fake_for_mis_back = pred_fake_for_mis_back.data.cpu()
        self.loss_D_fake_for_mis_back = self.criterionGAN(pred_fake_for_mis_back, self.for_mis_back_label)

        self.loss_D_fake = (self.loss_D_fake_center_mis_back + self.loss_D_fake_center_mis_for + self.loss_D_fake_back_mis_cen + self.loss_D_fake_back_mis_for +
                self.loss_D_fake_for_mis_cen + self.loss_D_fake_for_mis_back) * (1.0 / 6)

        # Calculate the loss for real/aligned cases
        pred_real_cen = self.netD(self.fake_cen_encoder_output.detach(), self.real_B_cen)
        #self.pred_real_cen = pred_real_cen.data.cpu()
        self.loss_D_real_cen = self.criterionGAN(pred_real_cen, self.cen_label)

        pred_real_back = self.netD(self.fake_back_encoder_output.detach(), self.real_B_back)
        #self.pred_real_back = pred_real_back.data.cpu()
        self.loss_D_real_back = self.criterionGAN(pred_real_back, self.back_label)

        pred_real_for = self.netD(self.fake_for_encoder_output.detach(), self.real_B_for)
        #self.pred_real_for = pred_real_for.data.cpu()
        self.loss_D_real_for = self.criterionGAN(pred_real_for, self.for_label)

        self.loss_D_real = (self.loss_D_real_cen + self.loss_D_real_back + self.loss_D_real_for) * (1.0 / 3)

        # Combined loss, evenly weighted so each is 1/9 weight
        self.loss_D = (self.loss_D_fake_center_mis_back + self.loss_D_fake_center_mis_for + self.loss_D_fake_back_mis_cen + self.loss_D_fake_back_mis_for +
                self.loss_D_fake_for_mis_cen + self.loss_D_fake_for_mis_back + self.loss_D_real_cen + self.loss_D_real_back + self.loss_D_real_for) * (1.0 / 9)

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        if not self.wo_G_GAN:
            pred_fake = self.netD(self.real_A, self.fake_B)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        else:
            self.loss_G_GAN = 0

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1((self.fake_B, self.fake_B_postnet), self.real_B)

        # Third, silence loss
        self.loss_G_silence = self.criterionL1((self.fake_B, self.fake_B_postnet), torch.zeros_like(self.real_B))

        # loss_G_GAN is adversarial loss, the other two term (loss_G_L1 and loss_G_silence) are reconstruction loss
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 * self.config.lambda_Oriloss + self.loss_G_silence * self.config.lambda_Silenceloss

        self.loss_G.backward()

    def backward_G_pairing_loss(self):
        # First, G(A) should fake the discriminator
        if not self.wo_G_GAN:
        #    # Since the hypothesis is that the generator will generate temporallly aligned audio, use that to fake discriminator
            pred_real_cen = self.netD(self.fake_cen_encoder_output, self.fake_B_cen)  #self.netD(self.real_A_cen, self.real_B_cen)
            self.loss_D_real_cen = self.criterionGAN(pred_real_cen, self.cen_label)

            pred_real_back = self.netD(self.fake_back_encoder_output, self.fake_B_back)  #self.netD(self.real_A_back, self.real_B_back)
            self.loss_D_real_back = self.criterionGAN(pred_real_back, self.back_label)

            pred_real_for = self.netD(self.fake_for_encoder_output, self.fake_B_for)  #self.netD(self.real_A_for, self.real_B_for)
            self.loss_D_real_for = self.criterionGAN(pred_real_for, self.for_label)
            #self.loss_G_GAN = (self.loss_D_real_cen + self.loss_D_real_back + self.loss_D_real_for) * (1.0 / 3)
            self.loss_temporal = (self.loss_D_real_cen + self.loss_D_real_back + self.loss_D_real_for) * (1.0 / 3)
        else:
        #    self.loss_G_GAN = 0
            # Calculate pairing loss, using encoder output as input
            pred_fake_center_mis_back = self.netD(self.fake_cen_encoder_output, self.real_B_cen_mis_back)
            #print("discriminator output shape: ", self.pred_fake_center_mis_back.shape)
            #print("Actual output: ", self.pred_fake_center_mis_back)
            self.loss_D_fake_center_mis_back = self.criterionGAN(pred_fake_center_mis_back, self.cen_mis_back_label)
            #print("pairing loss: ", self.loss_D_fake_center_mis_back)

            pred_fake_center_mis_for = self.netD(self.fake_cen_encoder_output, self.real_B_cen_mis_for)
            self.loss_D_fake_center_mis_for = self.criterionGAN(pred_fake_center_mis_for, self.cen_mis_for_label)

            pred_fake_back_mis_cen = self.netD(self.fake_back_encoder_output, self.real_B_back_mis_cen)
            self.loss_D_fake_back_mis_cen = self.criterionGAN(pred_fake_back_mis_cen, self.back_mis_cen_label)

            pred_fake_back_mis_for = self.netD(self.fake_back_encoder_output, self.real_B_back_mis_for)
            self.loss_D_fake_back_mis_for = self.criterionGAN(pred_fake_back_mis_for, self.back_mis_for_label)

            pred_fake_for_mis_cen = self.netD(self.fake_for_encoder_output, self.real_B_for_mis_cen)
            self.loss_D_fake_for_mis_cen = self.criterionGAN(pred_fake_for_mis_cen, self.for_mis_cen_label)

            pred_fake_for_mis_back = self.netD(self.fake_for_encoder_output, self.real_B_for_mis_back)
            self.loss_D_fake_for_mis_back = self.criterionGAN(pred_fake_for_mis_back, self.for_mis_back_label)

            self.loss_D_fake = (self.loss_D_fake_center_mis_back + self.loss_D_fake_center_mis_for + self.loss_D_fake_back_mis_cen + self.loss_D_fake_back_mis_for +
                    self.loss_D_fake_for_mis_cen + self.loss_D_fake_for_mis_back) * (1.0 / 6)

            # Calculate the loss for real/aligned cases
            pred_real_cen = self.netD(self.fake_cen_encoder_output, self.real_B_cen)
            self.loss_D_real_cen = self.criterionGAN(pred_real_cen, self.cen_label)

            pred_real_back = self.netD(self.fake_back_encoder_output, self.real_B_back)
            self.loss_D_real_back = self.criterionGAN(pred_real_back, self.back_label)

            pred_real_for = self.netD(self.fake_for_encoder_output, self.real_B_for)
            self.loss_D_real_for = self.criterionGAN(pred_real_for, self.for_label)

            self.loss_D_real = (self.loss_D_real_cen + self.loss_D_real_back + self.loss_D_real_for) * (1.0 / 3)

            # Combined temporal misalignment loss, evenly weighted so each is 1/9 weight
            self.loss_temporal = (self.loss_D_fake_center_mis_back + self.loss_D_fake_center_mis_for + self.loss_D_fake_back_mis_cen + self.loss_D_fake_back_mis_for +
                    self.loss_D_fake_for_mis_cen + self.loss_D_fake_for_mis_back + self.loss_D_real_cen + self.loss_D_real_back + self.loss_D_real_for) * (1.0 / 9)

        # Second, G(A) = B, alignment examples
        self.loss_G_L1_cen = self.criterionL1((self.fake_B_cen, self.fake_B_cen_postnet), self.real_B_cen)
        self.loss_G_L1_back = self.criterionL1((self.fake_B_back, self.fake_B_back_postnet), self.real_B_back)
        self.loss_G_L1_for = self.criterionL1((self.fake_B_for, self.fake_B_for_postnet), self.real_B_for)
        self.loss_G_L1 = (self.loss_G_L1_cen + self.loss_G_L1_back + self.loss_G_L1_for) * (1.0 / 3)

        # Third, silence loss
        self.loss_G_silence_cen = self.criterionL1((self.fake_B_cen, self.fake_B_cen_postnet), torch.zeros_like(self.real_B_cen))
        self.loss_G_silence_back = self.criterionL1((self.fake_B_back, self.fake_B_back_postnet), torch.zeros_like(self.real_B_back))
        self.loss_G_silence_for = self.criterionL1((self.fake_B_for, self.fake_B_for_postnet), torch.zeros_like(self.real_B_for))
        self.loss_G_silence = (self.loss_G_silence_cen + self.loss_G_silence_back + self.loss_G_silence_for) * (1.0 / 3)

        # loss_G_GAN is adversarial loss, the other two term (loss_G_L1 and loss_G_silence) are reconstruction loss
        # loss_temporal is the temporal misalignment loss
        self.loss_G = self.loss_temporal * self.config.temporal_alignment_lambda + self.loss_G_L1 * self.config.lambda_Oriloss + self.loss_G_silence * self.config.lambda_Silenceloss

        self.loss_G.backward()

    def optimize_parameters(self):
        self.n_iter += 1

        # Determine which forward function to use
        if config.pairing_loss:
            self.forward_pairing_loss()
        else:
            self.forward()

        # update D
        if not self.wo_G_GAN:
            print("Backpropping through discriminator...")
            if self.n_iter % self.D_interval == 0:
                self.set_requires_grad(self.netD, True)
                self.optimizer_D.zero_grad()

                if config.pairing_loss:
                    self.backward_D_pairing_loss()
                else:
                    self.backward_D()
                self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        
        if config.pairing_loss:
            self.backward_G_pairing_loss()
        else:
            self.backward_G()
        self.optimizer_G.step()
