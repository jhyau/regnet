import random
from torch.utils.tensorboard import SummaryWriter

class RegnetLogger(SummaryWriter):
    def __init__(self, logdir, exclude_D_r_f=False, exclude_gan_loss=False):
        super(RegnetLogger, self).__init__(logdir)
        self.exclude_D_r_f = exclude_D_r_f
        self.exclude_gan_loss = exclude_gan_loss

    def log_training(self, model, reduced_loss, learning_rate, duration,
                     iteration):
        self.add_scalar("training.loss", reduced_loss, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        #self.add_scalar("training.loss_G", model.loss_G, iteration)
        self.add_scalar("training.loss_L1", model.loss_L1, iteration)
        #self.add_scalar("training.loss_temporal", model.loss_temporal, iteration)

        #if not self.exclude_gan_loss:
        #    self.add_scalar("training.loss_G_GAN", model.loss_G_GAN, iteration)
        
        #self.add_scalar("training.loss_G_L1", model.loss_G_L1, iteration)
        #self.add_scalar("training.loss_G_silence", model.loss_G_silence, iteration)
        
        #if not self.exclude_gan_loss:
        #    self.add_scalar("training.loss_D", model.loss_D, iteration)
        #    self.add_scalar("training.loss_D_fake", model.loss_D_fake, iteration)
        #    self.add_scalar("training.loss_D_real", model.loss_D_real, iteration)

        if not self.exclude_D_r_f:
            #self.add_scalar("training.score_D_r-f", (model.pred_real - model.pred_fake).mean(), iteration)
            self.add_scalar("training.real-fake", (model.decoder_output - model.gt_raw_freqs).mean(), iteration)
        self.add_scalar("duration", duration, iteration)

    def log_testing(self, reduced_loss, epoch):
        self.add_scalar("testing.loss", reduced_loss, epoch)

    def log_plot(self, model, iteration, split="train"):
        output = model.fake_B
        output_postnet = model.fake_B_postnet
        target = model.real_B
        video_name = model.video_name

        idx = random.randint(0, output.size(0) - 1)

        self.add_image(
            "mel_spectrogram_{}".format(mode),
            plot_spectrogram(target[idx].data.cpu().numpy(),
                             output[idx].data.cpu().numpy(),
                             output_postnet[idx].data.cpu().numpy(),
                             video_name[idx], mode),
            iteration)
