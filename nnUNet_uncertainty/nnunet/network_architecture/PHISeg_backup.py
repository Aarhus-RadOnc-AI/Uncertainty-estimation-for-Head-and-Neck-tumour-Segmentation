import torch
import torch.nn as nn
import numpy as np

import revtorch as rv

from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def convert_to_onehot_torch(lblmap, nlabels):
    if len(lblmap.shape) == 3:
        # 2D image
        output = torch.zeros((nlabels, lblmap.shape[-2], lblmap.shape[-1]))
        for ii in range(nlabels):
            lbl = (lblmap == ii).view(lblmap.shape[-2], lblmap.shape[-1])
            output[ii, :, :] = lbl
    elif len(lblmap.shape) == 4:
        # ! 3D images from brats are already one hot encoded
        output = torch.zeros((nlabels, lblmap.shape[-3], lblmap.shape[-2], lblmap.shape[-1]))
        for ii in range(nlabels):
            lbl = (lblmap == ii).view(lblmap.shape[-3],lblmap.shape[-2], lblmap.shape[-1])
            output[ii, :, :, :] = lbl
    return output.long()

def convert_batch_to_onehot(lblbatch, nlabels):
    out = []

    for ii in range(lblbatch.shape[0]):
        lbl = convert_to_onehot_torch(lblbatch[ii,...], nlabels)
        # TODO: check change
        out.append(lbl.unsqueeze(dim=0))

    result = torch.cat(out, dim=0)
    return result

class Conv3D(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, kernel_size=3, padding=1, activation=torch.nn.LeakyReLU, norm=torch.nn.InstanceNorm3d,
                 norm_before_activation=True):
        super(Conv3D, self).__init__()

        if kernel_size == 3:
            padding = 1
        else:
            padding = 0

        layers = []
        layers.append(nn.Conv3d(input_dim, output_dim, kernel_size=kernel_size, padding=padding, stride=stride))
        if norm_before_activation:
            layers.append(norm(num_features=output_dim, eps=1e-5, momentum=0.01, affine=True ))
            layers.append(activation())
        else:
            layers.append(activation())
            layers.append(norm(num_features=output_dim, eps=1e-5, momentum=0.01, affine=True))

        self.convolution = nn.Sequential(*layers)

    def forward(self, x):
        return self.convolution(x)


class TransConv3D(nn.Module):
    def __init__(self, input_dim, output_dim, stride=2, kernel_size=2, activation=torch.nn.LeakyReLU, norm=torch.nn.InstanceNorm3d,
                 norm_before_activation=True):
        super(TransConv3D, self).__init__()

        layers = []
        layers.append(nn.ConvTranspose3d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, bias=False))
        if norm_before_activation:
            layers.append(norm(num_features=output_dim, eps=1e-5, momentum=0.01, affine=True ))
            layers.append(activation())
        else:
            layers.append(activation())
            layers.append(norm(num_features=output_dim, eps=1e-5, momentum=0.01, affine=True))

        self.convolution = nn.Sequential(*layers)

    def forward(self, x):
        return self.convolution(x)


class Conv3DSequence(nn.Module):
    """Block with 3D convolutions after each other with ReLU activation"""
    def __init__(self, input_dim, output_dim, kernel=3, depth=2, activation=torch.nn.LeakyReLU, norm=torch.nn.InstanceNorm3d, norm_before_activation=True):
        super(Conv3DSequence, self).__init__()

        assert depth >= 1
        if kernel == 3:
            padding = 1
        else:
            padding = 0

        layers = []
        layers.append(Conv3D(input_dim, output_dim, kernel_size=kernel, padding=padding, activation=activation, norm=norm))

        for i in range(depth-1):
            layers.append(Conv3D(output_dim, output_dim, kernel_size=kernel, padding=padding, activation=activation, norm=norm))

        self.convolution = nn.Sequential(*layers)

    def forward(self, x):
        return self.convolution(x)


class ReversibleSequence(nn.Module):
    """This class implements a a reversible sequence made out of n convolutions with ReLU activation and BN
        There is an initial 1x1 convolution to get to the desired number of channels.
    """
    def __init__(self, input_dim, output_dim, reversible_depth=3):
        super(ReversibleSequence, self).__init__()

        if input_dim  != output_dim:
            self.inital_conv = Conv3D(input_dim, output_dim, kernel_size=1)
        else:
            self.inital_conv = nn.Identity()

        blocks = []
        for i in range(reversible_depth):

            #f and g must both be a nn.Module whos output has the same shape as its input
            f_func = nn.Sequential(Conv3D(output_dim//2, output_dim//2, kernel_size=3, padding=1))
            g_func = nn.Sequential(Conv3D(output_dim//2, output_dim//2, kernel_size=3, padding=1))

            #we construct a reversible block with our F and G functions
            blocks.append(rv.ReversibleBlock(f_func, g_func))

        #pack all reversible blocks into a reversible sequence
        self.sequence = rv.ReversibleSequence(nn.ModuleList(blocks))

    def forward(self, x):
        x = self.inital_conv(x)
        return self.sequence(x)


class DownConvolutionalBlock(nn.Module):
    def __init__(self, input_dim, output_dim, initializers, depth=2, padding=True, pool=True, reversible=False):
        super(DownConvolutionalBlock, self).__init__()

        if depth < 1:
            raise ValueError

        layers = []

        if reversible:
            layers.append(ReversibleSequence(input_dim, output_dim, reversible_depth=1))
        else:
            if pool:
                #layers.append(nn.AvgPool3d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
                layers.append(Conv3D(input_dim, output_dim, kernel_size=2, stride=2, padding=int(padding)))
            else:
                layers.append(Conv3D(input_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))

            if depth > 1:
                for i in range(depth-1):
                    layers.append(Conv3D(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))

        self.layers = nn.Sequential(*layers)

        #self.layers.apply(init_weights)

    def forward(self, x):
        return self.layers(x)


class UpConvolutionalBlock(nn.Module):
    """
        A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
        If bilinear is set to false, we do a transposed convolution instead of upsampling
        """

    def __init__(self, input_dim, output_dim, initializers, padding, bilinear=False, reversible=False):
        super(UpConvolutionalBlock, self).__init__()
        self.bilinear = bilinear

        if self.bilinear:
            if reversible:
                self.upconv_layer = ReversibleSequence(input_dim, output_dim, reversible_depth=1)
            else:
                self.upconv_layer = nn.Sequential(
                    Conv3D(input_dim, output_dim, kernel_size=3, stride=1, padding=1),
                    Conv3D(output_dim, output_dim, kernel_size=3, stride=1, padding=1),
                    )
        else: 
            if reversible:
                self.upconv_layer = ReversibleSequence(input_dim, output_dim, reversible_depth=1)
            else:
                self.upconv_layer = nn.Sequential(
                    TransConv3D(input_dim, output_dim, kernel_size=2, stride=2),
                    Conv3D(output_dim, output_dim, kernel_size=3, stride=1, padding=1),
                    )

    def forward(self, x, bridge):
        if self.bilinear:
            x = nn.functional.interpolate(x, mode='trilinear', scale_factor=2, align_corners=True)
            x = self.upconv_layer(x)
        else:
            x = self.upconv_layer(x)

        assert x.shape[3] == bridge.shape[3]
        assert x.shape[2] == bridge.shape[2]
        out = torch.cat([x, bridge], dim=1)

        return out


class SampleZBlock(nn.Module):
    """
    Performs 2 3X3 convolutions and a 1x1 convolution to mu and sigma which are used as parameters for a Gaussian
    for generating z
    """
    def __init__(self, input_dim, z_dim0=2, depth=2, reversible=False):
        super(SampleZBlock, self).__init__()
        self.input_dim = input_dim

        layers = []

        if reversible:
            layers.append(ReversibleSequence(input_dim, input_dim, reversible_depth=1))
        else:
            for i in range(depth):
                layers.append(Conv3D(input_dim, input_dim, kernel_size=3, padding=1))

        self.conv = nn.Sequential(*layers)

        self.mu_conv = nn.Sequential(nn.Conv3d(input_dim, z_dim0, kernel_size=1))
        self.sigma_conv = nn.Sequential(nn.Conv3d(input_dim, z_dim0, kernel_size=1),
                                        nn.Softplus())

    def forward(self, pre_z):
        pre_z = self.conv(pre_z)
        mu = self.mu_conv(pre_z)
        sigma = self.sigma_conv(pre_z)

        z = mu + sigma * torch.randn_like(sigma, dtype=torch.float32)

        return mu, sigma, z


class Posterior(nn.Module):
    """
    Posterior network of the PHiSeg Module
    For each latent level a sample of the distribution of the latent level is returned
    Parameters
    ----------
    input_channels : Number of input channels, 1 for greyscale,
    is_posterior: if True, the mask is concatenated to the input of the encoder, causing it to be a ConditionalVAE
    """
    def __init__(self,
                 input_channels,
                 num_classes,
                 num_filters,
                 latent_levels,
                 initializers=None,
                 padding=True,
                 is_posterior=True,
                 reversible=False):
        super(Posterior, self).__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters

        self.latent_levels = latent_levels
        self.resolution_levels = len(num_filters)
        self.lvl_diff = self.resolution_levels - self.latent_levels
        self.num_classes = num_classes
        self.padding = padding
        self.activation_maps = []

        if is_posterior:
            # increase input channel by two to accomodate place for mask in one hot encoding
            self.input_channels += num_classes

        self.contracting_path = nn.ModuleList()

        for i in range(self.resolution_levels):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]

            pool = False if i == 0 else True

            self.contracting_path.append(DownConvolutionalBlock(input,
                                                                output,
                                                                initializers,
                                                                depth=2,
                                                                padding=padding,
                                                                pool=pool,
                                                                reversible=reversible)
                                         )

        self.upsampling_path = nn.ModuleList()

        for i in reversed(range(self.latent_levels)):  # iterates from [latent_levels -1, ... ,0]
            input = 2
            output = self.num_filters[0]*2
            self.upsampling_path.append(UpConvolutionalBlock(input, output, initializers, padding, reversible=reversible))

        self.sample_z_path = nn.ModuleList()
        for i in reversed(range(self.latent_levels)):
            input = 2*self.num_filters[0] + self.num_filters[i + self.lvl_diff]
            if i == self.latent_levels - 1:
                input = self.num_filters[i + self.lvl_diff]
                self.sample_z_path.append(SampleZBlock(input, depth=2, reversible=reversible))
            else:
                self.sample_z_path.append(SampleZBlock(input, depth=2, reversible=reversible))

    def forward(self, patch, segm=None, training_prior=False, z_list=None):

        if segm is not None:
            with torch.no_grad(): #segm[0].max().item()+1))\
                segm_one_hot = convert_batch_to_onehot(segm, nlabels=int(self.num_classes)) \
                    .to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

                segm_one_hot = segm_one_hot.float()
            patch = torch.cat([patch, torch.add(segm_one_hot, -0.5)], dim=1)

        blocks = []
        z = [None] * self.latent_levels # contains all hidden z
        sigma = [None] * self.latent_levels
        mu = [None] * self.latent_levels

        x = patch
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path) - 1:
                blocks.append(x)

        pre_conv = x
        for i, sample_z in enumerate(self.sample_z_path):
            if i != 0:
                pre_conv = self.upsampling_path[i-1](z[-i], blocks[-i])
            mu[-i-1], sigma[-i-1], z[-i-1] = self.sample_z_path[i](pre_conv)
            if training_prior:
                z[-i-1] = z_list[-i-1]

        del blocks

        return z, mu, sigma


def increase_resolution(times, input_dim, output_dim):
    """ Increase the resolution by n time for the beginning of the likelihood path"""
    module_list = []
    for i in range(times):
        module_list.append(nn.Upsample(
                    mode='trilinear',
                    scale_factor=2,
                    align_corners=True))
        if i != 0:
            input_dim = output_dim
        module_list.append(Conv3DSequence(input_dim=input_dim, output_dim=output_dim, depth=1)) # change from Conv3DSequence to Conv3D

    return nn.Sequential(*module_list)


class Likelihood(nn.Module):
    # TODO: add latent_level and resolution_levels to exp_config file
    def __init__(self,
                 input_channels,
                 num_classes,
                 num_filters,
                 latent_levels=5,
                 image_size=(128,128,1),
                 reversible=False,
                 initializers=None,
                 apply_last_layer=True,
                 padding=True):
        super(Likelihood, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters

        self.latent_levels = latent_levels
        self.resolution_levels = len(num_filters)
        self.lvl_diff = self.resolution_levels - latent_levels

        self.image_size = image_size
        self.reversible= reversible

        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        # LIKELIHOOD
        self.likelihood_ups_path = nn.ModuleList()
        self.likelihood_post_ups_path = nn.ModuleList()

        # final_nolinear acitvation funciton
        self.final_nonlin = softmax_helper
        # path for upsampling
        for i in reversed(range(self.latent_levels)):
            input = self.num_filters[i]
            output = self.num_filters[i]
            if reversible:
                self.likelihood_ups_path.append(ReversibleSequence(input_dim=2, output_dim=input, reversible_depth=1))
            else:
                self.likelihood_ups_path.append(Conv3DSequence(input_dim=2, output_dim=input, depth=2)) # change from Conv3DSequence to Conv3D

            self.likelihood_post_ups_path.append(increase_resolution(times=self.lvl_diff, input_dim=input, output_dim=input))

        # path after concatenation
        self.likelihood_post_c_path = nn.ModuleList()
        for i in range(latent_levels - 1):
            input = self.num_filters[i] + self.num_filters[i + 1 + self.lvl_diff]
            output = self.num_filters[i + self.lvl_diff]

            if reversible:
                self.likelihood_post_c_path.append(ReversibleSequence(input_dim=input, output_dim=output, reversible_depth=1))
            else:
                self.likelihood_post_c_path.append(Conv3DSequence(input_dim=input, output_dim=output, depth=2)) # change from Conv3DSequence to Conv3D

        self.s_layer = nn.ModuleList()
        output = self.num_classes
        for i in reversed(range(self.latent_levels)):
            input = self.num_filters[i + self.lvl_diff]
            self.s_layer.append(Conv3DSequence(
                input_dim=input, output_dim=output, depth=1, kernel=1, activation=torch.nn.Identity, norm=torch.nn.Identity)) # change from Conv3DSequence to Conv3D

    def forward(self, z): # , lvl1_skip
        """Likelihood network which takes list of latent variables z with dimension latent_levels"""
        s = [None] * self.latent_levels
        s_deep_supervision = [None] * self.latent_levels

        #s_full = [None] * self.latent_levels 
        post_z = [None] * self.latent_levels
        post_c = [None] * self.latent_levels

        # start from the downmost layer and the last filter
        for i in range(self.latent_levels):
            assert z[-i-1].shape[1] == 2
            assert z[-i-1].shape[2] == self.image_size[1] * 2**(-self.resolution_levels + 1 + i)
            post_z[-i - 1] = self.likelihood_ups_path[i](z[-i - 1])

            post_z[-i - 1] = self.likelihood_post_ups_path[i](post_z[-i - 1])
            assert post_z[-i - 1].shape[2] == self.image_size[1] * 2 ** (-self.latent_levels + i + 1)
            assert post_z[-i-1].shape[1] == self.num_filters[-i-1 - self.lvl_diff], '{} != {}'.format(post_z[-i-1].shape[1],self.num_filters[-i-1])

        post_c[self.latent_levels - 1] = post_z[self.latent_levels - 1]

        for i in reversed(range(self.latent_levels - 1)):
            ups_below = nn.functional.interpolate(
                post_c[i+1],
                mode='trilinear',
                scale_factor=2,
                align_corners=True)

            assert post_z[i].shape[3] == ups_below.shape[3]
            assert post_z[i].shape[2] == ups_below.shape[2]

            # Reminder: Pytorch standard is NCHW, TF NHWC
            concat = torch.cat([post_z[i], ups_below], dim=1)

            post_c[i] = self.likelihood_post_c_path[i](concat)
        #print('likelihood self.image_size', self.image_size)
        for i, block in enumerate(self.s_layer):
            #s_in = self.final_nonlin(block(post_c[-i-1])) # add activation in the last layer
            s_in = block(post_c[-i-1]) # no activation in the last layer  
            s[-i-1] = nn.functional.interpolate(s_in, size=[self.image_size[1],self.image_size[2], self.image_size[3]], mode='nearest')
            s_deep_supervision[-i-1] = s_in

        return s, s_deep_supervision


class PHISeg3D(SegmentationNetwork):
    """
    A PHISeg (https://arxiv.org/abs/1906.04045) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in PHISeg)
    padding: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self,
                 input_channels,
                 num_classes,
                 num_filters,
                 latent_levels=5,
                 image_size=(128,128,1),
                 reversible=False,
                 apply_last_layer=True,
                 exponential_weighting=True,
                 padding=True):
        super(PHISeg3D, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters

        self.latent_levels = latent_levels
        self.image_size = image_size

        self.loss_tot = 0

        self.loss_dict={}
        self.kl_divergence_loss_weight = 1.0

        self.beta = 1.0

        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        self.exponential_weighting = exponential_weighting
        self.exponential_weight = 4 # default was 4
        self.residual_multinoulli_loss_weight = 1.0

        self.kl_divergence_loss = 0
        self.reconstruction_loss = 0

        self.posterior = Posterior(input_channels, num_classes, num_filters, latent_levels=latent_levels,
                                   initializers=None, padding=True, reversible=reversible)
        self.likelihood = Likelihood(input_channels, num_classes, num_filters,latent_levels=latent_levels,
                                     initializers=None, apply_last_layer=True, padding=True, image_size=self.image_size,
                                     reversible=reversible)
        self.prior = Posterior(input_channels, num_classes, num_filters, latent_levels=latent_levels,
                               initializers=None, padding=True, is_posterior=False, reversible=reversible)

        self.s_out_list = [None] * self.latent_levels
        self.s_out_list_with_softmax = [None] * self.latent_levels
        self.s_deep_supervision = [None] * self.latent_levels
        self.segm = None
    def sample_posterior(self):
        z_sample = [None] * self.latent_levels
        mu = self.posterior_mu
        sigma = self.posterior_sigma
        for i, _ in enumerate(z_sample):
            z_sample[i] = mu[i] + sigma[i] * torch.randn_like(sigma[i])

        return z_sample

    def sample_prior(self):
        z_sample = [None] * self.latent_levels
        mu = self.prior_mu
        sigma = self.prior_sigma
        for i, _ in enumerate(z_sample):
            z_sample[i] = mu[i] + sigma[i] * torch.randn_like(sigma[i])
        return z_sample

    def sample(self, testing=True):
        if testing:
            sample, _ = self.reconstruct(self.sample_prior(), use_softmax=False)
            return sample
        else:
            raise NotImplementedError

    def reconstruct(self, z_posterior, use_softmax=True):
        layer_recon, _ = self.likelihood(z_posterior)
        return self.accumulate_output(layer_recon, use_softmax=use_softmax), layer_recon

    def accumulate_output(self, output_list, use_softmax=False):
        s_accum = output_list[-1]
        for i in range(len(output_list) - 1):
            s_accum += output_list[i]
        if use_softmax:
            return torch.nn.functional.softmax(s_accum, dim=1)
        return s_accum

    def KL_two_gauss_with_diag_cov(self, mu0, sigma0, mu1, sigma1):

        sigma0_fs = torch.mul(torch.flatten(sigma0, start_dim=1), torch.flatten(sigma0, start_dim=1))
        sigma1_fs = torch.mul(torch.flatten(sigma1, start_dim=1), torch.flatten(sigma0, start_dim=1))

        logsigma0_fs = torch.log(sigma0_fs + 1e-10)
        logsigma1_fs = torch.log(sigma1_fs + 1e-10)

        mu0_f = torch.flatten(mu0, start_dim=1)
        mu1_f = torch.flatten(mu1, start_dim=1)

        return torch.mean(
            0.5*torch.sum(
                torch.div(
                    sigma0_fs + torch.mul((mu1_f - mu0_f), (mu1_f - mu0_f)),
                    sigma1_fs + 1e-10)
                + logsigma1_fs - logsigma0_fs - 1, dim=1)
        )

    def calculate_hierarchical_KL_div_loss(self):

        prior_sigma_list = self.prior_sigma
        prior_mu_list = self.prior_mu
        posterior_sigma_list = self.posterior_sigma
        posterior_mu_list = self.posterior_mu

        if self.exponential_weighting:
            level_weights = [self.exponential_weight ** i for i in list(range(self.latent_levels))]
        else:
            level_weights = [1] * self.exp_config.latent_levels

        for ii, mu_i, sigma_i in zip(reversed(range(self.latent_levels)),
                                     reversed(posterior_mu_list),
                                     reversed(posterior_sigma_list)):

            self.loss_dict['KL_divergence_loss_lvl%d' % ii] = level_weights[ii]*self.KL_two_gauss_with_diag_cov(
                mu_i,
                sigma_i,
                prior_mu_list[ii],
                prior_sigma_list[ii])

            self.loss_tot += self.kl_divergence_loss_weight * self.loss_dict['KL_divergence_loss_lvl%d' % ii]

        return self.loss_tot

    def multinoulli_loss(self, reconstruction, target):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        batch_size = reconstruction.shape[0]

        recon_flat = reconstruction.view(batch_size, self.num_classes, -1)
        target_flat = target.view(batch_size, -1).long()
        return torch.mean(
            torch.sum(criterion(target=target_flat, input=recon_flat), axis=1)
        )

    def residual_multinoulli_loss(self, reconstruction, target):

        self.s_accumulated = [None] * self.latent_levels
        loss_tot = 0

        criterion = self.multinoulli_loss

        for ii, s_ii in zip(reversed(range(self.latent_levels)),
                            reversed(reconstruction)):

            if ii == self.latent_levels-1:

                self.s_accumulated[ii] = s_ii
                self.loss_dict['residual_multinoulli_loss_lvl%d' % ii] = criterion(self.s_accumulated[ii], target)

            else:

                self.s_accumulated[ii] = self.s_accumulated[ii+1] + s_ii
                self.loss_dict['residual_multinoulli_loss_lvl%d' % ii] = criterion(self.s_accumulated[ii], target)

            self.loss_tot += self.residual_multinoulli_loss_weight * self.loss_dict['residual_multinoulli_loss_lvl%d' % ii]
        return self.loss_tot

    def kl_divergence(self):
        loss = self.calculate_hierarchical_KL_div_loss()
        return loss

    def elbo(self, reconstruct_posterior_mean=False):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """
        self.loss_tot = 0
        z_posterior = self.posterior_latent_space

        self.kl_divergence_loss = self.kl_divergence()

        # Here we use the posterior sample sampled above
        #self.reconstruction, layer_reconstruction = self.reconstruct(z_posterior=z_posterior, use_softmax=False)


        #self.reconstruction_loss = self.residual_multinoulli_loss(reconstruction=self.s_out_list, target=self.segm)

        #return self.reconstruction_loss + self.kl_divergence_loss_weight * self.kl_divergence_loss
        return self.loss_tot

    def loss(self):
        return self.elbo()

    def forward(self, patch, mask, training=False):
        #print('self.image_size',self.image_size)
        self.segm = mask
        if training:
            self.posterior_latent_space, self.posterior_mu, self.posterior_sigma  = self.posterior(patch, mask) #lvl1_skip
            self.prior_latent_space, self.prior_mu, self.prior_sigma = self.prior(patch, training_prior=True, z_list=self.posterior_latent_space)
            self.s_out_list, self.s_deep_supervision = self.likelihood(self.posterior_latent_space)
        else:
            #print('this is a validation process')
            #self.posterior_latent_space, self.posterior_mu, self.posterior_sigma = self.posterior(patch, mask)
            self.prior_latent_space, self.prior_mu, self.prior_sigma = self.prior(patch, training_prior=False)
            self.s_out_list, self.s_deep_supervision = self.likelihood(self.prior_latent_space)

        return self.s_out_list, self.s_deep_supervision
        
class PHISeg(SegmentationNetwork):
    """
    PHISeg-3D 
    """
    def __init__(self, input_channels=None, num_classes=None, patch_size=False, deep_supervision=False):
        super().__init__()
        self.conv_op == nn.Conv3d
        self.PHISeg3D = PHISeg3D(input_channels,
                        num_classes,
                        num_filters = [32, 64, 128, 320, 320],
                        latent_levels=5,
                        image_size=patch_size,
                        reversible=False,
                        apply_last_layer=False,
                        exponential_weighting=True,
                        padding=True)

        self.num_classes = num_classes
        self.do_ds = deep_supervision

    def forward(self, patch, mask=None, training=False):
        
        seg_output, s_deep_supervision = self.PHISeg3D(patch, mask, training=training)

        if self.do_ds:
            if len(s_deep_supervision)> 5:
                return s_deep_supervision[:5]
            else:
                return s_deep_supervision
        else:
            output = self.PHISeg3D.accumulate_output(seg_output, use_softmax=False)
        return output


def train():

    net = PHISeg(input_channels=2, num_classes=1, patch_size=[2,128,128,128], deep_supervision=True)
    model = net.cuda()

    x = torch.rand((2,2,128,128,128)).cuda()
    for i in range(2):
        y = model(x)
        print(len(y), y[0].shape)
        for ds in y: 
            print(ds.shape)

if __name__ == "__main__":
    train()
