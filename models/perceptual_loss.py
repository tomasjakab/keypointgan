from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np
import scipy.misc
import scipy.io
from networks import NormalizedLoss



def conv(inputs, filters):
    return nn.Conv2d(
        inputs, filters, kernel_size=3, stride=1, padding=1, bias=True)

def pool():
    return nn.MaxPool2d(kernel_size=2, stride=2)


class VGG19(nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()
        self.conv1_1 = conv(3, 64)
        self.conv1_2 = conv(64, 128)
        self.pool1 = pool()
        self.conv2_1 = conv(128, 128)
        self.conv2_2 = conv(128, 256)
        self.pool2 = pool()
        self.conv3_1 = conv(256, 256)
        self.conv3_2 = conv(256, 256)
        self.conv3_3 = conv(256, 256)
        self.conv3_4 = conv(256, 512)
        self.pool3 = pool()
        self.conv4_1 = conv(512, 512)
        self.conv4_2 = conv(512, 512)
        self.conv4_3 = conv(512, 512)
        self.conv4_4 = conv(512, 512)
        self.pool4 = pool()
        self.conv5_1 = conv(512, 512)
        self.conv5_2 = conv(512, 512)
        self.conv5_3 = conv(512, 512)
        self.conv5_4 = conv(512, 512)
        self.pool5 = pool()


    def forward(self, x):
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(conv1_1)
        pool1 = self.pool1(conv1_2)
        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2 = self.pool2(conv2_2)
        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(conv3_2)
        conv3_4 = self.conv3_4(conv3_3)
        pool3 = self.pool3(conv3_4)
        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        conv4_3 = self.conv4_3(conv4_2)
        conv4_4 = self.conv4_4(conv4_3)
        pool4 = self.pool4(conv4_4)
        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)
        conv5_3 = self.conv5_3(conv5_2)
        conv5_4 = self.conv5_4(conv5_3)
        pool5 = self.pool5(conv5_4)

        return x, conv1_2, conv2_2, conv3_2, conv4_2, conv5_2


class PerceptualLoss(nn.Module):
    def __init__(self, vgg19_path):
        super(PerceptualLoss, self).__init__()

        net = VGG19()
        net = net.cuda()

        vgg_rawnet = scipy.io.loadmat(vgg19_path)
        vgg_layers = vgg_rawnet['layers'][0]

        #Weight initialization according to the pretrained VGG Very deep 19 network Network weights
        layers = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        att = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2',
              'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
              'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
        filt = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
        for l in range(len(layers)):
            getattr(net, att[l]).weight = nn.Parameter(torch.from_numpy(
                vgg_layers[layers[l]][0][0][2][0][0]).permute(3, 2, 0, 1).cuda(), requires_grad=False)
            getattr(net, att[l]).bias = nn.Parameter(torch.from_numpy(
                vgg_layers[layers[l]][0][0][2][0][1]).view(filt[l]).cuda(), requires_grad=False)

        self.net = net

        self.n_layers = 6
        self.losses = [NormalizedLoss(nn.MSELoss(), mu=0.99) for _ in range(self.n_layers)]


    def forward(self, input, target):
        # FIXME: how to handle normalized inputs
        input = ((input + 1.0) / 2.0) * 255.0
        target = ((target + 1.0) / 2.0) * 255.0

        mean = np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))
        mean = torch.from_numpy(mean).float().permute(0,3,1,2).cuda()

        input_f = self.net(input - mean)
        target_f = self.net(target - mean)

        # normalize
        # layer_w = [1.0, 1.6, 2.3, 1.8, 2.8, 0.008]
        # input_f = [f / torch.norm(f.view(f.shape[0], -1, 1, 1), p=2, dim=1, keepdim=True) for f in input_f]
        # target_f = [f / torch.norm(f.view(f.shape[0], -1, 1, 1), p=2, dim=1, keepdim=True) for f in target_f]

        losses = []
        for x, y, loss_fn in zip(input_f, target_f, self.losses):
            losses.append(loss_fn(x, y))
        loss = torch.mean(torch.stack(losses))

        return loss
