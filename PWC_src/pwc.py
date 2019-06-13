#!/usr/bin/env python

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from .correlation_package import correlation


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()

        self.moduleOne = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleTwo = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleThr = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFou = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFiv = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleSix = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

    def forward(self, tensorInput):
        tensorOne = self.moduleOne(tensorInput)
        tensorTwo = self.moduleTwo(tensorOne)
        tensorThr = self.moduleThr(tensorTwo)
        tensorFou = self.moduleFou(tensorThr)
        tensorFiv = self.moduleFiv(tensorFou)
        tensorSix = self.moduleSix(tensorFiv)

        return [ tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix ]


class Backward(nn.Module):
    def __init__(self):
        super(Backward, self).__init__()

    def forward(self, tensorInput, tensorFlow):
        if hasattr(self, 'tensorPartial') == False or self.tensorPartial.size(0) != tensorFlow.size(0) or self.tensorPartial.size(2) != tensorFlow.size(2) or self.tensorPartial.size(3) != tensorFlow.size(3):
            self.tensorPartial = tensorFlow.new_ones(tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3))

        if hasattr(self, 'tensorGrid') == False or self.tensorGrid.size(0) != tensorFlow.size(0) or self.tensorGrid.size(2) != tensorFlow.size(2) or self.tensorGrid.size(3) != tensorFlow.size(3):
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
            self.tensorGrid = torch.cat([tensorHorizontal, tensorVertical], 1).cuda()

        tensorInput = torch.cat([tensorInput, self.tensorPartial], 1)
        tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :]/((tensorInput.size(3)-1.0)/2.0), tensorFlow[:, 1:2, :, :]/((tensorInput.size(2)-1.0)/2.0)], 1)

        tensorOutput = F.grid_sample(input=tensorInput, grid=(self.tensorGrid + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
        tensorMask = tensorOutput[:, -1:, :, :]; tensorMask[tensorMask > 0.999] = 1.0; tensorMask[tensorMask < 1.0] = 0.0

        return tensorOutput[:, :-1, :, :] * tensorMask


class Decoder(nn.Module):
    def __init__(self, intLevel):
        super(Decoder, self).__init__()

        intPrevious = [None, None, 81+32+2+2, 81+64+2+2, 81+96+2+2, 81+128+2+2, 81, None][intLevel+1]
        intCurrent = [None, None, 81+32+2+2, 81+64+2+2, 81+96+2+2, 81+128+2+2, 81, None][intLevel+0]

        if intLevel < 6: self.moduleUpflow = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
        if intLevel < 6: self.moduleUpfeat = nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)

        if intLevel < 6: self.dblBackward = [None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel+1]
        if intLevel < 6: self.moduleBackward = Backward()

        self.moduleCorrelation = correlation.Correlation()
        self.moduleCorreleaky = nn.LeakyReLU(inplace=False, negative_slope=0.1)

        self.moduleOne = nn.Sequential(
            nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleTwo = nn.Sequential(
            nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleThr = nn.Sequential(
            nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFou = nn.Sequential(
            nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFiv = nn.Sequential(
            nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleSix = nn.Sequential(
            nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, tensorFirst, tensorSecond, objectPrevious):
        tensorFlow = None
        tensorFeat = None

        if objectPrevious is None:
            tensorFlow = None
            tensorFeat = None

            tensorVolume = self.moduleCorreleaky(self.moduleCorrelation(tensorFirst, tensorSecond))
            tensorFeat = torch.cat([tensorVolume], 1)

        elif objectPrevious is not None:
            tensorFlow = self.moduleUpflow(objectPrevious['tensorFlow'])
            tensorFeat = self.moduleUpfeat(objectPrevious['tensorFeat'])
            tensorVolume = self.moduleCorreleaky(self.moduleCorrelation(tensorFirst, self.moduleBackward(tensorSecond, tensorFlow*self.dblBackward)))
            tensorFeat = torch.cat([tensorVolume, tensorFirst, tensorFlow, tensorFeat], 1)

        tensorFeat = torch.cat([self.moduleOne(tensorFeat), tensorFeat], 1)
        tensorFeat = torch.cat([self.moduleTwo(tensorFeat), tensorFeat], 1)
        tensorFeat = torch.cat([self.moduleThr(tensorFeat), tensorFeat], 1)
        tensorFeat = torch.cat([self.moduleFou(tensorFeat), tensorFeat], 1)
        tensorFeat = torch.cat([self.moduleFiv(tensorFeat), tensorFeat], 1)
        tensorFlow = self.moduleSix(tensorFeat)

        return {
            'tensorFlow': tensorFlow,
            'tensorFeat': tensorFeat
        }


class Refiner(nn.Module):
    def __init__(self):
        super(Refiner, self).__init__()

        self.moduleMain = nn.Sequential(
            nn.Conv2d(in_channels=81+32+2+2+128+128+96+64+32, out_channels=128, kernel_size=3, stride=1, padding=1,  dilation=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2,  dilation=2),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4,  dilation=4),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8,  dilation=8),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16,  dilation=16),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1,  dilation=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1,  dilation=1)
        )

    def forward(self, tensorInput):
        return self.moduleMain(tensorInput)


class PWC_Net(nn.Module):
    def __init__(self, model_path=None):
        super(PWC_Net, self).__init__()
        self.model_path = model_path

        self.moduleExtractor = Extractor()
        self.moduleTwo = Decoder(2)
        self.moduleThr = Decoder(3)
        self.moduleFou = Decoder(4)
        self.moduleFiv = Decoder(5)
        self.moduleSix = Decoder(6)
        self.moduleRefiner = Refiner()
        self.load_state_dict(torch.load(self.model_path))

    def forward(self, tensorFirst, tensorSecond):
        tensorFirst = self.moduleExtractor(tensorFirst)
        tensorSecond = self.moduleExtractor(tensorSecond)

        objectEstimate = self.moduleSix(tensorFirst[-1], tensorSecond[-1], None)
        objectEstimate = self.moduleFiv(tensorFirst[-2], tensorSecond[-2], objectEstimate)
        objectEstimate = self.moduleFou(tensorFirst[-3], tensorSecond[-3], objectEstimate)
        objectEstimate = self.moduleThr(tensorFirst[-4], tensorSecond[-4], objectEstimate)
        objectEstimate = self.moduleTwo(tensorFirst[-5], tensorSecond[-5], objectEstimate)

        return objectEstimate['tensorFlow'] + self.moduleRefiner(objectEstimate['tensorFeat'])


if __name__ == '__main__':
    net = PWC_Net(model_path='models/sintel.pytorch')
    net.cuda()

