import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import numpy as np
import math
import quadrilinear4d

def weights_init_normal_generator(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def discriminator_block(in_filters, out_filters, normalization=False):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))

    return layers


class Generator_for_bias(nn.Module):
    def __init__(self, in_channels=3):
        super(Generator_for_bias, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256,256),mode='bilinear'),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32, normalization=True),
            *discriminator_block(32, 64, normalization=True),
            *discriminator_block(64, 128, normalization=True),
            *discriminator_block(128, 128),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 12, 8, padding=0),
        )

    def forward(self, img_input):
        return self.model(img_input)


def generator_block(in_filters, out_filters, normalization=False):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=1, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
        #layers.append(nn.BatchNorm2d(out_filters))
    return layers


class Generator_for_info(nn.Module):
    def __init__(self, in_channels=3):
        super(Generator_for_info, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
        )

        self.mid_layer = nn.Sequential(
            *generator_block(16, 16, normalization=True),
            *generator_block(16, 16, normalization=True),
            *generator_block(16, 16, normalization=True),
        )
        
        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )


    def forward(self, img_input):
        x = self.input_layer(img_input)
        identity = x
        out = self.mid_layer(x)
        out += identity
        out = self.output_layer(out)
        return out



class Generator4DLUT_identity(nn.Module):
    def __init__(self, dim=17):
        super(Generator4DLUT_identity, self).__init__()
        if dim == 17:
            file = open("Identity4DLUT17.txt", 'r')
        elif dim == 33:
            file = open("Identity4DLUT33.txt", 'r')
        lines = file.readlines()
        buffer = np.zeros((3,2,dim,dim,dim), dtype=np.float32)
        for p in range(0,2):
            for i in range(0,dim):
                for j in range(0,dim):
                    for k in range(0,dim):
                        n = p * dim*dim*dim + i * dim*dim + j*dim + k
                        x = lines[n].split()
                        buffer[0,p,i,j,k] = float(x[0])
                        buffer[1,p,i,j,k] = float(x[1])
                        buffer[2,p,i,j,k] = float(x[2])
        self.LUT_en = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))
        self.QuadrilinearInterpolation_4D = QuadrilinearInterpolation_4D()

    def forward(self, x):
        _, output = self.QuadrilinearInterpolation_4D(self.LUT_en, x)
        return output




class QuadrilinearInterpolation_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut, x):
        x = x.contiguous()
        output = x.new(x.size()[0],3,x.size()[2],x.size()[3])
        dim = lut.size()[-1]
        shift = 2 * dim ** 3
        binsize = 1.000001 / (dim-1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)
        assert 1 == quadrilinear4d.forward(lut, 
                                      x, 
                                      output,
                                      dim, 
                                      shift, 
                                      binsize, 
                                      W, 
                                      H, 
                                      batch)
        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]
        ctx.save_for_backward(*variables)
        
        return lut, output
    
    @staticmethod
    def backward(ctx, lut_grad, x_grad):
        # print()
        x_grad = x_grad.contiguous()
        output_grad = x_grad.new(x_grad.size()[0],4,x_grad.size()[2],x_grad.size()[3]).fill_(0)
        output_grad[:,1:,:,:] = x_grad
        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])

        assert 1 == quadrilinear4d.backward(x, 
                                       output_grad,
                                       lut,
                                       lut_grad,
                                       dim, 
                                       shift, 
                                       binsize, 
                                       W, 
                                       H, 
                                       batch)
        return lut_grad, output_grad


class QuadrilinearInterpolation_4D(torch.nn.Module):
    def __init__(self):
        super(QuadrilinearInterpolation_4D, self).__init__()

    def forward(self, lut, x):
        return QuadrilinearInterpolation_Function.apply(lut, x)


class TV_4D(nn.Module):
    def __init__(self, dim=17):
        super(TV_4D,self).__init__()

        self.weight_r = torch.ones(3,2,dim,dim,dim-1, dtype=torch.float)
        self.weight_r[:,:,:,:,(0,dim-2)] *= 2.0
        self.weight_g = torch.ones(3,2,dim,dim-1,dim, dtype=torch.float)
        self.weight_g[:,:,:,(0,dim-2),:] *= 2.0
        self.weight_b = torch.ones(3,2,dim-1,dim,dim, dtype=torch.float)
        self.weight_b[:,:,(0,dim-2),:,:] *= 2.0
        self.relu = torch.nn.ReLU()

    def forward(self, LUT):
        dif_context = LUT.LUT_en[:,:-1,:,:,:] - LUT.LUT_en[:,1:,:,:,:]
        dif_r = LUT.LUT_en[:,:,:,:,:-1] - LUT.LUT_en[:,:,:,:,1:]
        dif_g = LUT.LUT_en[:,:,:,:-1,:] - LUT.LUT_en[:,:,:,1:,:]
        dif_b = LUT.LUT_en[:,:,:-1,:,:] - LUT.LUT_en[:,:,1:,:,:]
        tv = torch.mean(torch.mul((dif_r ** 2),self.weight_r)) + torch.mean(torch.mul((dif_g ** 2),self.weight_g)) + torch.mean(torch.mul((dif_b ** 2),self.weight_b)) 
        mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b)) \
             + torch.mean(self.relu(dif_context))
        return tv, mn

