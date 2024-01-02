import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models_x import *
from datasetsMIT import *

import torch.nn as nn
import torch.nn.functional as F
import torch

# CUDA_VISIBLE_DEVICES
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from, 0 starts from scratch, >0 starts from saved checkpoints")
parser.add_argument("--n_epochs", type=int, default=1000, help="total number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="fiveK", help="name of the dataset")
parser.add_argument("--input_color_space", type=str, default="sRGB", help="input color space: sRGB or XYZ")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--lambda_smooth", type=float, default=0.0001, help="smooth regularization")
parser.add_argument("--lambda_monotonicity", type=float, default=10.0, help="monotonicity regularization")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
parser.add_argument("--output_dir", type=str, default="LUTs/paired/fiveK_randomp_exp1_test1", help="path to save model")
opt = parser.parse_args()

opt.output_dir = opt.output_dir + '_' + opt.input_color_space
print(opt)

os.makedirs("saved_models/%s" % opt.output_dir, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Loss functions
criterion_pixelwise = torch.nn.MSELoss()

# Initialize generator and discriminator
LUT_enhancement = Generator4DLUT_identity()
Generator_bias = Generator_for_bias()
Generator_context = Generator_for_info()
TV4 = TV_4D()
quadrilinear_enhancement_ = QuadrilinearInterpolation_4D() 

if cuda:
    LUT_enhancement = LUT_enhancement.cuda()
    Generator_bias = Generator_bias.cuda()
    Generator_context = Generator_context.cuda()
    criterion_pixelwise.cuda()
    TV4.cuda()
    TV4.weight_r = TV4.weight_r.type(Tensor)
    TV4.weight_g = TV4.weight_g.type(Tensor)
    TV4.weight_b = TV4.weight_b.type(Tensor)

if opt.epoch != 0:
    LUT_enhancements = torch.load("saved_models/%s/4DLUTs_enhancement_%d.pth" % (opt.output_dir, opt.epoch))
    LUT_enhancement.load_state_dict(LUT_enhancements)
    Generator_bias.load_state_dict(torch.load("saved_models/%s/generator_bias_%d.pth" % (opt.output_dir, opt.epoch)))
    Generator_context.load_state_dict(torch.load("saved_models/%s/generator_context_%d.pth" % (opt.output_dir, opt.epoch)))
else:
    # Initialize weights
    Generator_bias.apply(weights_init_normal_generator)
    torch.nn.init.constant_(Generator_bias.model[16].bias.data, 1.0)

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(Generator_bias.parameters(), Generator_context.parameters(), LUT_enhancement.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)) #, LUT3.parameters(), LUT4.parameters()

if opt.input_color_space == 'sRGB':
    dataloader = DataLoader(
        ImageDataset_sRGB("/fivek_dataset/MIT-Adobe5k-UPE/" , mode = "train"),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=1,
    )

    psnr_dataloader = DataLoader(
        ImageDataset_sRGB("/fivek_dataset/MIT-Adobe5k-UPE/" ,  mode="test"),
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )


def generator_train(img):

    context = Generator_context(img)
    pred = Generator_bias(img)

    context = context.new(context.size())

    context = Variable(context.fill_(0).type(Tensor))

    pred = pred.squeeze(2).squeeze(2)
    combine = torch.cat([context,img],1)

    gen_A0 = LUT_enhancement(combine)

    weights_norm = torch.mean(pred ** 2)

    combine_A = img.new(img.size())
    for b in range(img.size(0)):
        combine_A[b,0,:,:] = pred[b,0] * gen_A0[b,0,:,:] + pred[b,1] * gen_A0[b,1,:,:] + pred[b,2] * gen_A0[b,2,:,:] + pred[b,9]
        combine_A[b,1,:,:] = pred[b,3] * gen_A0[b,0,:,:] + pred[b,4] * gen_A0[b,1,:,:] + pred[b,5] * gen_A0[b,2,:,:] + pred[b,10]
        combine_A[b,2,:,:] = pred[b,6] * gen_A0[b,0,:,:] + pred[b,7] * gen_A0[b,1,:,:] + pred[b,8] * gen_A0[b,2,:,:] + pred[b,11]

    return combine_A, weights_norm

def generator_eval(img):
    context = Generator_context(img)
    pred = Generator_bias(img)

    context = context.new(context.size())
    context = Variable(context.fill_(0).type(Tensor))

    pred = pred.squeeze(2).squeeze(2).squeeze(0)   

    combine = torch.cat([context,img],1)
    
    new_LUT_enhancement = LUT_enhancement.LUT_en.new(LUT_enhancement.LUT_en.size())
    new_LUT_enhancement[0] = pred[0] * LUT_enhancement.LUT_en[0] + pred[1] * LUT_enhancement.LUT_en[1] + pred[2] * LUT_enhancement.LUT_en[2] + pred[9]
    new_LUT_enhancement[1] = pred[3] * LUT_enhancement.LUT_en[0] + pred[4] * LUT_enhancement.LUT_en[1] + pred[5] * LUT_enhancement.LUT_en[2] + pred[10]
    new_LUT_enhancement[2] = pred[6] * LUT_enhancement.LUT_en[0] + pred[7] * LUT_enhancement.LUT_en[1] + pred[8] * LUT_enhancement.LUT_en[2] + pred[11]
    
    weights_norm = torch.mean(pred[0] ** 2)
    combine_A = img.new(img.size())
    _, combine_A = quadrilinear_enhancement_(new_LUT_enhancement,combine)

    return combine_A, weights_norm

def calculate_psnr():
    Generator_bias.eval()
    Generator_context.eval()
    avg_psnr = 0
    for i, batch in enumerate(psnr_dataloader):
        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["A_exptC"].type(Tensor))
        fake_B, _ = generator_eval(real_A)
        fake_B = torch.clamp(fake_B,0.0,1.0)
        fake_B = torch.round(fake_B*255)
        real_B = torch.round(real_B*255)
        mse = criterion_pixelwise(fake_B, real_B)
        psnr = 10 * math.log10(255.0 * 255.0 / mse.cpu().detach().item())
        avg_psnr += psnr
    return avg_psnr/ len(psnr_dataloader)

def calculate_ssim():
    Generator_bias.eval()
    Generator_context.eval()
    avg_ssim = 0
    for i, batch in enumerate(psnr_dataloader):
        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["A_exptC"].type(Tensor))
        fake_B, _ = generator_eval(real_A)
        fake_B = torch.clamp(fake_B,0.0,1.0)
        fake_B = fake_B.squeeze(0).cpu().detach().numpy()
        real_B = real_B.squeeze(0).cpu().detach().numpy()
        fake_B = np.swapaxes(np.swapaxes(fake_B, 0, 2), 0, 1)
        real_B = np.swapaxes(np.swapaxes(real_B, 0, 2), 0, 1)
        fake_B = fake_B.astype(np.float32)
        real_B = real_B.astype(np.float32)
        ssim_val = ssim(real_B,fake_B, data_range=real_B.max() - fake_B.min(), multichannel=True, gaussian_weights=True, win_size=11)
        avg_ssim += ssim_val
    return avg_ssim / len(psnr_dataloader)



def visualize_result(epoch):
    """Saves a generated sample from the validation set"""
    Generator_bias.eval()
    Generator_context.eval()
    os.makedirs("images/%s/" % opt.output_dir +str(epoch), exist_ok=True)
    for i, batch in enumerate(psnr_dataloader):
        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["A_exptC"].type(Tensor))
        img_name = batch["input_name"]
        fake_B, _ = generator_eval(real_A)
        fake_B = torch.clamp(fake_B,0.0,1.0)
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -1)
        fake_B = torch.round(fake_B*255)
        real_B = torch.round(real_B*255)
        mse = criterion_pixelwise(fake_B, real_B)
        psnr = 10 * math.log10(255.0 * 255.0 / mse.item())
        save_image(img_sample, "images/%s/%s/%s.jpg" % (opt.output_dir,epoch, img_name[0]+'_'+str(psnr)[:5]), nrow=3, normalize=False)

# ----------
#  Training
# ----------
prev_time = time.time()
max_psnr = 0
max_epoch = 0
for epoch in range(opt.epoch, opt.n_epochs):
    mse_avg = 0
    psnr_avg = 0
    Generator_bias.train()
    Generator_context.train()
    for i, batch in enumerate(dataloader):
        # Model inputs
        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["A_exptC"].type(Tensor))
        # ------------------ 
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        fake_B, weights_norm = generator_train(real_A)

        # Pixel-wise loss
        mse = criterion_pixelwise(fake_B, real_B)

        tv_enhancement, mn_enhancement = TV4(LUT_enhancement)

        tv_cons = tv_enhancement
        mn_cons = mn_enhancement

        # loss = mse
        loss = mse + opt.lambda_smooth * (weights_norm + tv_cons) + opt.lambda_monotonicity * mn_cons
        psnr_avg += 10 * math.log10(1 / mse.item())

        mse_avg += mse.item()

        loss.backward()

        optimizer_G.step()


        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        if i % 500 == 0:
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [psnr: %f, tv: %f, mn: %f] ETA: %s"
                % (epoch,opt.n_epochs,i,len(dataloader),psnr_avg / (i+1),tv_cons, mn_cons, time_left,
                )
            )
        if i % 500 == 0:
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [psnr: %f, tv: %f, mn: %f] ETA: %s"
                % (epoch,opt.n_epochs,i,len(dataloader),psnr_avg / (i+1),tv_cons, mn_cons, time_left,
                )
            )
    avg_ssim = calculate_ssim()
    avg_psnr = calculate_psnr()
    if avg_psnr > max_psnr:
        max_psnr = avg_psnr
        max_epoch = epoch

        LUTs_enhancement = LUT_enhancement.state_dict()
        torch.save(LUTs_enhancement, "saved_models/%s/4DLUTs_enhancement_%d.pth" % (opt.output_dir, epoch))
        torch.save(Generator_bias.state_dict(), "saved_models/%s/generator_bias_%d.pth" % (opt.output_dir, epoch))
        torch.save(Generator_context.state_dict(), "saved_models/%s/generator_context_%d.pth" % (opt.output_dir, epoch))
        file = open('saved_models/%s/result.txt' % opt.output_dir,'a')
        file.write(" [PSNR: %f , SSIM: %f] [epoch: %d]\n"% (max_psnr,avg_ssim, max_epoch))
        file.close()

    sys.stdout.write(" [PSNR: %f, SSIM: %f] [max PSNR: %f, epoch: %d]\n"% (avg_psnr, avg_ssim, max_psnr, max_epoch))
    print(" [PSNR: %f, SSIM: %f] [max PSNR: %f, epoch: %d]\n"% (avg_psnr, avg_ssim, max_psnr, max_epoch))

    if (epoch+1) % 100 == 0:
       visualize_result(epoch+1)

