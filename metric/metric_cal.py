import numpy as np
import scipy.misc
import cv2
import os
import glob

import matplotlib.pyplot as plt
import sys
import math
from PIL import Image


# from TESTGAN import Model
from AG import avegrad
from NIQE import niqe
from VIFP import vifp_mscale
from FSIM import fsim
from SSIM import getssim



W=720
H=720

#dataset = 'DFC'
#dataset = 'pleiades'
dataset = 'pls'
method = 'RCAN'
scale = 4

save_dir = './test_log/'
img_path1 = '../test/testx{}/{}_{}_BIX{}_G10R20P64S1_epoch19_new/{}/'.format(scale, method, dataset, scale, method)
#img_path1 = '../test/testx{}/{}_per/{}/'.format(scale, method, method)
print(img_path1)
print(img_path1.split('/')[-3])
save_name = img_path1.split('/')[-3]


#img_path = '../../test/{}_test/'.format(dataset)
img_path = '..//..//data//test//{}_test//'.format(dataset)
if dataset == 'DFC':
    file0 = sorted(glob.glob('{}/*.tif'.format(img_path)))
elif dataset == 'pls':
    file0 = sorted(glob.glob('{}/*'.format(img_path)))
    print(file0)
file1 = sorted(glob.glob('{}/*'.format(img_path1)))
print(file1)


psnr_acc = np.zeros([len(file0),1])
ssim_acc = np.zeros([len(file0),1])
msssim_acc = np.zeros([len(file0),1])
AG_acc = np.zeros([len(file0),1])
NIQE_acc = np.zeros([len(file0),1])
VIFP_acc = np.zeros([len(file0),1])
FSIM_acc = np.zeros([len(file0),1])

def psnr(img1, img2):
    mse = np.mean((img1-img2)**2)
    if mse ==0:
        return 100
    pixel_max = 255.0
    return 20*math.log10(pixel_max/math.sqrt(mse))

#for f in file:
for i in range (len(file0)):
    print ('i = ', i)
    f = file0[i]
    print(f)
    print(file1[i])
    # pic_path = os.path.join(img_path, f)
    # pic_path1 = os.path.join(img_path1, file1[i])
    
    file_name = f
    img = cv2.imread(file0[i])
    img1 = cv2.imread(file1[i])
    
    AG_PECNN = avegrad(img1)
    # AG_bic = avegrad(img2)

    im_PECNN_Gray = np.array(Image.fromarray(img1, 'RGB').convert('LA'))[:, :, 0]
    # im_bic_Gray = np.array(Image.fromarray(img2, 'RGB').convert('LA'))[:, :, 0]
    input_Gray = np.array(Image.fromarray(img, 'RGB').convert('LA'))[:, :, 0]

    NIQE_PECNN = niqe(im_PECNN_Gray)
    # NIQE_bic = niqe(im_bic_Gray)

    VIFP_PECNN = vifp_mscale(img,img1)
    # VIFP_bic = vifp_mscale(img,img2)

    scales_PECNN_, FSIM_PECNN_, maps_PECNN_ = fsim(input_Gray,im_PECNN_Gray)
    # scales_bic_, FSIM_bic_, maps_bic_ = fsim(input_Gray,im_bic_Gray)

    ssim_PECNN1, ms_ssim_PECNN1 = getssim(input_Gray,im_PECNN_Gray)
    # ssim_bic1, ms_ssim_bic1 = getssim(input_Gray,im_bic_Gray)


    psnr_PECNN = psnr(img, img1)
    # psnr_bic = psnr(img, img2)

    # input_ = tf.convert_to_tensor(input_[0], dtype = tf.float32)
    # fake = tf.convert_to_tensor(fake[0], dtype = tf.float32)
    # bic_ref = tf.convert_to_tensor(bic_ref[0][0],dtype = tf.float32)
    # ssim_PECNN = sess.run( tf.image.ssim(input_,fake,max_val=1.0))
    # ssim_bic = sess.run( tf.image.ssim(input_,bic_ref,max_val=1.0))
    
    psnr_acc[i,0] = psnr_PECNN
    # psnr_acc[i,1] = psnr_bic
    
    ssim_acc[i,0] = ssim_PECNN1
    # ssim_acc[i,1] = ssim_bic1

    msssim_acc[i,0] = ms_ssim_PECNN1
    # msssim_acc[i,1] = ms_ssim_bic1
    
    AG_acc[i,0] = AG_PECNN
    # AG_acc[i,1] = AG_bic
    
    NIQE_acc[i,0] = NIQE_PECNN
    # NIQE_acc[i,1] = NIQE_bic
    
    VIFP_acc[i,0] = VIFP_PECNN
    # VIFP_acc[i,1] = VIFP_bic

    FSIM_PECNN = np.mean(FSIM_PECNN_)
    # FSIM_bic = np.mean(FSIM_bic_)
    FSIM_acc[i,0] = FSIM_PECNN
    # FSIM_acc[i,1] = FSIM_bic
    
                
    with open(os.path.join(save_dir, '{}9.txt'.format(save_name)), 'a+') as f:
        f.write(
            'Img num {0:3d} - PSNR: {1:0.6f},SSIM: {2:0.6f},FSIM: {3:0.6f}, VIFP:{4:0.6f},AG:{5:0.6f},NIQE:{6:0.6f}, ms_ssim: {7:0.6f}\n '.format(
                i, psnr_PECNN, ssim_PECNN1, FSIM_PECNN, VIFP_PECNN, AG_PECNN, NIQE_PECNN,  ms_ssim_PECNN1))
psnr_avg = np.sum(psnr_acc,axis=0)/len(file0)
ssim_avg = np.sum(ssim_acc,axis=0)/len(file0)
ms_ssim_avg = np.sum(msssim_acc,axis=0)/len(file0)
AG_avg = np.sum(AG_acc,axis=0)/len(file0)
NIQE_avg = np.sum(NIQE_acc,axis=0)/len(file0)
VIFP_avg = np.sum(VIFP_acc,axis=0)/len(file0)
FSIM_avg = np.sum(FSIM_acc,axis=0)/len(file0)


with open(os.path.join(save_dir, '{}9.txt'.format(save_name)), 'a+') as f:
    f.write(
        'Average:- PSNR_avg: {0:0.6f},SSIM_avg: {1:0.6f}, FSIM_avg: {2:0.6f}, VIFP_avg:{3:0.6f},AG_avg:{4:0.6f},NIQE_avg:{5:0.6f},ms_ssim_avg: {6:0.6f}\n'.format(
            psnr_avg[0], ssim_avg[0], FSIM_avg[0],  VIFP_avg[0], AG_avg[0], NIQE_avg[0],  ms_ssim_avg[0]))
