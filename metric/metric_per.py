import numpy as np
import scipy.misc
import cv2
import os
import glob

import matplotlib.pyplot as plt
import sys
import math
from PIL import Image

import lpips
import csv
from pathlib import Path

from AG import avegrad
from NIQE import niqe
from VIFP import vifp_mscale
from FSIM import fsim
from SSIM import getssim

W=720
H=720

dataset = 'DFC'
#dataset = 'pleiades'
#dataset = 'pls'


# method = 'RCAN'
# method = 'HAN'
# method = 'HANRS'
# method = 'HANRS_Hetconv'
# method = 'HANRSv3'
# method = 'HPANRS'
# method = 'HPANRSv3'
# method = 'HSCNRS'
# method = 'bic'
# method = 'LGCNet'
# method = 'SRCNN'
# method = 'VDSR'
# method = 'SAN'
# method = 'MHAN'
# method = 'RDN'
method = 'CAFRN'
scale = 2
G = 10
R = 20
P = 64

#save_dir = './test_log_lpips/'
# img_path1 = '../test/pls_test_new/retrain/x{}/{}_{}_BIX{}_G10R20P32S1_epoch19_new/{}/'.format(scale, method, dataset, scale, method)
save_dir = '..\\metric\\test_per\\X{}\\'.format(scale)
# img_path1 = '../test/pls_test_new/retrain/x{}/{}_{}_BIX{}_G{}R{}P{}S1_epoch19_new/{}/'.format(scale,method, dataset,scale, G, R, P, method)
# img_path1 = '../test/pls_test_new/retrain/x{}/{}_{}_BIX{}_epoch19_new/{}/'.format(scale,method, dataset,scale, method)
# img_path1 = '../test/pls_test_new/retrain/x{}/{}_{}_BIX{}_G{}R{}P{}S.2_epoch19_new/{}/'.format(scale,method, dataset,scale, G, R, P, method)
# img_path1 = '../test/pls_test_new/retrain/x{}/{}_{}_BIX{}_G{}R{}P{}S.2_epoch19_per/{}/'.format(scale,method, dataset,scale, G, R, P, method)
img_path1 = '../test/pls_test_new/retrain/x{}/{}_{}_BIX{}_epoch19_new/{}/'.format(scale,method, dataset,scale, method)
# img_path1 = '../test/pls_test_new/retrain/x{}/{}_new/{}/'.format(scale,method, method)

# img_path1 = '../test/testx{}/WV3/{}_{}_BIX{}_G{}R{}P{}S1_epoch19/{}/'.format(scale,method, dataset,scale, G, R, P, method)
# img_path1 = '../test/testx{}/WV3/{}_{}_BIX{}_epoch99/{}/'.format(scale,method, dataset,scale,method)
# img_path1 = '../test/testx{}/WV3/{}/{}/'.format(scale,method,method)
# img_path1 = '../test/testx{}/WV3/{}_{}_BIX{}_G{}R{}P{}S.2_epoch19/{}/'.format(scale,method, dataset,scale, G, R, P, method)
# img_path1 = '../test/pls_test_new/retrain/x{}/{}_{}_BIX{}_epoch19_new/{}/'.format(scale,method, dataset,scale, method)

# img_path1 = '..\\test\\dir1\\'

print(img_path1)

print(img_path1.split('/')[-3])
save_name = img_path1.split('/')[-3]

if not os.path.exists(save_dir):
        os.mkdir(save_dir)
if dataset == 'DFC':
    img_path = '..\\test\\DFC_test\\'
    # img_path = '..\\test\\dir0\\'
elif dataset == 'pls':
    img_path = '..\\test\\pls_test_new\\png\\'
file0 = glob.glob('{}/*'.format(img_path))
file1 = glob.glob('{}/*'.format(img_path1))
# img_path = '..//..//data//test//{}_test//'.format(dataset)
# if dataset == 'DFC':
    # file0 = glob.glob('{}/*.tif'.format(img_path))
# elif dataset == 'pls':
    # file0 = glob.glob('{}/*'.format(img_path))
# file1 = glob.glob('{}/*.png'.format(img_path1))
# print(file1)

lpips_acc = np.zeros([len(file0),1])
psnr_acc = np.zeros([len(file0),1])
ssim_acc = np.zeros([len(file0),1])
msssim_acc = np.zeros([len(file0),1])
AG_acc = np.zeros([len(file0),1])
NIQE_acc = np.zeros([len(file0),1])
VIFP_acc = np.zeros([len(file0),1])
FSIM_acc = np.zeros([len(file0),1])

loss_fn = lpips.LPIPS(net = 'vgg')
#loss_fn.cuda()

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
    img0 = lpips.im2tensor(cv2.imread(file0[i]))
    img1 = lpips.im2tensor(cv2.imread(file1[i]))
    
    #img0.cuda()
    #img1.cuda()
    d= loss_fn.forward(img0, img1)
    print(d)
    lpips_acc[i,0] = d.item()
    


    img0 = cv2.imread(file0[i])
    img1 = cv2.imread(file1[i])
    
    AG_PECNN = avegrad(img1)
    im_PECNN_Gray = np.array(Image.fromarray(img1, 'RGB').convert('LA'))[:, :, 0]
    input_Gray = np.array(Image.fromarray(img0, 'RGB').convert('LA'))[:, :, 0]
    NIQE_PECNN = niqe(im_PECNN_Gray)
    VIFP_PECNN = vifp_mscale(img0,img1)
    scales_PECNN_, FSIM_PECNN_, maps_PECNN_ = fsim(input_Gray,im_PECNN_Gray)
    ssim_PECNN1, ms_ssim_PECNN1 = getssim(input_Gray,im_PECNN_Gray)
    psnr_PECNN = psnr(img0, img1)
   
    psnr_acc[i,0] = psnr_PECNN  
    ssim_acc[i,0] = ssim_PECNN1
    msssim_acc[i,0] = ms_ssim_PECNN1
    AG_acc[i,0] = AG_PECNN
    NIQE_acc[i,0] = NIQE_PECNN
    VIFP_acc[i,0] = VIFP_PECNN
    FSIM_PECNN = np.mean(FSIM_PECNN_)
    FSIM_acc[i,0] = FSIM_PECNN



    
    if Path(f"{save_dir}{save_name}_lpips.csv").is_file():
        print ("File exist")
    else:
        print ("File not exist")
        with open(f"{save_dir}{save_name}_lpips.csv", 'a+',newline = "") as f:
            writer = csv.writer(f)
            writer.writerow(["Img num", "PSNR", "SSIM","VIFP","FSIM","AG","NIQE","MS_SSIM","LPIPS"])
                
    # with open(os.path.join(save_dir, '{}_lpips.txt'.format(save_name)), 'a+') as f:
        # f.write(
                  # 'Img num {0:3d} - LPIPS: {1:0.6f}\n '.format(i, d.item()))
    with open(f"{save_dir}{save_name}_lpips.csv", 'a+',newline = "") as f:
        writer = csv.writer(f)
        writer.writerow([f"{i}",f"{psnr_PECNN}", f"{ssim_PECNN1}", f"{VIFP_PECNN}", f"{FSIM_PECNN}", f"{AG_PECNN}", f"{NIQE_PECNN}",  f"{ms_ssim_PECNN1}", f"{d.item()}"])
lpips_avg = np.sum(lpips_acc,axis=0)/len(file0)
psnr_avg = np.sum(psnr_acc,axis=0)/len(file0)
ssim_avg = np.sum(ssim_acc,axis=0)/len(file0)
ms_ssim_avg = np.sum(msssim_acc,axis=0)/len(file0)
AG_avg = np.sum(AG_acc,axis=0)/len(file0)
NIQE_avg = np.sum(NIQE_acc,axis=0)/len(file0)
VIFP_avg = np.sum(VIFP_acc,axis=0)/len(file0)
FSIM_avg = np.sum(FSIM_acc,axis=0)/len(file0)

if Path(f"{save_dir}metric_lpips.csv").is_file():
    print ("File exist")
else:
    print ("File not exist")
    with open(f"{save_dir}metric_lpips_avg.csv", 'a+',newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "PSNR", "SSIM","VIFP","FSIM","AG","NIQE","MS_SSIM","LPIPS"])

# with open(os.path.join(save_dir, 'metric_lpips_avg.txt'.format(save_name)), 'a+') as f:
    # f.write(
        # 'Average:- LPIPS_avg: {0:0.6f}\n'.format(
            # lpips_avg[0]))
with open(f"{save_dir}metric_lpips_avg.csv", 'a+',newline = "") as f:
    writer = csv.writer(f)
    writer.writerow([f"{save_name}",f"{psnr_avg[0]}", f"{ssim_avg[0]}", f"{VIFP_avg[0]}", f"{FSIM_avg[0]}", f"{AG_avg[0]}", f"{NIQE_avg[0]}", f"{ms_ssim_avg[0]}", f"{lpips_avg[0]}"])
