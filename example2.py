import argparse
import os
import glob
import numpy as np
import PIL.Image as pil_image
import matplotlib.pyplot as plt
import tifffile
import cv2

from skimage.transform import resize
from scipy.io import savemat
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from model import RCAN
from model_HAN import HAN
from model_HANRS import HANRS
from model_HANRSv2 import HANRSv2
from model_HANRSv3 import HANRSv3
from model_HPANRS import HPANRS
from model_HPANRSv3 import HPANRSv3
from model_HSCNRS import HSCNRS
from model_HSCANRS import HSCANRS
from model_HANRS_Hetconv import HANRS_Hetconv

from models.SRCNN import Net as SRCNN
from models.VDSR import Net as VDSR
from models.LGCNet import Net as LGCNet
from models.EEGAN import Net as EEGAN
from models.model import Net as MHAN
from models.DDBPN import DDBPN
#from models.rcan import RCAN
from models.RDN import RDN
from models.san import Net as SAN
from models.cafrn import CAFRN


cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def tifread(path):
    img = pil_image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))
    return np.array(images)
def scale_range(input, min, max):
    input += -(np.min(input))
    input /= (1e-9 + np.max(input) / (max - min + 1e-9))
    input += min
    return input
    
    
def save_test_image(im, img_name,save_path):
    if len(im.shape) == 2:
        im = scale_range(im, 0, 255).astype(np.uint8)
        plt.figure(figsize=(16, 16), dpi= 80, facecolor='w', edgecolor='k')
        plt.imshow(im,cmap='gray')
        plt.savefig(os.path.join(save_path, "sr_{}.png".format(img_name)))
        # plt.show()

    elif len(im.shape) == 3:
        im = np.array([scale_range(i, 0, 255) for i in im.transpose((2,0,1))]).transpose(1,2,0)[...,:3].astype(np.uint8)
        #plt.figure(figsize=(16, 16), dpi= 80, facecolor='w', edgecolor='k')
        # plt.imshow(im)
        #plt.savefig(os.path.join(save_path, "sr_{}.png".format(img_name)))
        cv2.imwrite(os.path.join(save_path, "cvsr_{}.tif".format(img_name)), im)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='RCAN')
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--scale', type=int, required=True)
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--num_rg', type=int, default=10)
    parser.add_argument('--num_rcab', type=int, default=20)
    parser.add_argument('--reduction', type=int, default=16)
    parser.add_argument('--res_scale', type=float, default=1, help='residual scaling')
    parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
    parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
    parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    if opt.arch == 'RCAN':
        print('RCAN')
        model = RCAN(opt).to(device)
    elif opt.arch == 'HAN':
        print('HAN')
        model = HAN(opt).to(device)
    elif opt.arch == 'HANRS':
        print('HANRS')
        model  = HANRS(opt).to(device)
    elif opt.arch == 'HANRSv2':
        print('HANRSv2')
        model  = HANRSv2(opt).to(device)
    elif opt.arch == 'HANRSv3':
        print('HANRSv3')
        model  = HANRSv3(opt).to(device)
    elif opt.arch == 'HPANRS':
        print('HPANRS')
        model  = HPANRS(opt).to(device)
    elif opt.arch == 'HPANRSv3':
        print('HPANRSv3')
        model  = HPANRSv3(opt).to(device)
    elif opt.arch == 'HSCNRS':
        print('HSCNRS')
        model  = HSCNRS(opt).to(device)
    elif opt.arch == 'HANRS_Hetconv':
        print('HANRS_Hetconv')
        model = HANRS_Hetconv(opt).to(device)
    elif opt.arch == 'HSCANRS':
        print('HSCANRS')
        model  = HSCANRS(opt).to(device)
    elif opt.arch == 'EEGAN':
        print('EEGAN')
        model  = EEGAN(opt.scale).to(device)
    elif opt.arch == 'MHAN':
        print('MHAN')
        model = MHAN(3, 64, opt.scale).to(device)
    elif opt.arch == 'SAN':
        print('SAN')
        model = SAN(opt).to(device)
    elif opt.arch == 'VDSR':
        print('VDSR')
        model = VDSR().to(device)
    elif opt.arch == 'SRCNN':
        print('SRCNN')
        model = SRCNN().to(device)
    elif opt.arch == 'DDBPN':
        print('DDBPN')
        model = DDBPN(opt).to(device)
    elif opt.arch == 'RDN':
        print('RDN')
        model = RDN( opt.scale).to(device)
    elif opt.arch == 'LGCNet':
        print('LGCNet')
        model = LGCNet().to(device)
    elif opt.arch == 'CAFRN':
        print('CAFRN')
        model = CAFRN(in_channels=3, out_channels=3, num_features=64, num_steps=7, reduction=16, upscale_factor=opt.scale, act_type = 'prelu', norm_type = None).to(device)
        
        
        
        
    state_dict = model.state_dict()
    for n, p in torch.load(opt.weights_path, map_location=lambda storage, loc: storage).items():
    # for n, p in torch.load(opt.weights_path).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model = model.to(device)
    print('# parameters:', sum(param.numel() for param in model.parameters()))
    
    model.eval()
    
    imgs = glob.glob('{}/*'.format(opt.image_path))
    print(imgs)
    
    for i in range(len(imgs)):
        filename = os.path.basename(imgs[i]).split('.')[0]
        print(filename)
        
        
        ms = np.load(imgs[i])[...,:3].astype(np.float32)
        ms_norm = np.array([scale_range(i, 0, 1) for i in ms.transpose((2,0,1))])
        ms_norm = np.clip(ms_norm,0.0,1.0)
        save_test_image(ms_norm.transpose(1,2,0), i, opt.outputs_dir)
        print('ms_norm', ms_norm.shape)
            # cv2.imwrite('./ms/{:03d}.tif'.format(i), ms_norm.transpose((1,2,0)).astype(np.uint8))
        input = pil_image.fromarray(ms_norm.transpose((1,2,0)).astype(np.uint8))
            
            # input = pil_image.fromarray(scale_range(tifffile.imread(imgs[i])[:,:,[1,2,3]].astype(np.float32), 0, 255).astype(np.uint8) )
        
        bicubic = input.resize((input.width * opt.scale, input.height * opt.scale), pil_image.BICUBIC)
        
        bicubic.save(os.path.join(opt.outputs_dir, '{}_x{}_bicubic.png'.format(filename, opt.scale)))
        input.save(os.path.join(opt.outputs_dir, '{}_x{}_input.png'.format(filename, opt.scale)))
        # input = transforms.ToTensor()(lr).unsqueeze(0).to(device)
        input = transforms.ToTensor()(input).unsqueeze(0).to(device)
        bicubic = transforms.ToTensor()(bicubic).unsqueeze(0).to(device)
        
        #input = transforms.ToTensor()(input).unsqueeze(0).to(device)
        with torch.no_grad():
            if opt.arch == 'EEGAN':
                pred, _ =model(input)
            elif opt.arch == 'SRCNN':
                pred = model(bicubic)
            elif opt.arch == 'VDSR':
                pred = model(bicubic)
            elif opt.arch == 'LGCNet':
                pred = model(bicubic)
            else:
                pred = model(input)
        print(pred.size())
        # print(pred)
        # output = pred.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # output = pred.squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
        
        # output = np.array([scale_range(i, -1,1) for i in output.transpose((2,0,1))])
        # output = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
        output = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # print(output)
        mdic = {'output': output}
        savemat(os.path.join(opt.outputs_dir,'out_{:}.mat'.format(filename)),mdic)
        # output = pil_image.fromarray(output, mode='RGB')
        # print('output max---',np.max(output))
        # print(output.shape)
        output_dir = os.path.join(opt.outputs_dir, '{}'.format(opt.arch))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # output.save(os.path.join(output_dir, '{}_x{}_{}.png'.format(filename, opt.scale, opt.arch)))
        # save_test_image(bicubic.transpose(1,2,0),  'bicubic_x{}_{:04d}'.format(opt.scale,i), opt.outputs_dir)
        # cv2.imwrite(os.path.join(opt.outputs_dir, 'output_x{}_{:04d}.tif'.format(opt.scale,i)), output)
        # save_test_image(input.transpose(1,2,0),  'input_x{}_{:04d}'.format(opt.scale,i), opt.outputs_dir)
        save_test_image(output, i, output_dir)
        print('image saved!')
