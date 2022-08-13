import argparse
import os
import glob
import PIL.Image as pil_image
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
from model_HANRSv3_LAM import HANRSv5
from model_HANRSv3_noLAM import HANRSv4
from model_HANRSv3_Non import HANRSv6
from model_HSCNRS import HSCNRS
from model_HSCANRS import HSCANRS
from model_HANRS_Hetconv import HANRS_Hetconv
from model_NLSN import NLSN

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
from models.DDBPN import DDBPN

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
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
    elif opt.arch == 'HANRSv4':
        print('HANRSv4')
        model  = HANRSv4(opt).to(device)
    elif opt.arch == 'HANRSv5':
        print('HANRSv5')
        model  = HANRSv5(opt).to(device)
    elif opt.arch == 'HANRSv6':
        print('HANRSv6')
        model  = HANRSv6(opt).to(device) 
    elif opt.arch == 'HSCNRS':
        print('HSCNRS')
        model  = HSCNRS(opt).to(device)
    elif opt.arch == 'HANRS_Hetconv':
        print('HANRS_Hetconv')
        model = HANRS_Hetconv(opt).to(device)
    elif opt.arch == 'HSCANRS':
        print('HSCANRS')
        model  = HSCANRS(opt).to(device)
    elif opt.arch == 'NLSN':
        print('NLSN')
        model  = NLSN(opt).to(device)
    elif opt.arch == 'EEGAN':
        print('EEGAN')
        model  = EEGAN(opt.scale).to(device)
    elif opt.arch == 'MHAN':
        print('MHAN')
        model = MHAN(3, 64, opt.scale).to(device)
    elif opt.arch == 'DDBPN':
        print('DDBPN')
        model = DDBPN(opt).to(device)
    elif opt.arch == 'SAN':
        print('SAN')
        model = SAN(opt).to(device)
    elif opt.arch == 'VDSR':
        print('VDSR')
        model = VDSR().to(device)
    elif opt.arch == 'SRCNN':
        print('SRCNN')
        model = SRCNN().to(device)
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

        input = pil_image.open(imgs[i]).convert('RGB')

        lr = input.resize((input.width // opt.scale, input.height // opt.scale), pil_image.BICUBIC)
        

        bicubic = lr.resize((input.width, input.height), pil_image.BICUBIC)
        #bicubic.save(os.path.join(opt.outputs_dir, '{}_x{}_bicubic.png'.format(filename, opt.scale)))

        input = transforms.ToTensor()(lr).unsqueeze(0).to(device)
        bicubic = transforms.ToTensor()(bicubic).unsqueeze(0).to(device)

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

        # output = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
        # output = pil_image.fromarray(output, mode='RGB')
        output_dir = os.path.join(opt.outputs_dir, '{}'.format(opt.arch))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # output.save(os.path.join(output_dir, '{}_x{}_{}.png'.format(filename, opt.scale, opt.arch)))
        input = input.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
        input = pil_image.fromarray(input, mode='RGB')
        input.save(os.path.join(output_dir, '{}_x{}_{}.png'.format(filename, opt.scale, opt.arch)))
        print('image saved!')
