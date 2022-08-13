import argparse
import os
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import glob
from tqdm import tqdm
from model import RCAN
from model_HAN import HAN
from model_HANRS import HANRS
from model_HANRSv2 import HANRSv2
from model_HANRSv3 import HANRSv3
from model_HANRSv3_LAM import HANRSv5
from model_HANRSv3_noLAM import HANRSv4
from model_HANRSv3_Non import HANRSv6
from model_HPANRS import HPANRS
from model_HPANRSv3 import HPANRSv3
from model_HSCNRS import HSCNRS
from model_HSCANRS import HSCANRS
from model_HANRS_Hetconv import HANRS_Hetconv
from model_ABPN import ABPN_v5
from model_NLSN import NLSN

from models.SRCNN import Net as SRCNN
from models.VDSR import Net as VDSR
from models.LGCNet import Net as LGCNet
from models.EEGAN import Net as EEGAN
from models.model import Net as MHAN


from models.san import Net as SAN
from models.SRFBN import SRFBN


from models.cafrn_corona import CAFRN
from models.DDBPN_corona import DDBPN
from models.RDN_corona import RDN

from dataset_corona import Dataset
from dataset2 import Dataset as Dataset2
from utils import AverageMeter, TVLoss
# import lpips

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='RCAN')
    parser.add_argument('--dataset', type=str, default = 'pleiades', required=True)
    
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--mode', type=str, default = 'train')
    parser.add_argument('--mode_val', type=str, default = 'val')
    parser.add_argument('--scale', type=int, required=True)
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--num_rg', type=int, default=10)
    parser.add_argument('--num_rcab', type=int, default=20)
    parser.add_argument('--reduction', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--eval_epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--use_fast_loader', action='store_true')
    parser.add_argument('--use_submean', action='store_true')
    parser.add_argument('--random_downsampling', action='store_true')
    parser.add_argument("--RotateFlip", action="store_true",
                    help="use RotateFlip or not.")
    parser.add_argument('--res_scale', type=float, default=1, help='residual scaling')
    parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
    parser.add_argument('--n_colors', type=int, default=1,
                    help='number of color channels to use')
    parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
    parser.add_argument('--pre_trained', action='store_true', help='continue from')
    parser.add_argument('--contEpoch', type=int,
                    help='contiune from where we left', default=0)
    parser.add_argument('--TV_loss', action='store_true',
                    help='use TV_loss')
    parser.add_argument('--per_loss', action='store_true',
                    help='use perceptual_loss')
    parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
                  
    opt = parser.parse_args()
    
    print(opt)
    
    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    torch.manual_seed(opt.seed)
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
    elif opt.arch == 'HANRSv4':
        print('HANRSv4')
        model  = HANRSv4(opt).to(device)
    elif opt.arch == 'HANRSv5':
        print('HANRSv5')
        model  = HANRSv5(opt).to(device)
    elif opt.arch == 'HANRSv6':
        print('HANRSv6')
        model  = HANRSv6(opt).to(device)    
    elif opt.arch == 'HPANRS':
        print('HPANRS')
        model  = HPANRS(opt).to(device)
    elif opt.arch == 'HPANRSv3':
        print('HPANRSv3')
        model  = HPANRSv3(opt).to(device)
    elif opt.arch == 'HSCNRS':
        print('HSCNRS')
        model  = HSCNRS(opt).to(device)
    elif opt.arch == 'NLSN':
        print('NLSN')
        model  = NLSN(opt).to(device)
    elif opt.arch == 'HSCANRS':
        print('HSCANRS')
        model  = HSCANRS(opt).to(device)
    elif opt.arch == 'HANRS_Hetconv':
        print('HANRS_Hetconv')
        model = HANRS_Hetconv(opt).to(device)
    elif opt.arch == 'ABPN':
        print('ABPN')
        model  = ABPN_v5(input_dim=3, dim=32).to(device)
    elif opt.arch == 'EEGAN':
        print('EEGAN')
        model  = EEGAN(opt.scale).to(device)
    elif opt.arch == 'MHAN':
        print('MHAN')
        model = MHAN(3, 64, opt.scale).to(device)
    elif opt.arch == 'SRFBN':
        print('SRFBN')
        model = SRFBN(opt.scale).to(device)
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
        model = CAFRN(in_channels=1, out_channels=1, num_features=64, num_steps=7, reduction=16, upscale_factor=opt.scale, act_type = 'prelu', norm_type = None).to(device)
    #criterion = nn.L1Loss()
    criterion = nn.SmoothL1Loss().cuda()
    if opt.TV_loss:
        criterion_tv = TVLoss(TVLoss_weight= 2e-2).cuda()
    # elif opt.per_loss:
        # criterion_per = lpips.LPIPS(net = 'vgg').cuda()
    print(model)
    print('model', opt.arch)
    print('# parameters:', sum(param.numel() for param in model.parameters()))
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)


    if opt.arch == 'LGCNet':
        dataset=Dataset2(patch_size = opt.patch_size, scale = opt.scale,mode = 'train', dataset = opt.dataset, use_fast_loader = opt.use_fast_loader)
    elif opt.arch == 'SRCNN':
        dataset=Dataset2(patch_size = opt.patch_size, scale = opt.scale,mode = 'train', dataset = opt.dataset, use_fast_loader = opt.use_fast_loader)
  
    elif opt.arch == 'VDSR':
        dataset=Dataset2(patch_size = opt.patch_size, scale = opt.scale,mode = 'train', dataset = opt.dataset, use_fast_loader = opt.use_fast_loader)
  
    else:
        dataset = Dataset(patch_size = opt.patch_size, scale = opt.scale,mode = 'train', dataset = opt.dataset, target_dir="../data/DFC/corona/",   RotateFlip = opt.RotateFlip)
    # print(len(dataset))
    dataloader = DataLoader(dataset=dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.threads,
                            pin_memory=True,
                            drop_last=True)
    if opt.arch == 'LGCNet':
        val_dataset = Dataset2( patch_size = opt.patch_size, scale = opt.scale, mode = 'test', dataset = opt.dataset, use_fast_loader = opt.use_fast_loader)
    elif opt.arch == 'SRCNN':
        val_dataset = Dataset2( patch_size = opt.patch_size, scale = opt.scale, mode = 'test', dataset = opt.dataset, use_fast_loader = opt.use_fast_loader)
    elif opt.arch == 'VDSR':
        val_dataset = Dataset2( patch_size = opt.patch_size, scale = opt.scale, mode = 'test', dataset = opt.dataset, use_fast_loader = opt.use_fast_loader)
    else:
        val_dataset = Dataset( patch_size = opt.patch_size, scale = opt.scale,mode = 'test', dataset = opt.dataset, target_dir="../data/DFC/corona/",   RotateFlip = opt.RotateFlip)
        
    # print(len(val_dataset))
    val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.threads,
                            pin_memory=True,
                            drop_last=True                           
                          )
    pathlist = sorted(glob.glob(opt.outputs_dir + '*.pth'))
    print(pathlist)
                          
                          
    #if opt.pre_trained:
    if not pathlist: 
        step = 0
        step_val = 0
        contEpoch = 0
    else:
        contEpoch = sorted([int(path.split('.')[-2].split('_')[-1]) for path in pathlist])[-1]+1
        print('contEpoch', contEpoch)  
        model.load_state_dict(torch.load(os.path.join(opt.outputs_dir, '{}_epoch_{}.pth'.format(opt.arch, contEpoch-1))))
        step = int(len(dataset)/opt.batch_size * contEpoch)
        print(step)
        step_val = int(len(val_dataset)/opt.batch_size * contEpoch)
        print(step_val)
    
        
    for epoch in range(contEpoch, opt.num_epochs):
    
        torch.cuda.empty_cache()
        print('Evaluating...')
            
        epoch_val_losses = AverageMeter()
        model.eval()
        with tqdm(total=(len(val_dataset) - len(val_dataset) % opt.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, opt.num_epochs))    
            for data in val_dataloader:
                val_inputs, val_labels = data
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                
                if opt.arch == 'EEGAN':
                    val_preds, out2 = model(val_inputs)
                    val_loss1 = criterion(val_preds, val_labels)
                    val_loss2 = criterion(out2, val_labels)
                    val_loss = 10*val_loss1 + val_loss2
                else:
                    val_preds = model(val_inputs)
                    if opt.TV_loss:
                        val_loss = criterion(val_preds, val_labels) + criterion_tv(val_preds)
                    elif opt.per_loss:
                        #val_loss1 = criterion(val_preds, val_labels)
                        #val_loss2 = 0.001 * criterion_per.forward(val_preds, val_labels)[0]
                        val_loss = criterion(val_preds, val_labels)+ 0.001 * criterion_per(val_preds, val_labels)[0]
                    else:
                        val_loss = criterion(val_preds, val_labels)
                #print(val_loss1, val_loss2)
                epoch_val_losses.update(val_loss.item(), len(val_inputs))

                
                
                
                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_val_losses.avg))
                _tqdm.update(len(val_inputs))
                step_val += 1
                if step_val % 10 == 0:
                    with open(os.path.join(opt.outputs_dir, 'val_log.txt'), 'a+') as f:
                        f.write('epoch {} - Iter {} - loss:{} \n'.format(epoch, step_val, epoch_val_losses.avg))
                
                del val_inputs
                del val_labels
                del val_preds
                del val_loss
                
        epoch_losses = AverageMeter()
        
        with tqdm(total=(len(dataset) - len(dataset) % opt.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, opt.num_epochs))
            torch.cuda.empty_cache()
            for data in dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                if opt.arch == 'EEGAN':
                    preds, out2 = model(inputs)
                    loss1 = criterion(preds, labels)
                    loss2 = criterion(out2, labels)
                    loss = 10*loss1 + loss2
                else:
                    preds = model(inputs)
                    if opt.TV_loss:
                        loss = criterion(preds, labels) + criterion_tv(preds)
                    if opt.per_loss:
                        loss = criterion(preds, labels) + 0.001 * criterion_per(preds, labels)[0]
                    else:
                        loss = criterion(preds, labels)
                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                
                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                _tqdm.update(len(inputs))
                step += 1
                if step % 100 == 0:
                    with open(os.path.join(opt.outputs_dir, 'train_log.txt'), 'a+') as f:
                        f.write('Iter {} - loss:{} \n'.format(step, epoch_losses.avg))
                
                del preds
                del loss
                del inputs
                del labels
        torch.save(model.state_dict(), os.path.join(opt.outputs_dir, '{}_epoch_{}.pth'.format(opt.arch, epoch)))
        
        
        
            