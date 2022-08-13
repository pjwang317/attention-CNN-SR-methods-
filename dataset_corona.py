import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import glob
import numpy as np
# import PIL.Image as pil_image
from PIL import Image
from PIL import ImageFile
import tifffile

import torchvision.transforms as transforms
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.enable_eager_execution(config=config)
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Dataset(object):
    def __init__(self,  patch_size, scale, target_dir, dataset,mode, RotateFlip = False ):
        
        
        self.patch_size = patch_size
        self.scale = scale
        self.dataset = dataset
        self.mode = mode
        self.RotateFlip = RotateFlip
        #self.input_filenames = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if check_image_file(x)]
        
        if self.dataset == 'Corona':
            if self.mode == 'train':
                self.target_filenames = glob.glob(target_dir + '/HR_patches/train/*.png')
                self.input_filenames = glob.glob(target_dir + '/LR_patches/train/*.png')
            elif self.mode == 'test':
                self.target_filenames = glob.glob(target_dir + '/HR_patches/test/*.png')
                self.input_filenames = glob.glob(target_dir + '/LR_patches/test/*.png')  
            
        print(len(self.target_filenames), self.RotateFlip)
        
        if self.RotateFlip:
            self.transforms = transforms.Compose([
                              transforms.RandomRotation(90),
                              transforms.RandomRotation(180),
                              transforms.RandomRotation(270),
                              transforms.RandomHorizontalFlip(0.5),
                              transforms.ToTensor()
                              ])
        else:
            self.transforms = transforms.ToTensor()
        
        
    def __getitem__(self, idx):
               
        
        if self.dataset == 'Corona':
            target = Image.open(self.target_filenames[idx])
            #print(target)
            target = target.resize((self.patch_size*self.scale, self.patch_size*self.scale), resample=Image.BICUBIC)
            #print(target.size())
            input = Image.open(self.input_filenames[idx])
            #print(input.size())
        else:
            target = Image.open(self.target_filenames[idx])
            input = target.resize((self.patch_size, self.patch_size), resample=Image.BICUBIC)
        
        input = self.transforms(input)
        target = self.transforms(target)
        
        return input, target
        
        
        
        

        return input, target

    def __len__(self):
        
        return len(self.target_filenames)
        