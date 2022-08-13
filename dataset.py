import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import glob
import numpy as np
import PIL.Image as pil_image
from PIL import ImageFile
import tifffile
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.enable_eager_execution(config=config)
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Dataset(object):
    def __init__(self,  patch_size, scale, mode ,dataset ,random_downsampling = False, use_fast_loader=False):
        self.dataset = dataset
        if self.dataset == 'WV3':
            self.train_samples = 44080
            self.test_samples = 448
        elif self.dataset == 'pleiades':
            #self.train_samples = 8376
            #self.test_samples = 84
            self.train_samples = 7648 
            self.test_samples = 85
        elif self.dataset == 'Mdata':
            #self.train_samples = 8376
            #self.test_samples = 84
            self.train_samples = 62647
            self.test_samples = 3300
        # self.train_samples = 60
        # self.test_samples = 20
        
        self.random_downsampling = random_downsampling
        self.patch_size = patch_size
        self.scale = scale
        self.use_fast_loader = use_fast_loader
        self.mode = mode
        # print('111',self.mode)
        # if self.mode == 'train':
            # self.data_name = '../PanNet_ICCV17/training_data/traindata.npz' # training data
        # elif self.mode  == 'val':
            # self.data_name  = '../PanNet_ICCV17/training_data/validdata.npz' # validation data
        # elif self.mode == 'test':
            # self.data_name = '../PanNet_ICCV17/training_data/testdata.npz'
    def __getitem__(self, idx):
        
        if self.mode == "train":
            
            filename = idx
        elif self.mode == "test":
            # print(self.mode)
            filename = str(int(idx) + self.train_samples)
       
        # print('2222')
        # hr = pil_image.open(self.image_files[idx]).convert('RGB')
        # hr = tifffile.imread(r'D:\Dataset\Pleiades\train_croped\{}.tif'.format(filename))
        if self.dataset == 'pleiades':
            hr = pil_image.open(r'../data/DFC/pls_train2/{:06d}.tiff'.format(int(filename)+1)).convert('RGB')
            #hr = pil_image.open(r'../data/train_pls/{}.tif'.format(filename)).convert('RGB')
        elif self.dataset == 'WV3':
            #print('filename:', filename)
            #if filename == 44060 or filename == 14421:
            #   filename = str(int(filename) -100)
            #    print('filenamenew:',filename)
            #print(filename)
            hr = pil_image.open(r'../data/train/{}.tif'.format(filename)).convert('RGB')
        elif self.dataset == 'Mdata':
            
            hr = pil_image.open(r'../data/DFC/Mdata/{:06d}.tif'.format(int(filename))).convert('RGB')
        # randomly crop patch from training set
        crop_x = random.randint(0, hr.width - self.patch_size * self.scale)
        crop_y = random.randint(0, hr.height - self.patch_size * self.scale)
        hr = hr.crop((crop_x, crop_y, crop_x + self.patch_size * self.scale, crop_y + self.patch_size * self.scale))

        # degrade lr with Bicubic
        if self.random_downsampling:
            reduce_size = np.random.randint(20, 80)
            lr = hr.resize((reduce_size, reduce_size), resample=pil_image.BICUBIC)
            lr = lr.resize((self.patch_size, self.patch_size), resample=pil_image.BICUBIC)
        else:
            lr = hr.resize((self.patch_size, self.patch_size), resample=pil_image.BICUBIC)

        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)

        hr = np.transpose(hr, axes=[2, 0, 1])
        lr = np.transpose(lr, axes=[2, 0, 1])

        # normalization
        hr /= 255.0
        lr /= 255.0

        return lr, hr

    def __len__(self):
        if self.mode  == 'train':
            return self.train_samples
        elif self.mode == 'test':
            return self.test_samples
