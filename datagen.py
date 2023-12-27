# author: Niwhskal
# github : https://github.com/Niwhskal/SRNet

import os
from skimage import io
from skimage.transform import resize
import numpy as np
import random
import cfg
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2




class datagen_srnet(Dataset):
    def __init__(self, cfg, torp = 'train', transforms = None):
        
        if(torp == 'train'):
            self.data_dir = cfg.data_dir
            self.t_b_dir = cfg.t_b_dir
            self.batch_size = cfg.batch_size
            self.data_shape = cfg.data_shape
            self.name_list = os.listdir(os.path.join(self.data_dir, self.t_b_dir))
      
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        
        img_name = self.name_list[idx]
        
        i_t = cv2.imread(os.path.join(cfg.data_dir, cfg.i_t_dir, img_name))
        i_t = cv2.cvtColor(i_t, cv2.COLOR_BGR2RGB)
        i_t = cv2.resize(i_t, (128, 64))

        i_s = cv2.imread(os.path.join(cfg.data_dir, cfg.i_s_dir, img_name))
        i_s = cv2.cvtColor(i_s, cv2.COLOR_BGR2RGB)
        i_s = cv2.resize(i_s, (128, 64))

        t_sk = cv2.imread(os.path.join(cfg.data_dir, cfg.t_sk_dir, img_name))
        t_sk = cv2.cvtColor(t_sk, cv2.COLOR_BGR2GRAY)
        t_sk = cv2.resize(t_sk, (128, 64))

        t_t = cv2.imread(os.path.join(cfg.data_dir, cfg.t_t_dir, img_name))
        t_t = cv2.cvtColor(t_t, cv2.COLOR_BGR2RGB)
        t_t = cv2.resize(t_t, (128, 64))

        t_b = cv2.imread(os.path.join(cfg.data_dir, cfg.t_b_dir, img_name))
        t_b = cv2.cvtColor(t_b, cv2.COLOR_BGR2RGB)
        t_b = cv2.resize(t_b, (128, 64))

        t_f = cv2.imread(os.path.join(cfg.data_dir, cfg.t_f_dir, img_name))
        t_f = cv2.cvtColor(t_f, cv2.COLOR_BGR2RGB)
        t_f = cv2.resize(t_f, (128, 64))

        # mask_t = cv2.imread(os.path.join(cfg.data_dir, cfg.mask_t_dir, img_name))
        # mask_t = cv2.cvtColor(mask_t, cv2.COLOR_BGR2GRAY)
        # mask_t = cv2.resize(mask_t, (128, 64))
        
        return [i_t, i_s, t_sk, t_t, t_b, t_f]


class example_dataset(Dataset):
    
    def __init__(self, data_dir = cfg.example_data_dir, transform = None):
        
        self.files_s = glob.glob('/DATA/ocr_team_2/onkar/final_dataset/eng_hin/test/i_s/*.png')
        # self.files_s = glob.glob('icdar_crops/image_crops/*.png')
        self.files_t = glob.glob('/DATA/ocr_team_2/onkar/final_dataset/eng_hin/test/i_t/*.png')
        self.labels = glob.glob('/DATA/ocr_team_2/onkar/final_dataset/eng_hin/test/t_f/*.png')

        # shuffle.shuffle(self.files_t)
        # self.files_s.sort()
        # self.files_t.sort()
        
        self.transform = transform
        
    def __len__(self):
        # return 100
        return len(glob.glob('/DATA/ocr_team_2/onkar/final_dataset/eng_hin/test/i_s/*.png'))
    
    def __getitem__(self, idx):
        
        name = self.files_t[idx].split('/')[-1]
        i_s = io.imread('/DATA/ocr_team_2/onkar/final_dataset/eng_hin/test/i_s/{}.png'.format(idx))
        i_t = io.imread('/DATA/ocr_team_2/onkar/final_dataset/eng_hin/test/i_t/{}.png'.format(idx))
        # i_s = io.imread('icdar_crops/image_crops/1_1.png')
        to_scale = i_s.shape[:2]
        # print(name, i_s.shape)
        # if i_s.shape[1] <=128:
        #     s = (128, 64)

        # elif i_s.shape[1] <=256:
        #     s = (128, 64)
        
        # elif i_s.shape[1] <= 512:
        #     s = (256, 128)

        # elif i_s.shape[1] > 512:
        #     s = (256, 128)
        # else:
        #     s=() 
        # print("shape of S is: ", s)

        s = (128, 64)

      
        
        i_s = cv2.resize(i_s, s)[:,:,:3]
        h, w = i_t.shape[:2]
        scale_ratio = cfg.data_shape[0] / h
        to_h = cfg.data_shape[0]
        to_w = int(round(int(w * scale_ratio) / 8)) * 8
        to_scale = (to_scale[1], to_scale[0])
        
        i_t = cv2.resize(i_t, s)[:,:,:3]

        t_f = io.imread('/DATA/ocr_team_2/onkar/final_dataset/eng_hin/test/t_f/{}.png'.format(idx))
        t_f = cv2.resize(t_f, s)[:,:,:3]
        # i_s = cv2.resize(i_s, to_scale)[:,:,:3]
        # i_s = resize(i_s, to_scale, preserve_range=True)
        
        sample = (i_t, i_s, t_f, str(idx))  #1024 1024 ===> 320, 320 ==> detection ==> crops ==> 
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
        
class To_tensor(object):
    def __call__(self, sample):
        
        i_t, i_s, t_f, img_name = sample

        i_t = i_t.transpose((2, 0, 1)) /127.5 -1
        i_s = i_s.transpose((2, 0, 1)) /127.5 -1

        t_f = t_f /127.5 -1

        i_t = torch.from_numpy(i_t)
        i_s = torch.from_numpy(i_s)
        t_f = torch.from_numpy(t_f)

        return (i_t.float(), i_s.float(), t_f.float(), img_name)


class example_dataset_2(Dataset):
    
    def __init__(self, data_dir = cfg.example_data_dir, transform = None):
        
        self.files_s = glob.glob('/DATA/ocr_team_2/onkar/7th_feb_ppt/i_s/*.png')
        self.files_s.sort()
    
        # self.files_s = glob.glob('icdar_crops/image_crops/*.png')
        self.files_t = glob.glob('7th_feb_ppt/i_t_feb7_ppt/feb7_ppt_hin/*.png')
        self.files_t.sort()

        print(len(self.files_t), len(self.files_s))
        # random.shuffle(self.files_t)
        # self.labels = glob.glob('final_dataset/eng_ben/train/t_f/*.png')

        # self.files_s.sort()
        # self.files_t.sort()
        
        self.transform = transform
        
    def __len__(self):
        return len(self.files_s)
    
    def __getitem__(self, idx):
        
        name = self.files_t[idx]
        
        i_s = io.imread(self.files_s[idx])
        shape = i_s.shape[:2]
        # print(shape)
        i_t = io.imread(self.files_t[idx])
        # i_s = io.imread('icdar_crops/image_crops/1_1.png')
        name = self.files_s[idx].split('/')[-1]
        to_scale = i_s.shape[:2]
        # print(name, i_s.shape)
        # if i_s.shape[1] <=128:
        #     s = (128, 64)

        # elif i_s.shape[1] <=256:
        #     s = (128, 64)
        
        # elif i_s.shape[1] <= 512:
        #     s = (256, 128)

        # elif i_s.shape[1] > 512:
        #     s = (256, 128)
        # else:
        #     s=(256, 128) 
        # # print("shape of S is: ", s)

        s = (128, 64)

      
        
        i_s = cv2.resize(i_s, s)[:,:,:3]
        h, w = i_t.shape[:2]
        scale_ratio = cfg.data_shape[0] / h
        to_h = cfg.data_shape[0]
        to_w = int(round(int(w * scale_ratio) / 8)) * 8
        to_scale = (to_scale[1], to_scale[0])
        
        i_t = cv2.resize(i_t, s)[:,:,:3]

        
        # i_s = cv2.resize(i_s, to_scale)[:,:,:3]
        # i_s = resize(i_s, to_scale, preserve_range=True)
        
        sample = (i_t, i_s, name, shape)  #1024 1024 ===> 320, 320 ==> detection ==> crops ==> 
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
        
class To_tensor_2(object):
    def __call__(self, sample):
        
        i_t, i_s, img_name, shape = sample

        i_t = i_t.transpose((2, 0, 1)) /127.5 -1
        i_s = i_s.transpose((2, 0, 1)) /127.5 -1

        # t_f = t_f 

        i_t = torch.from_numpy(i_t)
        i_s = torch.from_numpy(i_s)
        # t_f = torch.from_numpy(t_f)

        return (i_t.float(), i_s.float(), img_name, shape)

class To_tensor_3(object):
    def __call__(self, sample):
        
        i_t, i_s = sample

        i_t = i_t.transpose((2, 0, 1)) /127.5 -1
        i_s = i_s.transpose((2, 0, 1)) /127.5 -1

        # t_f = t_f 

        i_t = torch.from_numpy(i_t)
        i_s = torch.from_numpy(i_s)
        # t_f = torch.from_numpy(t_f)

        return (i_t.float(), i_s.float())


class To_tensor_4(object):
    def __call__(self, sample):
        
        i_t, i_s, b_g = sample

        i_t = i_t.transpose((2, 0, 1)) /127.5 -1
        i_s = i_s.transpose((2, 0, 1)) /127.5 -1
        b_g = b_g.transpose((2, 0, 1)) /127.5 -1

        # t_f = t_f 

        i_t = torch.from_numpy(i_t)
        i_s = torch.from_numpy(i_s)
        b_g = torch.from_numpy(b_g)

        return (i_t.float(), i_s.float(), b_g.float())