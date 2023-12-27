# Predict script 
# author: Niwhskal

import os
import argparse
import cfg
import torch
from tqdm import tqdm
import numpy as np
from Diffusion.model import Generator, Discriminator, Vgg19
from utils import *
from eval_utils import *
from datagen import *
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import cv2
import math
# from skimage.measure import compare_ssim

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'



device = 'cuda'

def main():
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('--input_dir', help = 'Directory containing xxx_i_s and xxx_i_t with same prefix',
                        default = cfg.example_data_dir)
    # parser.add_argument('--checkpoint', help = 'ckpt', default = cfg.ckpt_path)
    args = parser.parse_args()

    # assert args.input_dir is not None
    # assert args.save_dir is not None
    # assert args.checkpoint is not None

    print_log('model compiling start.', content_color = PrintColor['yellow'])

    G = Generator(in_channels = 3).to(device)
    vgg_features = Vgg19().to(device)   
      

    # checkpoint = torch.load(cfg.trained_model_path)
    # G.load_state_dict(checkpoint['generator'])



    trfms = To_tensor()
    example_data = example_dataset(data_dir= args.input_dir, transform = trfms)
    example_loader = DataLoader(dataset = example_data, batch_size = 1)
    example_iter = iter(example_loader)

    print_log('Model compiled.', content_color = PrintColor['yellow'])

    print_log('Predicting', content_color = PrintColor['yellow'])

    G.eval()

    ss = SSIM().to(device)

    ssim = []
    PSNR = []

    with torch.no_grad():

      count = 0

      for step in tqdm(range(len(example_data))):
        # print(step)

        inp = next(example_iter)

        i_t = inp[0].to(device)
        i_s = inp[1].to(device)
        t_f = inp[2].to(device)

        t_f = t_f


        name = inp[2][0]
        

        o_sk, o_t, o_b, o_f = G(i_t, i_s, (i_t.shape[2], i_t.shape[3]))


        # tf = ((t_f + 1)*127.5)[0]
        # tf = tf.cpu().numpy().astype('uint8')

        t_f = t_f.permute(0, 3, 1, 2)[:,:,:63, :127]

        ps = getPSNR(o_f, t_f)

       
        # of = (o_f + 1)*127.5
        # of = of.permute(0, 2, 3, 1)[0].cpu().numpy().astype('uint8')

        # plt.imsave('test_evaluation/o_f/{}.png'.format(step), of)
        # plt.imsave('test_evaluation/t_f/{}.png'.format(step), tf)

        o_f = o_f.clamp(0, 1)
        t_f = t_f.clamp(0, 1)

        s = ss(o_f, t_f)

        PSNR.append(ps)
        ssim.append(s)



       
        if step == 1000:
          break
    print("PSNR: ",sum(PSNR)/len(PSNR))
    print("Structure Similiarity is: ",sum(ssim)/len(ssim))

            
if __name__ == '__main__':
  print('inmain')
  main()
  print_log('Evlaution finished.', content_color = PrintColor['yellow'])


