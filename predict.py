# Predict script 
# author: Niwhskal

import os
import argparse
import cfg
import torch
from tqdm import tqdm
import numpy as np
from model import Generator, Discriminator, Vgg19
from utils import *
from datagen import *
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import cv2

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = 'cpu'

print('aheharaebdbjk')

def main():
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('--input_dir', help = 'Directory containing xxx_i_s and xxx_i_t with same prefix',
                        default = cfg.example_data_dir)
    parser.add_argument('--save_dir', help = 'Directory to save result', default = cfg.predict_result_dir)
    parser.add_argument('--checkpoint', help = 'ckpt', default = cfg.ckpt_path)
    args = parser.parse_args()

    assert args.input_dir is not None
    assert args.save_dir is not None
    assert args.checkpoint is not None

    print_log('model compiling start.', content_color = PrintColor['yellow'])

    G = Generator(in_channels = 3).to(device)
    D1 = Discriminator(in_channels = 6).to(device)
    D2 = Discriminator(in_channels = 6).to(device)  
    vgg_features = Vgg19().to(device)   
      
    G_solver = torch.optim.Adam(G.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))
    D1_solver = torch.optim.Adam(D1.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))
    D2_solver = torch.optim.Adam(D2.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))

    checkpoint = torch.load('eng_hin_model_logs/train_step-80000.model')
    G.load_state_dict(checkpoint['generator'])
    D1.load_state_dict(checkpoint['discriminator1'])
    D2.load_state_dict(checkpoint['discriminator2'])
    G_solver.load_state_dict(checkpoint['g_optimizer'])
    D1_solver.load_state_dict(checkpoint['d1_optimizer'])
    D2_solver.load_state_dict(checkpoint['d2_optimizer'])

    trfms = To_tensor_2()
    example_data = example_dataset_2(data_dir= args.input_dir, transform = trfms)
    example_loader = DataLoader(dataset = example_data, batch_size = 1)
    example_iter = iter(example_loader)

    print_log('Model compiled.', content_color = PrintColor['yellow'])

    print_log('Predicting', content_color = PrintColor['yellow'])

    G.eval()
    D1.eval()
    D2.eval()

    with torch.no_grad():

      count = 0

      for step in tqdm(range(len(example_data))):
        # print(step)

        try:

          inp = example_iter.next()

        except StopIteration:

          example_iter = iter(example_loader)
          inp = example_iter.next()

        i_t = inp[0].to(device)
        i_s = inp[1].to(device)
        # t_f = inp[2].to(device)
        name = inp[2][0]
        shape = inp[3]

        h, w = shape
        # print(h[0].item(), w[0].item())
       
        

        o_sk, o_t, o_b, o_f = G(i_t, i_s, (i_t.shape[2], i_t.shape[3]))

        i_t = i_t.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        i_s = i_s.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        # t_f = t_f.detach().cpu().squeeze(0).numpy()



        i_t = 127.5*i_t + 127.5
        i_t = i_t.astype('uint8')

        i_s = 127.5*i_s + 127.5
        i_s = i_s.astype('uint8')

        # t_f = 127.5*t_f + 127.5
        # t_f = t_f.astype('uint8')

        # plt.imsave('predictions/eng-hin/i_t/i_t{}.png'.format(count), (i_t))
        # plt.imsave('predictions/eng-hin/i_s/i_s{}.png'.format(count), i_s)
        # plt.imsave('human_evaluation/gt/tam_gt_{}.png'.format(count), t_f)

        o_sk = o_sk.squeeze(0).detach().to('cpu')
        o_t = o_t.squeeze(0).detach().to('cpu')
        o_b = o_b.squeeze(0).detach().to('cpu')
        o_f = o_f.squeeze(0).detach().to('cpu').permute(1, 2, 0).numpy()

        o_f = 127.5*o_f + 127.5
        o_f = o_f.astype('uint8')

        # o_f = cv2.resize(o_f, (w[0].item(), h[0].item()))

        plt.imsave('7th_feb_ppt/results_hin/'+name, o_f)

        count += 1
       
        # if step == 10:
        #   break

            
if __name__ == '__main__':
  print('inmain')
  main()
  print_log('predicting finished.', content_color = PrintColor['yellow'])


