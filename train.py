
import numpy as np
import os
import cv2
import random
import torch
import torchvision.transforms
from utils import *
import cfg
from tqdm import tqdm
import torchvision.transforms.functional as F
import torch.nn as nn
from skimage.transform import resize
from skimage import io
from Diffusion.model import Generator, Discriminator, Vgg19
from torchvision import models, transforms, datasets
from loss import build_generator_loss, build_discriminator_loss
from datagen import datagen_srnet, example_dataset, To_tensor
from torch.utils.data import Dataset, DataLoader
import logging
from torch.utils.tensorboard import SummaryWriter


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        
def custom_collate(batch):
    
    i_t_batch, i_s_batch = [], []
    t_sk_batch, t_t_batch, t_b_batch, t_f_batch = [], [], [], []
    # mask_t_batch = []
    # r = random.randint(0, 2)
    # if r == 0:
    #   resize_factor = (512, 256)
    # elif r == 1:
    #   resize_factor = (256, 128)

    # elif r == 2:
    #   resize_factor = (128, 64)
    # w_sum = 0
    
    for items in batch:
        
        # items = []

        # for i in item:
        #   i = cv2.resize(i, resize_factor)
        #   items.append(i)
   
        i_t, i_s, t_sk, t_t, t_b, t_f = items

        t_sk = np.expand_dims(t_sk, axis = -1) 
        

        i_t = i_t.transpose((2, 0, 1))
        i_s = i_s.transpose((2, 0, 1))
        t_sk = t_sk.transpose((2, 0, 1))
        t_t = t_t.transpose((2, 0, 1))
        t_b = t_b.transpose((2, 0, 1))
        t_f = t_f.transpose((2, 0, 1))
         

        i_t_batch.append(i_t) 
        i_s_batch.append(i_s)
        t_sk_batch.append(t_sk)
        t_t_batch.append(t_t) 
        t_b_batch.append(t_b) 
        t_f_batch.append(t_f)


    i_t_batch = np.stack(i_t_batch)
    i_s_batch = np.stack(i_s_batch)
    t_sk_batch = np.stack(t_sk_batch)
    t_t_batch = np.stack(t_t_batch)
    t_b_batch = np.stack(t_b_batch)
    t_f_batch = np.stack(t_f_batch)
    

    i_t_batch = torch.from_numpy(i_t_batch.astype(np.float32) / 127.5 - 1.) 
    i_s_batch = torch.from_numpy(i_s_batch.astype(np.float32) / 127.5 - 1.) 
    t_sk_batch = torch.from_numpy(t_sk_batch.astype(np.float32) / 255.) 
    t_t_batch = torch.from_numpy(t_t_batch.astype(np.float32) / 127.5 - 1.) 
    t_b_batch = torch.from_numpy(t_b_batch.astype(np.float32) / 127.5 - 1.) 
    t_f_batch = torch.from_numpy(t_f_batch.astype(np.float32) / 127.5 - 1.) 
      

      
    return [i_t_batch, i_s_batch, t_sk_batch, t_t_batch, t_b_batch, t_f_batch]

def clip_grad(model):
    
    for h in model.parameters():
        h.data.clamp_(-0.01, 0.01)

def main():
  
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # device = 'cuda:0,1'#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_name = get_train_name()
    
    print_log('Initializing SRNET', content_color = PrintColor['yellow'])
    
    train_data = datagen_srnet(cfg)
    exdata = train_data
    
    train_data = DataLoader(
                            dataset = train_data, 
                            batch_size = cfg.batch_size, 
                            shuffle = True, 
                            collate_fn=custom_collate, 
                            pin_memory = True, 
                            num_workers=2
                            )
    
    trfms = To_tensor()
    # example_data = example_dataset(transform = trfms)
        
    example_loader = DataLoader(dataset = exdata, batch_size = 1, collate_fn = custom_collate, shuffle = False)
    
    print_log('training start.', content_color = PrintColor['yellow'])
        
    G = Generator(in_channels = 3).cuda()
    
    D1 = Discriminator(in_channels = 6).cuda()
    
    D2 = Discriminator(in_channels = 6).cuda()
        
    vgg_features = Vgg19().cuda()
        
    G_solver = torch.optim.Adam(G.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))
    D1_solver = torch.optim.Adam(D1.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))
    D2_solver = torch.optim.Adam(D2.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))

    #g_scheduler = torch.optim.lr_scheduler.MultiStepLR(G_solver, milestones=[30, 200], gamma=0.5)
    
    #d1_scheduler = torch.optim.lr_scheduler.MultiStepLR(D1_solver, milestones=[30, 200], gamma=0.5)
    
    #d2_scheduler = torch.optim.lr_scheduler.MultiStepLR(D2_solver, milestones=[30, 200], gamma=0.5)
    
    try:
      checkpoint = torch.load(cfg.ckpt_path)
      G.load_state_dict(checkpoint['generator'])
      D1.load_state_dict(checkpoint['discriminator1'])
      D2.load_state_dict(checkpoint['discriminator2'])
      G_solver.load_state_dict(checkpoint['g_optimizer'])
      D1_solver.load_state_dict(checkpoint['d1_optimizer'])
      D2_solver.load_state_dict(checkpoint['d2_optimizer'])
      
      # g_scheduler.load_state_dict(checkpoint['g_scheduler'])
      # d1_scheduler.load_state_dict(checkpoint['d1_scheduler'])
      # d2_scheduler.load_state_dict(checkpoint['d2_scheduler'])
      
      print('Resuming after loading...')
    except FileNotFoundError:

      print('checkpoint not found')
      pass 
    

    requires_grad(G, False)

    requires_grad(D1, True)
    requires_grad(D2, True)


    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0
    count = 0
        
    
    trainiter = iter(train_data)
    example_iter = iter(example_loader)


    
    K = torch.nn.ZeroPad2d((0, 1, 1, 0))

    for step in tqdm(range(cfg.max_iter)):
        
        D1_solver.zero_grad()
        D2_solver.zero_grad()
        
        if ((step+1) % cfg.save_ckpt_interval == 0):
            
            torch.save(
                {
                    'generator': G.state_dict(),
                    'discriminator1': D1.state_dict(),
                    'discriminator2': D2.state_dict(),
                    'g_optimizer': G_solver.state_dict(),
                    'd1_optimizer': D1_solver.state_dict(),
                    'd2_optimizer': D2_solver.state_dict(),
                    #'g_scheduler' : g_scheduler.state_dict(),
                    #'d1_scheduler':d1_scheduler.state_dict(),
                    #'d2_scheduler':d2_scheduler.state_dict(),
                },
                cfg.checkpoint_savedir+f'train_step-{step+1}.model',
            )
        
        try:

          i_t, i_s, t_sk, t_t, t_b, t_f = next(trainiter)

        except StopIteration:

          trainiter = iter(train_data)
          i_t, i_s, t_sk, t_t, t_b, t_f = next(trainiter)
                
        i_t = i_t.cuda()
        i_s = i_s.cuda()
        t_sk = t_sk.cuda()
        t_t = t_t.cuda()
        t_b = t_b.cuda()
        t_f = t_f.cuda()


                
        #inputs = [i_t, i_s]
        labels = [t_sk, t_t, t_b, t_f]
        
        o_sk, o_t, o_b, o_f = G(i_t, i_s, (i_t.shape[2], i_t.shape[3])) #Adding dim info

        
        o_sk = K(o_sk)
        o_t = K(o_t)
        o_b = K(o_b)
        o_f = K(o_f)
                
        #print(o_sk.shape, o_t.shape, o_b.shape, o_f.shape)
        #print('------')
        #print(i_s.shape)
        
        i_db_true = torch.cat((t_b, i_s), dim = 1)
        i_db_pred = torch.cat((o_b, i_s), dim = 1)
        
        i_df_true = torch.cat((t_f, i_t), dim = 1)
        i_df_pred = torch.cat((o_f, i_t), dim = 1)
        
        o_db_true = D1(i_db_true)
        o_db_pred = D1(i_db_pred)
        
        o_df_true = D2(i_df_true)
        o_df_pred = D2(i_df_pred)
        
        i_vgg = torch.cat((t_f, o_f), dim = 0)
        
        out_vgg = vgg_features(i_vgg)
        
        db_loss = build_discriminator_loss(o_db_true,  o_db_pred)
        
        df_loss = build_discriminator_loss(o_df_true, o_df_pred)
                
        db_loss.backward()
        df_loss.backward()
        
        D1_solver.step()
        D2_solver.step()
        
        #d1_scheduler.step()
        #d2_scheduler.step()
        
        clip_grad(D1)
        clip_grad(D2)
        
        
        if ((step+1) % 2 == 0):
            
            requires_grad(G, True)

            requires_grad(D1, False)
            requires_grad(D2, False)
            
            G_solver.zero_grad()
            
            o_sk, o_t, o_b, o_f = G(i_t, i_s, (i_t.shape[2], i_t.shape[3]))
            
            o_sk = K(o_sk)
            o_t = K(o_t)
            o_b = K(o_b)
            o_f = K(o_f)



            #print(o_sk.shape, o_t.shape, o_b.shape, o_f.shape)
            #print('------')
            #print(i_s.shape)

            i_db_true = torch.cat((t_b, i_s), dim = 1)
            i_db_pred = torch.cat((o_b, i_s), dim = 1)

            i_df_true = torch.cat((t_f, i_t), dim = 1)
            i_df_pred = torch.cat((o_f, i_t), dim = 1)

            o_db_pred = D1(i_db_pred)

            o_df_pred = D2(i_df_pred)

            i_vgg = torch.cat((t_f, o_f), dim = 0)

            out_vgg = vgg_features(i_vgg)
            
            out_g = [o_sk, o_t, o_b, o_f]
        
            out_d = [o_db_pred, o_df_pred]
        
            g_loss, detail = build_generator_loss(out_g, out_d, out_vgg, labels)    
                
            g_loss.backward()
            
            G_solver.step()
            
            #g_scheduler.step()
                        
            requires_grad(G, False)

            requires_grad(D1, True)
            requires_grad(D2, True)
            
        if ((step+1) % cfg.write_log_interval == 0):
            
            print('Iter: {}/{} | Gen: {} | D_bg: {} | D_fus: {}'.format(step+1, cfg.max_iter, g_loss.item(), db_loss.item(), df_loss.item()))



def plot(path):
  im = cv2.imread(path)
  plt.imshow(im)
  plt.show()
  im = cv2.resize(im, (128, 64))
  plt.imshow(im)
  plt.show()

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

class IAMDataset(Dataset):
  def __init__(self, annotation_path):

    self.class_dict = {}
    for i, j in enumerate(os.listdir('content/dataset')):
      self.class_dict[j] = i
    
    self.paths = torch.load(annotation_path)
    self.images = list(self.paths.keys())
    
    self.transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    self.token = torch.load('ICDAR-2023/char_token_dict.json')

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, idx):

    img = cv2.imread(self.images[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 64))

    img = self.transforms(img)
    label = self.tokenizer(self.paths[self.images[idx]])

    s_id = torch.tensor(self.class_dict[self.images[idx].split('/')[-2]]).to(torch.int64)

    return img, label, s_id

  def tokenizer(self, text):
    text = list(text)
    text = torch.tensor([self.token[i] for i in text])
    return text.to(torch.int64)

# token = torch.load('ICDAR-2023/char_token_dict.json')
def tokenizer(text):
  text = list(text)
  text = torch.tensor([token[i] for i in text])
  return text.to(torch.int64)

def collate_fn(batch):

    images, labels, s_id = [], [], []
    for i, l, s in batch:
        images += [i]
        labels += [l]
        s_id += [s]
    tensors = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0.)
    images = torch.stack(images)
    s_id = torch.stack(s_id)

    return images, tensors, s_id

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

        self.lstm = nn.LSTM(300, channels)

    def forward(self, x, x_text):
        x = x.view(-1, self.channels, self.size[0] * self.size[1]).swapaxes(1, 2)
        x_ln = self.ln(x)
        x_t, (_, _) = self.lstm(x_text)
        
        attention_value, _ = self.mha(x_ln, x_t, x_t)
        # print(attention_value.shape)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size[0], self.size[1])


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, **kwargs):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=(1, 1), bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=(1, 1), bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

        self.sid_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t, s_em):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        sid = self.sid_layer(s_em)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb + sid


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

        self.sid_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t, s_em):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        sid = self.sid_layer(s_em)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb + sid


class UNet(nn.Module):
    def __init__(self,emb, c_in=3, c_out=3, time_dim=256, device="cuda", 
                 num_classes=200, embedding_size=256):
        super().__init__()

        # emb = np.load('/content/char_embedding.npy')
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, (32, 64))
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, (16, 32))
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, (8, 16))

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.sa4 = SelfAttention(256, (8, 16))
        self.up1 = Up(512, 128)

        self.sa5 = SelfAttention(128, (16, 32))
        self.up2 = Up(256, 64)

        self.sa6 = SelfAttention(64, (32, 64))
        self.up3 = Up(128, 64)

        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.embedding_text = nn.Embedding(94, 300)
        self.embedding_text.weight.data = torch.tensor(emb).to(torch.float32)

        self.emb_sid = nn.Embedding(num_classes, embedding_size)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, x_t, s_id):

        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        sid = self.emb_sid(s_id)
        x_t = self.embedding_text(x_t)
        
        # print(t.shape)
        x1 = self.inc(x)
        x2 = self.down1(x1, t, sid)
        x2 = self.sa1(x2, x_t)
        x3 = self.down2(x2, t, sid)
        x3 = self.sa2(x3, x_t)
        x4 = self.down3(x3, t, sid)
        x4 = self.sa3(x4, x_t)
        
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x4 = self.sa4(x4, x_t)
        x = self.up1(x4, x3, t, sid)
        
        x = self.sa5(x, x_t)
        x = self.up2(x, x2, t, sid)
        
        x = self.sa6(x, x_t)
        x = self.up3(x, x1, t, sid)
        
        output = self.outc(x)
        return output

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=(64, 128), device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, x_text, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x_text = tokenizer(x_text)
            x_text = x_text.repeat(n, 1).to(self.device)
            x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, x_text, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, x_text, labels)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

def train(epochs, run_name, num_classes,loader, device):

    emb = np.load('ICDAR-2023/char_embedding.npy')
    setup_logging(run_name)

    model = UNet(emb=emb, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('models/ICDAR_demo/ckpt.pt'))


    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    optimizer.load_state_dict(torch.load('models/ICDAR_demo/optim.pt'))

    mse = nn.MSELoss()
    diffusion = Diffusion(device=device)
    logger = SummaryWriter(os.path.join("runs", run_name))
    l = len(loader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    ema_model.load_state_dict(torch.load('models/ICDAR_demo/ema_ckpt.pt'))

    for epoch in range(61, epochs+1):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(loader)
        for i, (images, text, s_id) in enumerate(pbar):
            images = images.to(device)
            text = text.to(device)
            s_id = s_id.to(device)

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, text, s_id)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            labels = torch.arange(10).long().to(device)
            x_text = 'onkarsus'
            sampled_images = diffusion.sample(model, n=len(labels), x_text=x_text, labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), x_text=x_text, labels=labels)
            # plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", run_name, f"optim.pt"))

            
                
if __name__ == '__main__':
    import warnings
    warnings.simplefilter("ignore")
    main()
