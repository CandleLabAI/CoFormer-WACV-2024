import cv2
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import cfg
from Diffusion.vtnet import Generator
import gradio as gr
import glob
import os
from skimage import io
import numpy as np
from datagen import To_tensor_3
# from multi_translate import multilingual_translate
from torchvision import transforms as T
from PIL import Image
# from remote_generation import create_image_remotely



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G_hin = Generator(in_channels = 3).to(device)
G_hin_2 = Generator(in_channels = 3).to(device)
# G_ben = Generator(in_channels = 3).to(device)
# G_tam = Generator(in_channels = 3).to(device)
# G_chi = Generator(in_channels = 3).to(device)
# G_eng = Generator(in_channels = 3).to(device)

G_hin.eval()
G_hin_2.eval()
# G_tam.eval()
# G_ben.eval()
# G_chi.eval()

ckpt_hin_1 = '/DATA/ocr_team_2/VTNet/checkpoint/train_step-700000.model'
ckpt_hin_2 = '/DATA/ocr_team_2/VTNet/checkpoint/larger_vocab_eng_hin/train_step-680000.model'
# ckpt_ben = 'eng_ben_model_logs/train_step-320000.model'
# ckpt_tam = 'eng_tam_model_logs/train_step-280000.model'
# ckpt_chi = 'eng_chi_model_logs/train_step-280000.model'
# ckpt_eng = '/DATA/ocr_team_2/onkar/hin_eng_model_logs/train_step-280000.model'


# from trained model choose model accornding to language you want to transfer
G_hin.load_state_dict(torch.load(ckpt_hin_1)['generator'])
G_hin_2.load_state_dict(torch.load(ckpt_hin_2)['generator'])
# G_ben.load_state_dict(torch.load(ckpt_ben)['generator'])
# G_chi.load_state_dict(torch.load(ckpt_chi)['generator'])
# G_tam.load_state_dict(torch.load(ckpt_tam)['generator'])
# G_eng.load_state_dict(torch.load(ckpt_eng)['generator'])

# parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()


lang_id = {
  'english':'eng',
  'hindi':'hin',
  'bengali':'ben',
  'chinese':'chi',
  'tamil':'tam'
}


size = (128, 64)
def infer(i_s, i_t, size, model, path):

    tmfr = To_tensor_3()

    i_s = io.imread(i_s)
    # Check the number of channels in the image
    if len(i_s.shape) == 2 or i_s.shape[2] == 1:  # Grayscale image with 2 dimensions
        i_s = np.repeat(i_s[:, :, np.newaxis], 3, axis=2)  # Repeat the single channel to create a 3-channel image
    else:
        i_s = i_s[:, :, :3]
    orig_i_s_size = i_s.shape
    orig_i_s_size = (orig_i_s_size[1], orig_i_s_size[0])
    i_s = cv2.resize(i_s, size)

    i_t = io.imread(i_t)
    i_t = cv2.resize(i_t, size)
    i_t, i_s = tmfr([i_t, i_s])

    i_t = i_t.unsqueeze(0).cuda()
    i_s = i_s.unsqueeze(0).cuda()

    with torch.no_grad():
      o_sk, o_t, o_b, o_f = model(i_t, i_s, (i_t.shape[2], i_t.shape[3]))

    o_f = o_f.squeeze(0).detach().to('cpu').permute(1, 2, 0).numpy()
    o_f = 127.5*o_f + 127.5
    o_f = o_f.astype('uint8')
    o_f = cv2.resize(o_f, orig_i_s_size)
    plt.imsave(path, o_f)



def get_text(img):
    img_transform = T.Compose([
                T.Resize((32, 128), T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(0.5, 0.5)
            ])

    # img = Image.open('/DATA/ocr_team_2/VTNet/i_s/i_s_1.png').convert('RGB')
    img = img_transform(img).unsqueeze(0)

    logits = parseq(img)

    # Greedy decoding
    pred = logits.softmax(-1)
    label, confidence = parseq.tokenizer.decode(pred)
    return label[0]

def infer_gr(i_s, language):
    
    shape = i_s.shape
    text = get_text(Image.fromarray(i_s))
    print('recognition: ', text)
    translate_text = multilingual_translate(
      text,
      'facebook/m2m100_418M',
      language
    )

    print(translate_text)

    i_s = cv2.resize(i_s, (128, 64))

    i_s = torch.from_numpy(i_s).unsqueeze(0).permute(0, 3, 1, 2)
    i_s = (i_s / 127.5) - 1
    i_s = i_s.to(device)

    # i_t = cv2.imread(i_t)
    # i_t = cv2.cvtColor(i_t, cv2.COLOR_BGR2RGB)
    create_image_remotely(translate_text, lang_id[language])
    i_t = cv2.imread('i_t.png')
    i_t = cv2.cvtColor(i_t, cv2.COLOR_BGR2RGB)
    i_t = cv2.resize(i_t, (128, 64))
    i_t = torch.from_numpy(i_t).unsqueeze(0).permute(0, 3, 1, 2)
    i_t = (i_t / 127.5) - 1
    i_t = i_t.to(device)

    with torch.no_grad():
      if language == 'hindi':
        o_sk, o_t, o_b, o_f = G_hin(i_t, i_s, (i_t.shape[2], i_t.shape[3]))
      
      elif language == 'bengali':
        o_sk, o_t, o_b, o_f = G_ben(i_t, i_s, (i_t.shape[2], i_t.shape[3]))
      
      elif language == 'tamil':
        o_sk, o_t, o_b, o_f = G_tam(i_t, i_s, (i_t.shape[2], i_t.shape[3]))
      
      elif language == 'chinese':
        o_sk, o_t, o_b, o_f = G_chi(i_t, i_s, (i_t.shape[2], i_t.shape[3]))
      
      elif language == 'english':
        o_sk, o_t, o_b, o_f = G_eng(i_t, i_s, (i_t.shape[2], i_t.shape[3]))

    o_f = o_f.squeeze(0).detach().to('cpu').permute(1, 2, 0).numpy()
    o_f = 127.5*o_f + 127.5
    o_f = o_f.astype('uint8')
    # o_f = cv2.resize(o_f, (shape[1], shape[0]))
    # plt.imsave('results/{}.png'.format(name), o_f)
    return o_f

if __name__ == "__main__":

  # Give folder path here
  # if not os.path.exists(fpath + '/results'):
  #   os.mkdir(fpath + '/results')
  
  # path = 'absolute path'
  # i_s = glob.glob(path + '/i_s/*.png')
  # i_t = glob.glob(path + '/i_t/*.png')

  # size = (128, 64)
  # result_path = fpath + '/results/'

  # count = 0
  # for i, j in zip(i_s, i_t):
  #   name = i.split('/')[-1]
  #   infer(i, j, size, G_hin, name, result_path)
  #   count += 1

#   demo = gr.Interface(
#   infer_gr, 
#   inputs=[gr.Image(), gr.Dropdown(
#             ["english", "hindi", "bengali", "tamil"], label="Select Language", info="Select the language of your choice"
#         )], 
#   outputs="image",
#   title="Visual Translation, IIT-Jodhpur",
# )
#   demo.launch(share=True)


  # idx = 18
  # infer(f'/DATA/ocr_team_2/onkar/final_dataset/eng_hin/test/i_s/{idx}.png', f'/DATA/ocr_team_2/onkar/final_dataset/eng_hin/test/i_t/{idx}.png', (128, 64), G_hin, f'out_{idx}', 'results')
  # os.system(f'cp /DATA/ocr_team_2/onkar/final_dataset/eng_hin/test/i_s/{idx}.png ./results/origs/i_s_{idx}.png')
  # os.system(f'cp /DATA/ocr_team_2/onkar/final_dataset/eng_hin/test/i_t/{idx}.png ./results/origs/i_t_{idx}.png')

  data_root = './1k_translations/'
  num_samples = len(os.listdir(os.path.join(data_root, 'i_s')))
  
  for idx in tqdm(range(num_samples), desc="samples processed"):
    
    i_s_path = os.path.join(data_root, 'i_s', f'{idx}.jpg')
    i_t_path = os.path.join(data_root, 'i_t', f'{idx}.png')

    out_path_1 = os.path.join(data_root, 'out_m1', f'{idx}.png')
    out_path_2 = os.path.join(data_root, 'out_m2', f'{idx}.png')
    
    infer(i_s_path, i_t_path, (128, 64), G_hin, out_path_1)
    infer(i_s_path, i_t_path, (128, 64), G_hin_2, out_path_2)
  
  print("completed!")