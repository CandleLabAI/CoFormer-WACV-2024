import torch
from torchvision import transforms as T
import cv2

import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule

# Load model and image transforms
parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
def inference(img):
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



