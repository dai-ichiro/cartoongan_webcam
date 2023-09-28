import os
import sys 
import argparse
import numpy as np
import cv2
import torch
from  torchvision import transforms
from torchvision.datasets.utils import download_url

parser = argparse.ArgumentParser()
parser.add_argument(
    '--style',
    type=str,
    default='Hayao',
    choices=['Hayao', 'Hosoda', 'Paprika', 'Shinkai']
)
parser.add_argument(
    '--video',
    type=str,
    required=True,
)
opt = parser.parse_args()

style = opt.style
video = opt.video

pth_url = f"http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/{style}_net_G_float.pth"
download_url(pth_url, root = 'models', filename = os.path.basename(pth_url))

transformer_url = 'https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch/raw/master/network/Transformer.py'
transformer_fname = os.path.basename(transformer_url)

download_url(transformer_url, root = './network', filename = transformer_fname)
from network.Transformer import Transformer

max_size = 512
model_path = 'models'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Transformer()
model.load_state_dict(torch.load(os.path.join(model_path, f"{style}_net_G_float.pth")))
model.eval().to(device)

cap = cv2.VideoCapture(video)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

aspect = w / h
if aspect > 1:
    h = round(max_size / aspect)
    w = max_size
else:
    h = max_size
    w = round(max_size * aspect)

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('output.mp4',fourcc, fps, (w,h))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    input_array = cv2.resize(frame, dsize=(w, h), interpolation = cv2.INTER_CUBIC)

    input_tensor = transforms.ToTensor()(input_array).unsqueeze(0)
    input_tensor = -1 + 2 * input_tensor

    with torch.no_grad():
        output_tensor = model(input_tensor.to(device))

    output_image = output_tensor[0]
    output_image = (output_image * 0.5 + 0.5).to('cpu')

    output_array = (output_image.numpy() * 255).clip(0,255).astype('uint8')
    output_array = np.transpose(output_array, (1, 2, 0))

    out.write(output_array)  

cap.release()
out.release()

