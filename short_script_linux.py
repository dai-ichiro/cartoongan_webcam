import os
import sys 
import numpy as np
import cv2
import torch
from  torchvision import transforms
from autogluon.core.utils import download, mkdir

mkdir('models')
download('http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Hayao_net_G_float.pth', path='models')
download('http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Hosoda_net_G_float.pth', path='models')
download('http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Paprika_net_G_float.pth', path='models')
download('http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Shinkai_net_G_float.pth', path='models')

mkdir('network')
download('https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch/raw/master/network/Transformer.py', path='network')

from network.Transformer import Transformer

max_size = 450
model_path = 'models'
style = sys.argv[1]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Transformer()
model.load_state_dict(torch.load(os.path.join(model_path, style + '_net_G_float.pth')))
model.eval().to(device)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    h, w = frame.shape[:2]
    aspect = w / h
    if aspect > 1:
        h = round(max_size / aspect)
        w = max_size
    else:
        h = max_size
        w = round(max_size / aspect)

    input_array = cv2.resize(frame, dsize=(w, h), interpolation = cv2.INTER_CUBIC)

    input_tensor = transforms.ToTensor()(input_array).unsqueeze(0)
    input_tensor = -1 + 2 * input_tensor

    with torch.no_grad():
        output_tensor = model(input_tensor.to(device))

    output_image = output_tensor[0]
    output_image = (output_image * 0.5 + 0.5).to('cpu')

    output_array = (output_image.numpy() * 255).clip(0,255).astype('uint8')
    output_array = np.transpose(output_array, (1, 2, 0))

    cv2.imshow('result', output_array)   

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
