from models import *
import torch
import torch.nn as nn
import torchvision.transforms as tfs
from torchvision.transforms.functional import to_pil_image

def net_enhance(image):

    model_dir='trained_models/its_train_ffa_3_20.pk'
    device='cuda' if torch.cuda.is_available() else 'cpu'
    ckp=torch.load(model_dir,map_location=device)
    net=FFA(gps=3,blocks=20)
    net=nn.DataParallel(net)
    net.load_state_dict(ckp['model'])
    net.eval()


    haze1 = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
    ])(image)[None,::]

    with torch.no_grad():
        pred = net(haze1)
    ts=torch.squeeze(pred.clamp(0,1).cpu())
    output_image = to_pil_image(ts)  # 转为 PIL.Image

    return output_image

