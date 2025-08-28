import sys
import torch
from pathlib import Path
from PIL import Image
import numpy as np

# 添加 yolov5 文件夹路径到系统路径，方便导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'yolov5'))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, cv2
from utils.plots import Annotator, colors

# 加载模型，只加载一次
model_path = str(Path(__file__).resolve().parent.parent / 'web' / 'trained_models' / 'best.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(model_path, device=device)
stride, names, pt = model.stride, model.names, model.pt

def letterbox(im, new_shape=640, color=(114, 114, 114), stride=32, auto=True):
    import math
    shape = im.shape[:2]  # 当前形状 [高, 宽]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2

    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:
        # Calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    return coords


def yolov5_recognize(pil_img):
    # PIL转numpy BGR
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img0 = img.copy()
    img, ratio, (dw, dh) = letterbox(img, 640, stride=stride, auto=pt)

    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to CHW
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndim == 3:
        img = img.unsqueeze(0)

    pred = model(img)

    conf_thres = 0.25
    iou_thres = 0.45
    pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000)

    for det in pred:
        annotator = Annotator(img0, line_width=2, example=str(names))
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))

        img0 = annotator.result()

    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img0)
