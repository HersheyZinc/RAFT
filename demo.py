import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz, frame_utils
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)



def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    #import matplotlib.pyplot as plt
    #plt.imshow(img_flo / 255.0)
    #plt.show()

    #cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    #cv2.waitKey()
    cv2.imwrite("test.jpg", flo)


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    outpath = args.outpath
    os.makedirs(outpath, exist_ok=True)
    for_path, back_path = os.path.join(outpath, "flow"), os.path.join(outpath, "flow_backward")
    os.makedirs(for_path, exist_ok=True)
    os.makedirs(back_path, exist_ok=True)

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        i=0
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            name = "{:05d}.flo".format(i)
            i+=1

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flow_low, flow_down = model(image2, image1, iters=20, test_mode=True)
            flow_up_ = padder.unpad(flow_up[0]).permute(1, 2, 0).cpu().numpy()
            flow_down_ = padder.unpad(flow_down[0]).permute(1, 2, 0).cpu().numpy()
            filename_fr = imfile1.split("/")[-1].split(".")[0]
            filename_bk = imfile2.split("/")[-1].split(".")[0]
            output_file_up = os.path.join(for_path, name)
            output_file_down = os.path.join(back_path,name)
            frame_utils.writeFlow(output_file_up, flow_up_)
            frame_utils.writeFlow(output_file_down, flow_down_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--outpath',help="path to store flow output")
    args = parser.parse_args()

    demo(args)
