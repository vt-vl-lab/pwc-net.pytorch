import cv2
import numpy as np

import torch

from PWC_src import PWC_Net
from PWC_src import flow_to_image

FLOW_SCALE = 20.0


if __name__ == '__main__':
    # Prepare img pair (size need to be a multipler of 64)
    im1 = cv2.imread('example/0img0.ppm')
    im2 = cv2.imread('example/0img1.ppm')
    im1 = torch.from_numpy((im1/255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    im2 = torch.from_numpy((im2/255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    im1_v = im1.cuda()
    im2_v = im2.cuda()

    # Build model
    pwc = PWC_Net(model_path='models/sintel.pytorch')
    #pwc = PWC_Net(model_path='models/chairs-things.pytorch')
    pwc = pwc.cuda()
    pwc.eval()

    import time
    start = time.time()
    flow = FLOW_SCALE*pwc(im1_v, im2_v)
    print(time.time()-start)
    flow = flow.data.cpu()
    flow = flow[0].numpy().transpose((1,2,0))
    flow_im = flow_to_image(flow)

    # Visualization
    import matplotlib.pyplot as plt
    plt.imshow(flow_im)
    plt.show()

