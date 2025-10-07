import cv2
import torch
import timeit
import numpy as np
Input_Img = "TestImg.png"
DEVICE = "cuda"

img = cv2.imread (Input_Img)
img_gpu = torch.from_numpy(img).to(DEVICE)

def cpu_roll():
    t = []
    for i in range (-5, 5, 2):
        t = np.roll (img, shift=i, axis=1)
    return t

def gpu_roll():
    t = []
    for i in range (-5, 5, 2):
        t.append(torch.roll (img_gpu, shifts=i, dims=1).cpu().numpy())
    torch.cuda.synchronize()
    return t

#print(timeit.timeit(cpu_roll, number=200))
#print(timeit.timeit(gpu_roll, number=200))
