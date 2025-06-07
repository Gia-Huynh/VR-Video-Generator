import cv2
import torch
from torch.multiprocessing import Pool, set_start_method
set_start_method("spawn", force=True)
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
#from decord import VideoReader, cpu
from tqdm import tqdm
from moviepy import ImageSequenceClip
import os, sys
import gc
os.environ["DECORD_THREADS"] = "1"
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

import logging

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


SubClipDir = "SubclipOutput/"
VideoDir = "hhd800.com@SORA-343.mp4"
Max_Frame_Count = 128
Num_Workers = 8

class LoggerWriter:
    def __init__(self, level):
        pass

    def write(self, message):
        pass

    def flush(self):
        pass
        
def load_model():
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    encoder = 'vitb'
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'depth_anything_v2/checkpoints/depth_anything_v2_vitb.pth', map_location=DEVICE))
    model = model.to(DEVICE).eval()
    return model

def nibba_woka(begin, end, max_frame_count = Max_Frame_Count, file_path = VideoDir):
    #logging.info(f"Processing frames from {begin} to {end}")
    #print ("Loading models")
    model = load_model()
    
    #Silence all output of child process
    if sys.stdout is None:
        log = logging.getLogger('foobar')
        sys.stdout = LoggerWriter(log.debug)
        sys.stderr = LoggerWriter(log.warning)
        
    cap  = cv2.VideoCapture(VideoDir)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # get total number of frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, begin) # set frame position
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        FrameList = []
        for i in range (begin, min (end, video_length)):
            _, raw_img = cap.read()
            FrameList.append(left_side_sbs(raw_img[:,:,[2,1,0]], model))
            if (len (FrameList) == max_frame_count) or (i == (min (end, video_length)-1)):
                #logging.info(f"Writing {i-len (FrameList)+1}_{i}.mp4")
                #Replace this part in the future with https://github.com/kkroening/ffmpeg-python/issues/246#issuecomment-916579919
                clip = ImageSequenceClip(FrameList, fps= fps)
                clip.write_videofile(SubClipDir+str(i-len (FrameList)+1)+"_"+str(i+1)+".mp4")
                FrameList = []
                gc.collect()
        return 0
        #return [i, left_side_sbs(raw_img)]
    except Exception as e:
        print(f"[ERROR] Frame {begin} failed: {e}")
        raise e
        return None
    
def left_side_sbs(raw_img, model):
    with torch.no_grad():  # Ensure single-threaded inference
        depth = model.infer_image(raw_img) 
        
    #depth = cv2.blur(depth, (3, 3))    
    result_img = np.zeros(raw_img.shape, dtype=raw_img.dtype)
    offset_range = [-0.01 * raw_img.shape[0], 0.015 * raw_img.shape[0]]
    step = 1
    min_step, max_step = 0, np.min([int(depth.max() + 0.5), 14])
    
    for i in np.arange(min_step, max_step, step):
        bin_mask = ((i - 0.1 * step) <= depth) & (depth < i + 1.25 * step)
        masked_img = np.where(bin_mask[:, :, None], raw_img, 0)
        offset_x = int((i - min_step) / (max_step - step - min_step) * (offset_range[1] - offset_range[0]) + offset_range[0])
        if offset_x != 0:
            translation_matrix = np.float32([[1, 0, offset_x], [0, 1, 0]])
            masked_img = cv2.warpAffine(masked_img, translation_matrix, (masked_img.shape[1], masked_img.shape[0]))
        result_img = np.where(masked_img == 0, result_img, masked_img)
    
    result_zero_mask = np.all(result_img == 0, axis=-1)
    result_img[result_zero_mask] = cv2.blur(result_img, (7, 7))[result_zero_mask]
    return cv2.hconcat([result_img, raw_img])
if __name__ == "__main__":      
    cap  = cv2.VideoCapture(VideoDir)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # get total number of frames
    step = int(video_length/Num_Workers + 1)
    frame_indices = range(0, video_length, step)
    #STOP
    del cap
    with Pool(processes=Num_Workers) as pool:
        results = list(tqdm(pool.starmap(nibba_woka, [(i, i + step) for i in frame_indices]), total=len(frame_indices)))


#executor.submit(nibba_woka, i, i+step, VideoCap): i for i in frame_indices
