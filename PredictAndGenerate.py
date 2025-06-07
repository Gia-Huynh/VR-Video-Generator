#Basic system import
import os, sys, time, random, traceback, gc
import logging #Don't remove this
from tqdm import tqdm

import cv2
import numpy as np
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
from torch.amp import autocast #For FP16
from torch.multiprocessing import Pool, set_start_method #MultiProcessing
from depth_anything_v2.dpt import DepthAnythingV2
#Video Export Libraries
import multiprocessing as mp
from multiprocessing import shared_memory
from moviepy import ImageSequenceClip
#Support Functions
from SupportFunction import get_cutoff, load_model, load_and_set_video, random_sleep
DebugDir = "Debug/"
# Initialize logging
logging.basicConfig(level=logging.DEBUG, filename=DebugDir+'debug.txt', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

SubClipDir = "SubclipOutput/"
VideoDir = "Videos/She s A Beautiful Female Teacher, The Homeroom Teacher, Advisor To Our Team Sports, And My Lover Maria Nagai (1080).mp4"

Num_Workers = 7 #7 is Maximum for vitb, 13 for vits
encoder = 'vitb'
encoder_path = f'depth_anything_v2/checkpoints/depth_anything_v2_vitb.pth'

start_frame = 0
end_frame = 9999999999000 #if larger than video length, will be clipped off
Max_Frame_Count = 100

offset_bg = -0.0175
offset_fg =  0.02

#Global Variable
last_depth_flag = True
last_depth = None
last_frame = None
        
    
def nibba_woka(begin, end, max_frame_count = Max_Frame_Count, file_path = VideoDir):
    #Silence all output of child process
    if sys.stdout is None:
        out_file = open(DebugDir + str(begin)+'.txt', 'w')
        sys.stdout = out_file
        sys.stderr = out_file
        print ("Worker begin from ",begin," to ",end)
        sys.stdout.flush()

    random_sleep ((0, 24), "Sleeping a lil bit, then load models")
    model = load_model(encoder, encoder_path, DEVICE)
    cap, fps, video_length, width, height = load_and_set_video (file_path, begin)
    print ("video_length: ",video_length, ", begin and end: ",begin, end)
    random_sleep ((0, int(120 * (begin/video_length))), "Sleeping some more to diverge them process")

    begin_time = time.time()
    try:
        FrameList = []
        last_i = begin
        for i in range (begin, min (end, video_length)):
            _, raw_img = cap.read()
            if (raw_img is not None):
                FrameList.append(left_side_sbs(raw_img[:,:,[2,1,0]], model))
            else:
                FrameList.append(np.zeros((height, 2*width, 3), dtype = np.uint8))
                print ("Frame read error at i = ",i,", adding black frame to compensate.")
            #print (i,' ',len (FrameList),' ',(i == (min (end, video_length-1))),' ',end,' ',video_length)
            if (len (FrameList) == max_frame_count) or (i == (min (end-1, video_length-1))):
                print ("Writing file ", i, "with length (in frames):  ", len(FrameList))
                total_step = (min (end, video_length) - begin)
                step_taken = i - begin
                time_taken = (time.time() - begin_time)
                time_total = time_taken / step_taken * total_step
                time_left = time_taken / step_taken * (total_step - step_taken)
                print ("",str(step_taken / total_step * 100), "%, Time elapsed (minutes):", time_taken/60.0,", ETA:", time_left/60.0,", Estimated Total Time (minutes):", time_total/60.0)
                sys.stdout.flush()
                #Replace this part in the future with https://github.com/kkroening/ffmpeg-python/issues/246#issuecomment-916579919
                clip = ImageSequenceClip(FrameList, fps= fps)
                clip.write_videofile(SubClipDir+str(last_i)+"_"+str(i)+".mp4")
                last_i = i+1
                FrameList = []
                gc.collect()
        print ("Worker ending")
        sys.stdout.flush()
        try:
            out_file.close()
        except:
            pass
        return 0
        #return [i, left_side_sbs(raw_img)]
    except Exception as e:
        print(f"[ERROR] Segment {begin} failed: {e}")
        print(f"[ERROR] {begin} failed at frame {i}")
        print(traceback.format_exc())
        sys.stdout.flush()
        raise e
        return None
def left_side_sbs(raw_img, model):
    #Reuse old depth if frame is not much different shenanigan.
    global last_depth_flag
    global last_frame
    global last_depth
    if (last_frame is not None) and (np.sum (cv2.absdiff (cv2.stackBlur(raw_img, (5, 5)), cv2.stackBlur(last_frame, (3, 3)))) < 6000000) and (last_depth_flag == True):
        depth = last_depth
    else:     
        last_frame = raw_img.copy()
        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float16):
            depth = model.infer_image(cv2.stackBlur(raw_img, (5,5)))
        depth = cv2.stackBlur(depth, (3, 3)) #OG la (3, 3)
        #FUCK IT WE NORMALIZE
        depth = depth - depth.min()
        depth = depth/depth.max()*15
        if (last_depth_flag == False):
            depth = depth*0.6 + last_depth*0.4
            last_depth = depth.copy()
            last_depth_flag = True
        else:
            last_depth_flag = False
            last_depth = depth.copy()
    

    result_blank_mask = np.zeros(raw_img.shape[:2], dtype=np.uint8)[..., np.newaxis]
    result_img = np.zeros(raw_img.shape, dtype=raw_img.dtype)
    offset_range = [offset_bg * raw_img.shape[0], offset_fg * raw_img.shape[0]]
    limit_step = 16
    max_depth = limit_step
    
    cu = sorted(get_cutoff(depth, last_depth))
    nt = [cu[i+1]-cu[i] for i in range(len(cu)-1)]
    cu.pop() #Remove last element
    
    for i, curr_step in zip(cu, nt):
        bin_mask = ((i - 0.1 * curr_step) <= depth) & (depth < i + 1.1 * curr_step)
        masked_img = np.where(bin_mask[:, :, None], raw_img, 0)
        masked_mask = np.where(bin_mask[:, :, None], 1, 0)
        
        offset_x = int((i + (0.5 * curr_step)) / (limit_step - curr_step) * (offset_range[1] - offset_range[0]) + offset_range[0])
        if offset_x != 0:
            masked_img = np.roll(masked_img, shift=offset_x, axis=1)  # Shift along the width (x-axis)
            masked_mask = np.roll(masked_mask, shift=offset_x, axis=1)
        result_img = np.where(masked_img == 0, result_img, masked_img)
        result_blank_mask = np.where(masked_mask == 0, result_blank_mask, masked_mask)
        
    kernel_size = round(0.0037 * raw_img.shape[0]) #0.0047 is the OG, then 0.0036 works fine, 0.0024 is a bit too low.
    result_zero_mask = np.all(result_blank_mask == 0, axis=-1)
    
    
    kernel_expand = np.ones ((max(kernel_size-1, 1),  max(kernel_size-1, 1)))
    result_zero_mask = cv2.dilate(result_zero_mask.astype(np.uint8), kernel_expand,iterations = 1)

    #Fill result_img with blurred value from zero_mask
    result_zero_mask = result_zero_mask.astype(bool)
    result_img[result_zero_mask] = (cv2.stackBlur
                                    (np.roll
                                     (raw_img, shift=round(offset_x/3), axis=1)
                                    ,(limit_step*2 + 3, round(limit_step/8)*2 + 1)
                                    )
                                   )[result_zero_mask]
    result_img[:, 0:round(offset_x/3), :] = raw_img[:, 0:round(offset_x/3), :]
    return cv2.hconcat([result_img, raw_img])
def generate_from_video (VideoDir):
    pass
if __name__ == "__main__":
    set_start_method("spawn", force=True)
    _, _, video_length, _, _ = load_and_set_video (VideoDir, 0)
    step = int((min(end_frame, video_length) - start_frame)/Num_Workers + 1)
    frame_indices = range(start_frame, min(end_frame, video_length), step)
    with Pool(processes=Num_Workers) as pool:
        results = list(tqdm(pool.starmap(nibba_woka, [(i, i + step) for i in frame_indices]), total=len(frame_indices)))

    #nibba_woka (15550, 15610)
