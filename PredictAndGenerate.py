#Profiling
from line_profiler import LineProfiler
#Basic system import
import os, sys, time, random, traceback, gc, math
import logging #Don't remove this
from tqdm import tqdm
import cv2
import numpy as np
import torch
#DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
from torch.amp import autocast #For FP16
from depth_anything_v2.dpt import DepthAnythingV2
#Video Export Libraries
import subprocess
import multiprocessing as mp
from multiprocessing import shared_memory
from multiprocessing import Process, Queue, set_start_method
from moviepy import ImageSequenceClip
#Support Functions
from SupportFunction import dump_line_profile_to_csv, get_length, get_cutoff, load_model, load_and_set_video, random_sleep, redirrect_stdout, print_flush, remove_all_file
import SupportFunction as SpF
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--DebugDir',   type=str,   default="Debug/")
parser.add_argument('--SubClipDir', type=str,   default="D:/TEMP/JAV Subclip/")
parser.add_argument('--VideoDir',   type=str,   default="Videos/Maria Nagai.mp4")
parser.add_argument('--OutputDir',  type=str,   default="SBS Maria Nagai.mp4")
parser.add_argument('--encoder',    type=str,   default='vitb')
parser.add_argument('--encoder_path',
                                    type=str,   default='depth_anything_v2/checkpoints/depth_anything_v2_vitb.pth')
parser.add_argument('--offset_fg',  type=float, default=0.0125)
parser.add_argument('--offset_bg',  type=float, default=-0.0225)
parser.add_argument('--Num_Workers',type=int,   default=20)
parser.add_argument('--num_gpu',    type=int,   default=1)
parser.add_argument('--Num_GPU_Workers',
                                    type=int,   default=3)
parser.add_argument('--Max_Frame_Count',
                                    type=int,   default=20)
parser.add_argument('--start_frame',type=int,   default=0)
parser.add_argument('--end_frame',  type=int,   default=9999999999999) #82800 + 450 #9999999999999, 27000 is 15 minutes, 9000 5 minutes

args = parser.parse_args()

DebugDir = args.DebugDir
SubClipDir = args.SubClipDir
VideoDir = args.VideoDir
OutputDir = args.OutputDir
encoder = args.encoder
encoder_path = args.encoder_path
offset_fg = args.offset_fg
offset_bg = args.offset_bg
Num_Workers = args.Num_Workers
num_gpu = args.num_gpu
Num_GPU_Workers = args.Num_GPU_Workers
Max_Frame_Count = args.Max_Frame_Count
start_frame = args.start_frame
end_frame = args.end_frame

"""DebugDir = "Debug/"
SubClipDir = "D:/TEMP/JAV Subclip/" #"Debug/"
VideoDir = "Videos/She s A Beautiful Female Teacher, The Homeroom Teacher, Advisor To Our Team Sports, And My Lover Maria Nagai (1080).mp4"
OutputDir = "SBS Maria Nagai Harder.mp4"
encoder = 'vitb'
encoder_path = f'depth_anything_v2/checkpoints/depth_anything_v2_vitb.pth'

offset_fg = 0.0125 #0.009 #0.0117
offset_bg = 0.025 * -1

Num_Workers = 30
num_gpu = 1
Num_GPU_Workers = 3 #Total
Max_Frame_Count = 20
start_frame = 82800 #82800 #0
end_frame = 82800 + 3600 #82800 + 1800 #9999999999999, 27000 is 15 minutes, 9000 5 minutes"""
#if smaller than video length, will be clipped off

if __name__ == "__main__":
    remove_all_file (DebugDir)
    remove_all_file (SubClipDir)
    
#Yes, order of inputs is important: ffmpeg [global options] [input options] -i input [output options] output.
#Options in [input options] are applied to the input. Options in [output options] are applied to the output.
_, fps, video_length, width, height = load_and_set_video (VideoDir, 0)
ffmpeg_device = 'nvidia'
ffmpeg_config = [
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'rgb24',
    '-s',
    f'{2*width}x{height}', '-r', str(fps),
    '-i', '-',  # stdin
    '-an',
    '-pix_fmt', 'yuv420p'
    ]

if (ffmpeg_device == 'cpu'):
    ffmpeg_encoder = 'libx264' #'libx264' for cpu
    ffmpeg_config += ['-c:v', ffmpeg_encoder]
    ffmpeg_crf = '20'
    ffmpeg_preset = 'veryfast'    
    ffmpeg_config += ['-crf', ffmpeg_crf, '-preset', ffmpeg_preset]  # use crf with libx264
elif (ffmpeg_device == 'nvidia'):
    ffmpeg_encoder = 'hevc_nvenc' #h264_nvenc for h264, hevc_nvenc for h265.
    ffmpeg_config += ['-c:v', ffmpeg_encoder]
    ffmpeg_cq = '29'
    ffmpeg_tune = '5'# -(default hq)  hq:1 uhq:5         
    ffmpeg_preset = 'p7' #Lower is faster
    ffmpeg_multipass = '2' #disabled:0, qres:1, fullres:2
    ffmpeg_config = ffmpeg_config + ['-cq', ffmpeg_cq,
                                     '-rc', 'vbr',
                                     '-preset', ffmpeg_preset,
                                     '-multipass', ffmpeg_multipass,
                                     '-tune', ffmpeg_tune,
                                     '-rc-lookahead', '4']

# Initialize logging
logging.basicConfig(level=logging.DEBUG, filename=DebugDir+'logging.txt', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

#Global Variable
last_depth_flag = True
last_depth = None
last_frame = None
print_once = False
def inference_worker (in_queue, out_queue, DEVICE):
    redirrect_stdout(DebugDir + f"inference_worker_{os.getpid()}.txt")
    print_flush (encoder, encoder_path, DEVICE)
    print_flush ("Torch model loading into device name: ", torch.cuda.get_device_name(DEVICE))
    model = load_model(encoder, encoder_path, DEVICE)
    print_flush ("Model loaded, trying to infer an image...")
    model.infer_image (np.zeros((1080, 1920, 3), dtype = np.uint8))
    torch.cuda.empty_cache()
    print_flush ("Model loaded")
    while True:
        task = in_queue.get()
        if task is None:
            break
        img = task[0]
        with torch.no_grad(), autocast(device_type=DEVICE.type, dtype=torch.float16):
            result = model.infer_image(img)
        out_queue.put(result)
def left_side_sbs(raw_img, inference_queue, result_queue):
    #Reuse old depth if frame is not much different shenanigan.
    global last_depth_flag
    global last_frame
    global last_depth
    global print_once
    #Used to be  (np.sum (cv2.absdiff (cv2.stackBlur(raw_img, (5, 5)), cv2.stackBlur(last_frame, (3, 3)))) < 6000000)
    if (last_frame is not None) and (np.sum (cv2.absdiff (cv2.stackBlur(raw_img, (3, 3)), cv2.stackBlur(last_frame, (3, 3)))) < 2000000) and (last_depth_flag == True):
        depth = last_depth
    else:     
        last_frame = raw_img.copy()
        inference_queue.put((raw_img,)) #Khong can stackblur raw_img vi img cung bi resize ve 518 default cua DepthAnything
        depth = result_queue.get()
        if (last_depth_flag == False):
            depth = depth*0.6 + last_depth*0.4
            last_depth = depth.copy()
            #last_depth_flag = True
        else:
            last_depth_flag = False
            last_depth = depth.copy()

    #Normal image fill
    result_blank_mask = np.zeros(raw_img.shape[:2], dtype=bool)
    result_img = np.zeros(raw_img.shape, dtype=raw_img.dtype)
    shaded_result_img = np.zeros(raw_img.shape, dtype=raw_img.dtype)
    #Edge blurring DOES NOT CONSUME CPU TIME MUCH.
    edge_fill_positive = np.zeros(raw_img.shape[:2], dtype=bool)
    edge_fill_negative = np.zeros(raw_img.shape[:2], dtype=bool)
    limit_step = math.ceil(depth.max())
    offset_range = [offset_bg * raw_img.shape[0], offset_fg * raw_img.shape[0] * limit_step/14.0]
    max_depth = limit_step
    kernel_size = round(0.0047 * raw_img.shape[0]) #0.0047 is the OG, then 0.0036 works fine, 0.0024 is a bit too low.
    kernel_expand = np.ones ((max(kernel_size, 1),  max(kernel_size, 1)))
    
    cutoff_list = []
    for i in range (int(offset_range[0]), -1, 2): #Cai nay giup save 20% time, 550 phut xuong 450 phut
        cutoff_list.append ((i - offset_range[0]) / (0.00001+offset_range[1] - offset_range[0]) * (0.00001+limit_step))
    #0
    cutoff_list.append ((0 - offset_range[0]) / (0.00001+offset_range[1] - offset_range[0]) * (0.00001+limit_step))
    for i in range (1, int(offset_range[1]), 2):
        cutoff_list.append ((i - offset_range[0]) / (0.00001+offset_range[1] - offset_range[0]) * (0.00001+limit_step))
    cutoff_list.append (limit_step)
    cutoff_list = sorted (cutoff_list)
    cutoff_list [0] = 0
    step_list = [cutoff_list[i+1]-cutoff_list[i] for i in range(len(cutoff_list)-1)]

    if print_once == False:
        for i, curr_step in zip(cutoff_list, step_list):
            t = (i) / (0.00001+limit_step) * (0.00001+offset_range[1] - offset_range[0]) + offset_range[0]
            print_flush (t,' ',round(t),' ',int(t))

    """color_list = [
        (255,   0,   0), #red
        (240,  15,   0),
        (225,  30,   0),
        (210,  45,   0),
        (195,  60,   0),
        (180,  75,   0),
        (165,  90,   0),
        (150, 105,   0),
        (135, 120,   0),
        (120, 135,   0),
        (105, 150,   0),
        ( 90, 165,   0),
        ( 75, 180,   0),
        ( 60, 195,   0),
        ( 45, 210,   0),
        ( 30, 225,   0),
        (  0, 255,   0), #green
    ]
    color_t = 0"""
    for i, curr_step in zip(cutoff_list, step_list):
        bin_mask = (((i - 0.075 * curr_step) <= depth) & (depth < i + 1.05 * curr_step)).astype(bool)

        rows, cols = np.nonzero(bin_mask)
        #masked_img = np.zeros_like(raw_img)
        #masked_img[rows, cols, :] = raw_img[rows, cols, :]
        #offset_x = int((i+0.5*curr_step) / (0.00001+limit_step - curr_step) * (0.00001+offset_range[1] - offset_range[0]) + offset_range[0])
        offset_x = round((i) / (0.00001+limit_step) * (0.00001+offset_range[1] - offset_range[0]) + offset_range[0])
        #print ("offset_x: ", offset_x)
        if offset_x != 0:
            bin_mask = np.roll(bin_mask, shift=offset_x, axis=1).astype (np.bool)
        masked_mask = bin_mask
        #This one is for edge filling for "close-by" objects
        if (offset_x > 0): #From >0
           edge_fill_positive |= cv2.filter2D(masked_mask.astype(np.int16), -1, np.array([[-2, 1, 1]], dtype=np.int16))>0
        if (offset_x < 0):
           edge_fill_negative |= cv2.filter2D(masked_mask.astype(np.int16), -1, np.array([[1, 1, -2]], dtype=np.int16))>0
        
        #As fast as you can get here
        rows, cols = np.nonzero(bin_mask)
        result_img[rows, cols, :] = np.roll(raw_img, shift=offset_x, axis=1)[rows, cols, :]# masked_img [rows, cols, :]

        #Color injecting:
        """shaded = np.roll(raw_img, shift=offset_x, axis=1)[rows, cols, :].astype(np.float32) * 0.65 + 0.35 * np.array(color_list[color_t%len(color_list)])
        color_t = color_t+1
        shaded = np.clip(shaded, 0, 255).astype(np.uint8)
        shaded_result_img[rows, cols, :] = shaded"""

        result_blank_mask |= masked_mask

    result_zero_mask = ~result_blank_mask  # inverted boolean mask where no pixel was filled
    result_zero_mask = cv2.morphologyEx(result_zero_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_expand) #BETTER
    #Fill result_img with blurred value from zero_mask
    result_zero_mask = result_zero_mask.astype(bool)
    """color = (0,0,0) #black
    shaded_result_img[result_zero_mask] = (cv2.stackBlur
                                            (raw_img
                                            ,(limit_step*2 + 3, round(limit_step/8)*2 + 1)
                                            )
                                            )[result_zero_mask]
    shaded_result_img[result_zero_mask] = ((cv2.stackBlur
                                            (shaded_result_img
                                            ,(limit_step, round(limit_step/8)*2 + 1)
                                            )
                                           ) * 0.4 + 0.6 * np.array(color))[result_zero_mask]"""
    result_img[result_zero_mask] = (cv2.stackBlur
                                            (raw_img
                                            ,(limit_step*2 + 3, round(limit_step/8)*2 + 1)
                                    ))[result_zero_mask]
    result_img[result_zero_mask] = (cv2.stackBlur
                                            (result_img
                                            ,(limit_step + (limit_step%2==0), round(limit_step/8)*2 + 1)
                                    ))[result_zero_mask]
    #Is this line necessary?
    #edge_fill_positive = cv2.dilate(edge_fill_positive.view(np.uint8), np.ones((1, 3)), iterations = 1).astype(bool)

    """color = (0,255,255) #cyan
    shaded_result_img[edge_fill_positive] = (cv2.stackBlur (result_img, (kernel_size+(kernel_size%2==0), kernel_size+(kernel_size%2==0))) * 0.5 + np.array(color) * 0.5)[edge_fill_positive]
    color = (255,255,0) #yellow
    shaded_result_img[edge_fill_negative] = (cv2.stackBlur (result_img, (kernel_size+(kernel_size%2==0), kernel_size+(kernel_size%2==0))) * 0.5 + np.array(color) * 0.5)[edge_fill_negative]
    """
    result_img[edge_fill_positive] = cv2.stackBlur (result_img, (kernel_size+(kernel_size%2==0), kernel_size+(kernel_size%2==0)))[edge_fill_positive]
    result_img[edge_fill_negative] = cv2.stackBlur (result_img, (kernel_size+(kernel_size%2==0), kernel_size+(kernel_size%2==0)))[edge_fill_negative]

    result_img[:, 0:round(offset_x/3), :] = raw_img[:, 0:round(offset_x/3), :]
    print_once = True
    if last_frame is not None:
        #return cv2.hconcat([shaded_result_img, result_img])
        return cv2.hconcat([result_img, last_frame])
    else:
        #return cv2.hconcat([shaded_result_img, result_img])
        return cv2.hconcat([result_img, raw_img])

def nibba_woka(begin, end, inference_queue, result_queue, max_frame_count = Max_Frame_Count, file_path = VideoDir):
    #Silence all output of child process
    redirrect_stdout(DebugDir + str (begin//(end-begin))+'_' + str(begin)+'.txt')
    cap, fps, video_length, width, height = load_and_set_video (file_path, begin)
    print_flush ("Worker begin from ",begin," to ",end)    
    print_flush ("video length: ", get_length (file_path), "frame count: ",video_length, ", begin and end: ",begin, end)
    #random_sleep ((0, int(30 * (begin/video_length))), "Sleeping some more to diverge them process")
    begin_time = time.time()
    global ffmpeg_config
    try:
        ffmpeg_proc = None
        last_i = begin
        FrameList = []
        #profiler = LineProfiler()
        #profiler.add_function(left_side_sbs)
        #profiler.add_function(get_cutoff)
        #profiler.enable()
        for i in range (begin, min (end, video_length)):
            _, raw_img = cap.read()
            if (raw_img is not None):
                #left_side_sbs(raw_img[:,:,[2,1,0]])
                FrameList.append(left_side_sbs(raw_img[:,:,[2,1,0]], inference_queue, result_queue))
                #profiler.dump_stats("{}_{}".format(left_side_sbs.__name__, str(begin)))
                #dump_line_profile_to_csv (profiler, DebugDir+"Profile.csv")
            else:
                FrameList.append(np.zeros((height, 2*width, 3), dtype = np.uint8))
                gc.collect()
                print_flush ("Frame read error at i = ",i,", adding black frame to compensate.")
            if (len (FrameList) == max_frame_count) or (i == (min (end-1, video_length-1))):
                total_step = (min (end, video_length) - begin)
                step_taken = i - begin
                time_taken = (time.time() - begin_time)
                time_total = time_taken / step_taken * total_step
                time_left = time_taken / step_taken * (total_step - step_taken)
                print_flush ("Writing file ", i, "with length (in frames): ", len(FrameList))
                print_flush ("",str(int(step_taken / total_step * 10000)/100), "%, Time elapsed (minutes):", time_taken/60.0,", ETA:", time_left/60.0,", Estimated Total Time (minutes):", time_total/60.0)
                gc.collect()
                compare_time = time.time()
                
                if (ffmpeg_proc is not None): #If the previous ffmpeg subprocess did not finished, wait till it's done before generating new one
                    ffmpeg_proc.wait()
                ffmpeg_proc = subprocess.Popen(ffmpeg_config + [f'{SubClipDir}{last_i}_{i}.mp4'],
                                           stdin=subprocess.PIPE)
                for frame in FrameList:
                    ffmpeg_proc.stdin.write(frame.tobytes())
                ffmpeg_proc.stdin.close()
                print_flush ("ffmpeg pipe write time: ",time.time()-compare_time)
                #Random sanity check to protect my sanity
                if (i%1000 == 0):
                    vid_length = get_length(SubClipDir+str(last_i)+"_"+str(i)+".mp4")
                    print ("FrameList length: ",len(FrameList),", Actual length: ", vid_length)
                    #assert (len(FrameList) == temp_cap)
                last_i = i+1
                FrameList = []
                gc.collect()
                #print_flush ("Worker ending")
                #profiler.disable()
                #profiler.print_stats()
                #return 0              
        print_flush ("Worker ending")
        try:
            #out_file.close()
            #ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
        except:
            pass
        return 0
        #return [i, left_side_sbs(raw_img)]
    except Exception as e:
        print_flush(f"[ERROR] Segment {begin} failed: {e}")
        print_flush(f"[ERROR] {begin} failed at frame {i}")
        print_flush(traceback.format_exc())
        redirrect_stdout(DebugDir + 'ERROR.txt')
        print_flush (str(begin)+'.txt error')
        print_flush(f"[ERROR] Segment {begin} failed: {e}")
        print_flush(f"[ERROR] {begin} failed at frame {i}")
        print_flush(traceback.format_exc())
        raise e
        return None
if __name__ == "__main__":
    #set_start_method("spawn", force=True) #no-op on Windows, uncomment this on Linux
    
    step = math.ceil((min(end_frame, video_length) - start_frame)/Num_Workers)
    frame_indices = range(start_frame, min(end_frame, video_length), step)
    
    inference_queue_list = [Queue() for i in range (0, Num_GPU_Workers)]
    result_queue_list = [Queue() for i in range (0, Num_GPU_Workers)]
    inference_workers = [Process(target=inference_worker, args=(inference_queue_list[i], result_queue_list[i], torch.device('cuda', (i%num_gpu))))
                             for i in range (0, Num_GPU_Workers)]
    
    for j in range (0, Num_GPU_Workers):
        inference_workers[j].start()
        random_sleep ((0,0.5), "staggered model load")
    """nibba_woka (1785, 1795, inference_queue_list[0], result_queue_list[0])
    for inference_queue in inference_queue_list:
        inference_queue.put(None)        
    for w in inference_workers:
        w.join()
    """
    workers = []
    for idx, i in enumerate(frame_indices):
        inference_workers_idx = idx % Num_GPU_Workers
        worker = Process(target = nibba_woka, args = (i, i + step, inference_queue_list[inference_workers_idx], result_queue_list[inference_workers_idx]))
        print ("idx: ", idx,", GPU Worker: ",inference_workers_idx, ", worker: ", worker)
        random_sleep ((0, 1), "Staggered worker start") #There is a reason for this, Don't mess with it
        worker.start()
        workers.append(worker)
    for w in workers:
        w.join()
    for inference_queue in inference_queue_list:
        inference_queue.put(None)        
    for w in inference_workers:
        w.join()
    from Combine_Clips import combine_clips
    combine_clips (SubClipDir, VideoDir, OutputDir, just_combine = 0)
    
