#Profiling
from line_profiler import LineProfiler
#Basic system import
import os, sys, time, random, traceback, gc, math
#import logging #Don't remove this
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.amp import autocast #For FP16
from depth_anything_v2.dpt import DepthAnythingV2
#Video Export Libraries
import subprocess
import multiprocessing as mp
from multiprocessing import shared_memory
from multiprocessing import Process, Queue, set_start_method
#Support Functions
from SupportFunction import get_length, get_cutoff, get_ffmpeg_config, load_model, load_and_set_video, redirrect_stdout, print_flush, dump_line_profile_to_csv, remove_all_file, random_sleep
import SupportFunction as SpF
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--DebugDir',   type=str,
                                    default="Debug/")
parser.add_argument('--SubClipDir', type=str,
                                    default="D:/TEMP/JAV Subclip/")
parser.add_argument('--VideoDir',   type=str,
                                    default="Videos/Drive.2011.1080p.BluRay.DDP5.1.x265.10bit-GalaxyRG265.mkv")
parser.add_argument('--OutputDir',  type=str,
                                    default="SBS Margin Call.mp4")
parser.add_argument('--encoder',    type=str,
                                    default='vitb')
parser.add_argument('--encoder_path',type=str,
                                    default='depth_anything_v2/checkpoints/depth_anything_v2_vitb.pth')
parser.add_argument('--offset_fg',  type=float,
                                    default=0.02) #0.0125 chưa đủ đô
parser.add_argument('--offset_bg',  type=float,
                                    default=-0.02) #-0.0225 chưa đủ đô
parser.add_argument('--Num_Workers',type=int,
                                    default=24)
parser.add_argument('--num_gpu',    type=int,
                                    default=1)
parser.add_argument('--Num_GPU_Workers',type=int,
                                    default=3)
parser.add_argument('--Max_Frame_Count',type=int,
                                    default=30)
parser.add_argument('--start_frame',type=int,
                                    default=0)
parser.add_argument('--end_frame',  type=int,
                                    default=9999999999999) #82800 + 450 #9999999999999, 27000 is 15 minutes, 9000 5 minutes
parser.add_argument('--repair_mode',type=int,
                                    default=0) #Repair mode 0: Default option, clear debug/subclip dir, rerun everything and combine
                                                #Repair mode 1: Just run videos, clear only debug dir, rerun everything, no combine
                                                #Repair mode 2: Just combine videos with audio
                                                #Repair mode 3: Combine video only, outputing temp.mp4

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
repair_mode = args.repair_mode

if (offset_bg >= 0) and (offset_bg * offset_fg > 0):
    offset_bg = offset_bg * (-1)

#if smaller than video length, will be clipped off
video_length = 0
    
#Yes, order of inputs is important: ffmpeg [global options] [input options] -i input [output options] output.
#Options in [input options] are applied to the input. Options in [output options] are applied to the output.
ffmpeg_device = 'nvidia'
video_length, ffmpeg_config = SpF.get_ffmpeg_config(VideoDir, ffmpeg_device)
# Initialize logging
#logging.basicConfig(level=logging.DEBUG, filename=DebugDir+'logging.txt', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

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
        
class LeftSBSProcessor:
    def __init__(self):
        self.depth_dampening_count = 4 #fourth one is only 0.04125
        self.depth_dampening_ratio = 0.5
        self.depth_dampening_initial_value = 0.33
        self.depth_list = []
        
        t = self.depth_dampening_initial_value
        t_accumulate_sum = 0
        for i in range (self.depth_dampening_count):
            t_accumulate_sum = t_accumulate_sum + t
            t = t*self.depth_dampening_ratio
        self.depth_dampening_original_ratio = 1 - t_accumulate_sum
            
        self.last_depth_flag = True
        self.last_depth = None
        self.last_frame = None
        self.print_once = False
    def get_cutoff (self, depth):
        limit_step = math.ceil(depth.max())
        offset_range = [offset_bg * depth.shape[0], offset_fg * depth.shape[0] * limit_step/14.0]
        cutoff_list = []
        for i in range (int(offset_range[0]), -1, 2): #Cai nay giup save 20% time, 550 phut xuong 450 phut
            cutoff_list.append ((i - offset_range[0]) / (0.00001+offset_range[1] - offset_range[0]) * (0.00001+limit_step))
        cutoff_list.append ((0 - offset_range[0]) / (0.00001+offset_range[1] - offset_range[0]) * (0.00001+limit_step))
        for i in range (1, int(offset_range[1]), 1): #What's in front of you should not be coarse
            cutoff_list.append ((i - offset_range[0]) / (0.00001+offset_range[1] - offset_range[0]) * (0.00001+limit_step))
        cutoff_list.append (limit_step)
        cutoff_list = sorted (cutoff_list)
        cutoff_list [0] = 0
        step_list = [cutoff_list[i+1]-cutoff_list[i] for i in range(len(cutoff_list)-1)]

        offset_x_list = []
        for i, curr_step in zip(cutoff_list, step_list):
            offset_x = round((i) / (0.00001+limit_step) * (0.00001+offset_range[1] - offset_range[0]) + offset_range[0])
            offset_x_list.append (offset_x)
        return cutoff_list, offset_range, step_list, limit_step, offset_x_list
    def get_depth (self, raw_img, inference_queue, result_queue):
        if (self.last_frame is not None) and (np.sum (cv2.absdiff (cv2.stackBlur(raw_img, (3, 3)), cv2.stackBlur(self.last_frame, (3, 3)))) < 2000000) and (self.last_depth_flag == True):
            depth = self.depth_list[-1]
            return depth

        #add raw_img to gpu queue and get result
        self.last_frame = raw_img.copy()
        inference_queue.put((raw_img,)) #Khong can stackblur raw_img vi img cung bi resize ve 518 default cua DepthAnything
        depth = result_queue.get()

        #Intialization
        while (len(self.depth_list)<self.depth_dampening_count):
            self.depth_list.append(depth.copy())

        t = self.depth_dampening_initial_value
        depth = depth*self.depth_dampening_original_ratio
        for i in range (len(self.depth_list)-1, -1, -1):
            depth =  depth + self.depth_list[i]*t
            t = t*self.depth_dampening_ratio
        self.depth_list.pop(0)
        self.depth_list.append (depth.copy())
        return depth
    def gpu_roll (self, img, shift, axis):  #Same input signature as np.roll to ensure replacability
        #Note for future Gia: Not much speed improvement I'm afraid, but there was... or so I believe
        #print ("shift roll: ",shift)
        #return np.roll(img, shift=shift, axis=axis).astype (bool)
        return torch.roll (torch.from_numpy(img).to(torch.device('cuda')), shifts=shift, dims=axis).cpu().numpy()
    
    def gpu_roll_with_offset (self, img, offset_list, axis):
        result = []
        img_gpu = torch.from_numpy(img).to(torch.device('cuda'), non_blocking=True)
        for i in offset_list:
            result.append (torch.roll (img_gpu, shifts=i, dims=axis)[None, :]) #.unsqueeze(0)
        result = torch.cat(result, dim=0)  # stack on GPU
        return result.cpu().numpy()
    def left_side_sbs(self, raw_img, inference_queue, result_queue):
        #Reuse old depth if frame is not much different shenanigan.
        depth = self.get_depth(raw_img, inference_queue, result_queue)

        #Initialization
        #Normal image fill
        result_blank_mask = np.zeros(raw_img.shape[:2], dtype=bool)
        result_img = np.zeros(raw_img.shape, dtype=raw_img.dtype)
        shaded_result_img = np.zeros(raw_img.shape, dtype=raw_img.dtype)
        #Edge blurring DOES NOT CONSUME CPU TIME MUCH.
        edge_fill_positive = np.zeros(raw_img.shape[:2], dtype=bool)
        edge_fill_negative = np.zeros(raw_img.shape[:2], dtype=bool)
        #Kernel init
        kernel_size = round(0.0047 * raw_img.shape[0]) #0.0047 is the OG, then 0.0036 works fine, 0.0024 is a bit too low.
        

        #Get cut-off and related matrix.
        cutoff_list, offset_range, step_list, limit_step, offset_x_list = self.get_cutoff(depth)
        
        #if self.print_once == False: #Debug printing
        #    for i, curr_step in zip(cutoff_list, step_list):
        #        t = (i) / (0.00001+limit_step) * (0.00001+offset_range[1] - offset_range[0]) + offset_range[0]
        #        print_flush (t,' ',round(t),' ',int(t))
                
        offset_img = self.gpu_roll_with_offset(raw_img, offset_list = offset_x_list, axis=1)
        for idx, i, curr_step in zip(range(len(cutoff_list)), cutoff_list, step_list):
            bin_mask = (((i - 0.05 * curr_step) <= depth) & (depth < i + 1.05 * curr_step)).astype(bool)
            
            offset_x = offset_x_list[idx]
            if offset_x != 0:
                bin_mask = np.roll(bin_mask, shift=offset_x, axis=1).astype (bool)
            #masked_mask = bin_mask
            
            #This one is for edge expanding for "close-by" objects
            if (offset_x > 0): #From >0
               edge_fill_positive |= cv2.filter2D(bin_mask.astype(np.int16), -1, np.array([[-2, 1, 1]], dtype=np.int16))>0
            if (offset_x < 0):
               edge_fill_negative |= cv2.filter2D(bin_mask.astype(np.int16), -1, np.array([[1, 1, -2]], dtype=np.int16))>0
            
            #As fast as you can get here
            rows, cols = np.nonzero(bin_mask)
            result_img[rows, cols, :] = offset_img[idx][rows, cols, :]# masked_img [rows, cols, :]

            result_blank_mask |= bin_mask

        result_zero_mask = ~result_blank_mask  # inverted boolean mask where no pixel was filled
        kernel_expand = np.ones ((max(kernel_size, 1),  max(kernel_size, 1)))
        result_zero_mask = cv2.morphologyEx(result_zero_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_expand) #BETTER
        #Fill result_img with blurred value from zero_mask
        result_zero_mask = result_zero_mask.astype(bool)

        result_img[result_zero_mask] = (cv2.stackBlur
                                                (raw_img
                                                #,(limit_step*2 + 3, round(limit_step/8)*2 + 1)
                                                ,(limit_step*2 + 3, limit_step*2 + 1)
                                        ))[result_zero_mask] #Help fill black gap
        result_img[result_zero_mask] = (cv2.stackBlur
                                                (result_img
                                                ,(limit_step + (limit_step%2==0), round(limit_step/8)*2 + 1)
                                        ))[result_zero_mask] #Help smoothen out the transition
        #Is this line necessary?
        #edge_fill_positive = cv2.dilate(edge_fill_positive.view(np.uint8), np.ones((1, 3)), iterations = 1).astype(bool)


        result_img[edge_fill_positive] = cv2.stackBlur (result_img, (kernel_size+(kernel_size%2==0), kernel_size+(kernel_size%2==0)))[edge_fill_positive]
        result_img[edge_fill_negative] = cv2.stackBlur (result_img, (kernel_size+(kernel_size%2==0), kernel_size+(kernel_size%2==0)))[edge_fill_negative]

        result_img[:, 0:round(offset_x/3), :] = raw_img[:, 0:round(offset_x/3), :]
        self.print_once = True
        return cv2.hconcat([result_img, raw_img])

def nibba_woka(begin, end, inference_queue, result_queue, max_frame_count = Max_Frame_Count, file_path = VideoDir, repair_mode = repair_mode):
    #Silence all output of child process
    redirrect_stdout(DebugDir + str (begin//(end-begin))+'_' + str(begin)+'.txt')
    cap, fps, video_length, width, height = load_and_set_video (file_path, begin)
    total_step = (min (end, video_length) - begin)
    sbsObj = LeftSBSProcessor()
    
    print_flush ("Worker begin from ",begin," to ",end, "\n","video length: ", get_length (file_path), "frame count: ",video_length, ", begin and end: ",begin, end)  
    begin_time = time.time()

    global ffmpeg_config
    ffmpeg_proc = None
    last_i = begin
    FrameList = []
    try:
        for i in range (begin, min (end, video_length)):
            _, raw_img = cap.read()
            if (raw_img is not None):
                FrameList.append(sbsObj.left_side_sbs(raw_img[:,:,[2,1,0]], inference_queue, result_queue))
            else:
                FrameList.append(np.zeros((height, 2*width, 3), dtype = np.uint8))
                print_flush ("Frame read error at i = ",i,", adding black frame to compensate.")
            if (len (FrameList) == max_frame_count) or (i == (min (end-1, video_length-1))):
                step_taken = i - begin
                print_flush ("Writing file ", i, "with length (in frames): ", len(FrameList))
                print_flush ("Time elapsed (minutes):", ((time.time() - begin_time))/60.0,", ETA:", ((time.time() - begin_time) / step_taken * (total_step - step_taken))/60.0,", Estimated Total Time (minutes):", ((time.time() - begin_time) / step_taken * total_step)/60.0)
                print_flush (str(int(step_taken / total_step * 10000)/100), " %")            

                if (ffmpeg_proc is not None): #If the previous ffmpeg subprocess did not finished, wait till it's done before generating new one
                    ffmpeg_proc.wait()
                ffmpeg_proc = subprocess.Popen(ffmpeg_config + [f'{SubClipDir}{last_i}_{i}.mp4'],
                                           stdin=subprocess.PIPE) #ffmpeg write time should only takes about 0.5 seconds, tiny amount
                for frame in FrameList:
                    ffmpeg_proc.stdin.write(frame.tobytes())
                ffmpeg_proc.stdin.close()
                
                last_i = i+1
                FrameList = []
                gc.collect()
        print_flush ("Worker ending")
        try:
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
def we_ballin ():
    step = math.ceil((min(end_frame, video_length) - start_frame)/Num_Workers)
    frame_indices = range(start_frame, min(end_frame, video_length), step)
    
    inference_queue_list = [Queue() for i in range (0, Num_GPU_Workers)]
    result_queue_list = [Queue() for i in range (0, Num_GPU_Workers)]
    inference_workers = [Process(target=inference_worker, args=(inference_queue_list[i], result_queue_list[i], torch.device('cuda', (i%num_gpu))))
                             for i in range (0, Num_GPU_Workers)]
    
    for j in range (0, Num_GPU_Workers):
        inference_workers[j].start()
        random_sleep ((0,0.5), "staggered model load")

    workers = []
    for idx, i in enumerate(frame_indices):
        inference_workers_idx = idx % Num_GPU_Workers
        worker = Process(target = nibba_woka, args = (i, min(end_frame, i + step), inference_queue_list[inference_workers_idx], result_queue_list[inference_workers_idx]))
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
    
if __name__ == "__main__":
    remove_all_file (DebugDir)
    if (repair_mode == 0):
        remove_all_file (SubClipDir)
        
if __name__ == "__main__":
    #set_start_method("spawn", force=True) #no-op on Windows, uncomment this on Linux
    if (repair_mode in [0, 1]):
        we_ballin()
        
    if (repair_mode in [0, 2]):
        from Combine_Clips import combine_clips
        combine_clips (SubClipDir, VideoDir, OutputDir, just_combine = 0)
        
    if (repair_mode in [3]):
        from Combine_Clips import combine_clips
        combine_clips (SubClipDir, VideoDir, OutputDir, just_combine = 1)
    
