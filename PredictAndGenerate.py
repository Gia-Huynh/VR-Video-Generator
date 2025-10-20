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
from torchvision.transforms.v2.functional import gaussian_blur_image, gaussian_blur
from depth_anything_v2.dpt import DepthAnythingV2
#Video Export Libraries
import subprocess
import multiprocessing as mp
from multiprocessing import shared_memory
from multiprocessing import Process, Queue, set_start_method
import queue
#Support Functions
from SupportFunction import get_length, get_cutoff, get_ffmpeg_config, load_model, load_and_set_video, redirrect_stdout, print_flush, dump_line_profile_to_csv, remove_all_file, random_sleep
import SupportFunction as SpF
import argparse

#import pystuck
#pystuck.run_server()
#pystuck.run_server(port=)
#pystuck_port = 10000

parser = argparse.ArgumentParser()
parser.add_argument('--DebugDir',   type=str,
                                    default="Debug/")
parser.add_argument('--SubClipDir', type=str,
                                    default="D:/TEMP/JAV Subclip/")
parser.add_argument('--VideoDir',   type=str,
                                    default="Videos/Input/Original/Maria Nagai.mp4")
parser.add_argument('--OutputDir',  type=str,
                                    default="DeleteThis.mkv")
parser.add_argument('--encoder',    type=str,
                                    default='vits')
parser.add_argument('--encoder_path',type=str,
                                    default='depth_anything_v2/checkpoints/depth_anything_v2_vits.pth')
parser.add_argument('--offset_fg',  type=float,
                                    default=0.02) #0.0125 chưa đủ đô
parser.add_argument('--offset_bg',  type=float,
                                    default=-0.05) #-0.0225 chưa đủ đô
parser.add_argument('--Num_Workers',type=int,
                                    default=8)
parser.add_argument('--num_gpu',    type=int,
                                    default=1)
parser.add_argument('--Num_GPU_Workers',type=int,
                                    default=2)
parser.add_argument('--Max_Frame_Count',type=int,
                                    default=15)
parser.add_argument('--start_frame',type=int,
                                    default=0)
parser.add_argument('--end_frame',  type=int,
                                    default=99999999999999) #82800 + 450 #9999999999999, 27000 is 15 minutes, 9000 5 minutes
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

if (offset_bg * offset_fg > 0):
    if (offset_bg >= 0):
        offset_bg = offset_bg * (-1)
    else:
        offset_fg = offset_fg * (-1)        

#if smaller than video length, will be clipped off
video_length = 0
gpu_memory_cache_cleaning_percentage = 0.5
cache_clearing_frequency = 10
ffmpeg_device = 'cpu'
video_length, ffmpeg_config = SpF.get_ffmpeg_config(VideoDir, ffmpeg_device)
# Initialize logging
#logging.basicConfig(level=logging.DEBUG, filename=DebugDir+'logging.txt', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

def inference_worker_backup (in_queue_list, out_queue_list, notify_queue_list, DEVICE):
    redirrect_stdout(DebugDir + f"inference_worker_{os.getpid()}.txt")
    #print_flush ("Pystuck port: ",os.getpid())
    print_flush (encoder, encoder_path, DEVICE)
    print_flush ("Torch model loading into device name: ", torch.cuda.get_device_name(DEVICE))
    model = load_model(encoder, encoder_path, DEVICE)
    print_flush ("Model loaded, trying to infer an image...")
    temp_result = model.infer_image_gpu (np.zeros((1080, 1920, 3), dtype = np.uint8))
    torch.cuda.empty_cache()
    print_flush ("Model loaded")
    device = torch.device('cuda:0')
    
    #Initialize Result List
    result_list = [[temp_result.detach().clone(), temp_result.detach().clone()] for i in range (len(out_queue_list))]
    del temp_result
    torch.cuda.empty_cache()
    
    first_run = True
    while True:
            
        queue_idx = notify_queue_list.get()
        if queue_idx is None:
            break
        task = in_queue_list[queue_idx[0]].get()
        if task is None:
            break
        img = task[0]
        del result_list[queue_idx[0]][0]
        with torch.no_grad(), autocast(device_type=DEVICE.type, dtype=torch.float16):
            result_list[queue_idx[0]].append(model.infer_image_gpu(img))
        out_queue_list[queue_idx[0]].put(result_list[queue_idx[0]][1])

        #Gpu memory spike issue, seems to be fixed, delete this part later
        """free, total = torch.cuda.mem_get_info(device)
        mem_used_MB = (total - free) / 1024 ** 2
        if (total-free)/total > gpu_memory_cache_cleaning_percentage:
            print ("High gpu memory detected, cleaning", free," out of ", total,"free ratio: ", free/total)
            torch.cuda.empty_cache() #Memory issue"""
            #gc.collect() #Memory issue
        #torch.cuda.empty_cache() #Memory issue
def inference_worker (in_queue_list, out_queue_list, notify_queue_list, DEVICE):
    """redirrect_stdout(DebugDir + f"inference_worker_{os.getpid()}.txt")
    print_flush (encoder, encoder_path, DEVICE)
    print_flush ("Torch model loading into device name: ", torch.cuda.get_device_name(DEVICE))
    model = load_model(encoder, encoder_path, DEVICE)
    print_flush ("Model loaded, trying to infer an image...")
    model.infer_image (np.zeros((1080, 1920, 3), dtype = np.uint8))
    torch.cuda.empty_cache()
    print_flush ("Model loaded")"""
    
    #profiler = LineProfiler()
    #profiler.add_function(inference_worker_backup)
    #profiler.enable()
    #torch.cuda.set_per_process_memory_fraction(0.7, device=0)
    #pystuck.run_server(port = os.getpid())
    inference_worker_backup(in_queue_list, out_queue_list, notify_queue_list, DEVICE)
    #dump_line_profile_to_csv(profiler, filename=DebugDir + f"inference_worker_{os.getpid()}_Profiler.csv")
    """while True:
        queue_idx = notify_queue_list.get()
        if queue_idx is None:
            break
        task = in_queue_list[queue_idx[0]].get()
        if task is None:
            break
        img = task[0]
        with torch.no_grad(), autocast(device_type=DEVICE.type, dtype=torch.float16):
            result = model.infer_image_gpu(img)
        out_queue_list[queue_idx[0]].put(result)"""
class LeftSBSProcessor:
    def __init__(self, gpu_notify_queue, gpu_notify_worker_idx, debug_config = [None]):
        self.debug_filePrefix = debug_config [0]
        #gpu_woker_notification
        self.gpu_notify_queue = gpu_notify_queue
        self.gpu_notify_worker_idx = gpu_notify_worker_idx

        #Depth spatial smoothing related variable
        self.depth_list = []
        self.depth_dampening_count = 2
        self.depth_dampening_ratio = 0.4
        self.depth_dampening_initial_value = 0.3
        t = self.depth_dampening_initial_value
        t_accumulate_sum = 0
        for i in range (self.depth_dampening_count):
            t_accumulate_sum = t_accumulate_sum + t
            t = t*self.depth_dampening_ratio
        self.depth_dampening_original_ratio = 1 - t_accumulate_sum

        #Persistent variable
        self.sigmaboi = 3
        self.last_depth_flag = True
        self.last_depth = None
        self.last_frame = None
        self.print_once = False

        #huge memory tensor, keeping it to reuse later.
        self.gay_tensor_1 = None
        self.gay_tensor_2 = None
        
        self.non_zero_indices = None
    def get_cutoff (self, depth):
        limit_step = math.ceil(depth.max())
        offset_range = [offset_bg * depth.shape[0] * limit_step/15.0, #Divided by 14 to normalize
                        offset_fg * depth.shape[0] * limit_step/13.0]
        cutoff_list = []
        for i in range (int(offset_range[0]), 0, 1): #(...,0, 2) Cai nay giup save 20% time, 550 phut xuong 450 phut
            cutoff_list.append ((i - offset_range[0]) / (0.00001+offset_range[1] - offset_range[0]) * (0.00001+limit_step))
        cutoff_list.append ((0 - offset_range[0]) / (0.00001+offset_range[1] - offset_range[0]) * (0.00001+limit_step))
        for i in range (1, int(offset_range[1]), 1): #What's in front of you should not be coarse
            cutoff_list.append ((i - offset_range[0]) / (0.00001+offset_range[1] - offset_range[0]) * (0.00001+limit_step))
        cutoff_list.append (limit_step)
        cutoff_list = sorted (cutoff_list)
        cutoff_list [0] = 0
        step_list = [cutoff_list[i+1]-cutoff_list[i] for i in range(len(cutoff_list)-1)]

        offset_x_list = []
        for depth_threshold, curr_step in zip(cutoff_list, step_list):
            offset_x = round((depth_threshold) / (0.00001+limit_step) * (0.00001+offset_range[1] - offset_range[0]) + offset_range[0])
            offset_x_list.append (offset_x)
        #if self.print_once == False: #Debug printing
            #print ("limit_step: ", limit_step)
            #print ("offset_range: ", offset_range)
            #for depth_threshold, curr_step in zip(cutoff_list, step_list):
            #    t = (depth_threshold) / (0.00001+limit_step) * (0.00001+offset_range[1] - offset_range[0]) + offset_range[0]
            #    print_flush (depth_threshold,' ',curr_step,' ',t,' ',round(t),' ',int(t))
            #self.print_once = True
        if random.randint(0, 30) == 15:
            print ("limit_step (or a.k.a math.ceil(depth.max())): ", limit_step)
            print ("offset_range: ", offset_range)
            for depth_threshold, curr_step in zip(cutoff_list, step_list):
                    t = (depth_threshold) / (0.00001+limit_step) * (0.00001+offset_range[1] - offset_range[0]) + offset_range[0]
                    print_flush (depth_threshold,'_',round(t),', ', end='')
            print_flush ("")

        return cutoff_list, offset_range, step_list, limit_step, offset_x_list
    def add_frame (self, raw_img, job_queue, result_queue):
        self.gpu_notify_queue.put((self.gpu_notify_worker_idx,))
        job_queue.put((raw_img,)) #Khong can stackblur raw_img vi img cung bi resize ve 518 default cua DepthAnything
        
    def get_depth (self, raw_img, job_queue, result_queue):

        #if (self.last_frame is None):
        #    self.gpu_notify_queue.put((self.gpu_notify_worker_idx,))
        #    job_queue.put((raw_img,))
        
        #add raw_img to gpu queue and get result
        #self.last_frame = raw_img.copy()
        self.last_frame = 1
        #self.gpu_notify_queue.put((self.gpu_notify_worker_idx,))
        #job_queue.put((raw_img,)) #Khong can stackblur raw_img vi img cung bi resize ve 518 default cua DepthAnything
        depth = result_queue.get()     # I CHEAT HERE

        #DEBUG
        #depth/=(depth.max().item())
        #depth*=15
        #depth = torch.nan_to_num(depth, nan=0)
        
        #Intialization
        #while (len(self.depth_list)<self.depth_dampening_count):
        #    self.depth_list.append(depth.clone())
        #t = self.depth_dampening_initial_value
        #depth = depth*self.depth_dampening_original_ratio
        #depth*=self.depth_dampening_original_ratio
        #if (len(self.depth_list) > self.depth_dampening_count):
        #    print_flush ("EXCUSE WE WTF IS THIS DEPTH LIST LENGTH?", len(self.depth_list))
        #for i in range (len(self.depth_list)-1, -1, -1):
        #    depth =  depth + self.depth_list[i]*t
        #    t = t*self.depth_dampening_ratio
        #self.depth_list.pop(0)
        #del self.depth_list[0]
        #self.depth_list.append (depth)
        #self.depth_list[-1] = depth.clone() #DEBUG ONLY
        return depth
    def gpu_roll (self, img, shift, axis):  #Same input signature as np.roll to ensure replacability
        #Note for future Gia: Not much speed improvement I'm afraid, but there was... or so I believe
        return torch.roll (torch.from_numpy(img).to(torch.device('cuda'), non_blocking=True), shifts=shift, dims=axis).cpu().numpy()
    
    def gpu_roll_with_offset (self, img_gpu, offset_list, axis):
        result = []
        for i in offset_list:
            result.append (torch.roll (img_gpu, shifts=i, dims=axis)[None, :]) #.unsqueeze(0)
        result = torch.cat(result, dim=0)  # stack on GPU
        return result #return result.cpu().numpy(), img_gpu
    
    def left_side_sbs(self, raw_img, job_queue, result_queue):

        img_gpu = torch.from_numpy(raw_img).to(torch.device('cuda'), non_blocking=True)
        depth = self.get_depth(raw_img, job_queue, result_queue) #Bottleneck here

        #Initialization
        edge_fill = torch.zeros(raw_img.shape[:2], dtype=torch.bool, device = torch.device('cuda'))
        result_img = torch.zeros(raw_img.shape, dtype=torch.uint8, device = torch.device('cuda'))
        debug_result_img = torch.zeros(raw_img.shape, dtype=torch.uint8, device = torch.device('cuda'))
        result_blank_mask = torch.zeros(raw_img.shape[:2], dtype=torch.bool, device = torch.device('cuda'))
        #Kernel init
        kernel_size = round(0.0036 * raw_img.shape[0]) #0.0047 is the OG, then 0.0036 works fine, 0.0024 is a bit too low.
        
        #Get cut-off and related matrix.
        cutoff_list, offset_range, step_list, limit_step, offset_x_list = self.get_cutoff(depth)
        offset_img = self.gpu_roll_with_offset(img_gpu, offset_list = offset_x_list, axis=1)

        debug_state = 0
        for idx, depth_threshold, curr_step in zip(range(len(cutoff_list)), cutoff_list, step_list):
            bin_mask = (((depth_threshold - 0.05 * curr_step) <= depth) & (depth < depth_threshold + 1.05 * curr_step)).to(torch.bool)
            
            offset_x = offset_x_list[idx]
            if offset_x != 0:
                bin_mask = torch.roll(bin_mask, shifts=offset_x, dims=1).to(torch.bool)
            
            """if offset_x > 0:
                if self.gay_tensor_1 is None:
                    self.gay_tensor_1 =torch.tensor([[[[-1, 1]]]], dtype=torch.float32, device=bin_mask.device)
                kernel = self.gay_tensor_1
                edge_fill |= (
                    torch.nn.functional.conv2d(bin_mask.float().unsqueeze(0).unsqueeze(0), kernel, padding='same')[0, 0] > 0
                )

            elif offset_x < 0:
                if self.gay_tensor_2 is None:
                    self.gay_tensor_2 =torch.tensor([[[[1, -1]]]], dtype=torch.float32, device=bin_mask.device)
                kernel = self.gay_tensor_2
                edge_fill |= (
                    torch.nn.functional.conv2d(bin_mask.float().unsqueeze(0).unsqueeze(0), kernel, padding='same')[0, 0] > 0
                )"""
                
            #As fast as you can get here, literally, other torch.nonzero option tested
            self.rows, self.cols = torch.nonzero(bin_mask,as_tuple=True)
            result_img[self.rows, self.cols, :] = offset_img[idx][self.rows, self.cols, :]
            #Debug
            if debug_state == 0:
                debug_result_img[self.rows, self.cols, :] = offset_img[idx][self.rows, self.cols, :]# masked_img [rows, cols, :]
                debug_state = 1
            else:
                red_img = torch.tensor([150, 0, 0], dtype=torch.uint8, device=torch.device('cuda')).expand(offset_img[idx].size()).clone()
                debug_result_img[self.rows, self.cols, :] = red_img[self.rows, self.cols, :]# masked_img [rows, cols, :]
                debug_state = 0

            result_blank_mask |= bin_mask
        #cv2.imwrite(self.debug_filePrefix + "_0_InitialResultImg.png", torch.clone(result_img).cpu().numpy()[:,:,[2,1,0]])
        result_zero_mask = (~result_blank_mask).to (torch.float32)
        kernel_expand = torch.ones ((max(kernel_size, 1),  max(kernel_size, 1)), dtype=torch.float32, device=bin_mask.device).unsqueeze(0).unsqueeze(0)
        """result_zero_mask =  torch.nn.functional.conv2d(
                                                        (torch.nn.functional.conv2d(
                                                            result_zero_mask.unsqueeze(0).unsqueeze(0),
                                                            kernel_expand,
                                                            padding='same')[0, 0] > 0
                                                        ).to (torch.float32).unsqueeze(0).unsqueeze(0),
                                                        kernel_expand,
                                                        padding='same'
                                                    )[0, 0] >= max(kernel_size, 1) * max(kernel_size, 1)"""
        
        #Fill result_img with blurred value from zero_mask
        result_zero_mask = result_zero_mask.to(torch.bool)
        """result_img[result_zero_mask] = (gaussian_blur(img_gpu.permute (2,0,1),
                                                      (kernel_size*2 + 3, kernel_size*2 + 1),
                                                      #(limit_step*2 + 3, limit_step*2 + 1),
                                                      sigma = self.sigmaboi
                                        )).permute (1,2,0)[result_zero_mask] #Help fill black gap"""
        result_img[result_zero_mask] = offset_img[int(len(offset_img)*3/5)][result_zero_mask]        
        result_img[result_zero_mask] = (gaussian_blur(result_img.permute (2,0,1),
                                                      (kernel_size*2 + 3, kernel_size*2 + 1),
                                                      #(limit_step + (limit_step%2==0), round(limit_step/4)*2 + 1),
                                                      sigma = self.sigmaboi
                                        )).permute (1,2,0)[result_zero_mask] #Help smoothen out the transition
        #Is this line necessary?
        # (Deleted, cv2.dilate edge_fill_positive)
        #t = gaussian_blur_image(result_img.permute (2,0,1), (kernel_size+(kernel_size%2==0), kernel_size+(kernel_size%2==0)),sigma = self.sigmaboi).permute (1,2,0)
        #result_img[edge_fill] = t [edge_fill]                    
        result_img[:, 0:round(offset_x/3), :] = img_gpu[:, 0:round(offset_x/3), :]
        result = torch.concat([result_img, img_gpu], dim=1) #.detach().cpu().numpy()
        debug_result = torch.concat([debug_result_img, (depth[:,:,None]*15).expand(-1, -1, 3).to(debug_result_img.dtype) ], dim=1)  #.detach().cpu().numpy() #
        debug_result = torch.concat([result, debug_result], dim=0).detach().cpu().numpy()
        return debug_result#cv2.hconcat([result_img, raw_img])

def nibba_woka(begin, end, job_queue, result_queue, gpu_notify_list, max_frame_count = Max_Frame_Count, file_path = VideoDir, repair_mode = repair_mode):
    #Silence all output of child process
    redirrect_stdout(DebugDir + str (begin//(end-begin))+'_' + str(begin)+'.txt')
    #pystuck.run_server(port = os.getpid())
    #print_flush ("Pystuck port: ",os.getpid())
    gpu_notify_queue = gpu_notify_list[0]
    gpu_notify_worker_idx = gpu_notify_list[1]
    cap, fps, video_length, width, height = load_and_set_video (file_path, begin)
    total_step = (min (end, video_length) - begin)
    sbsObj = LeftSBSProcessor(gpu_notify_queue, gpu_notify_worker_idx, [(DebugDir + str (begin//(end-begin))+'_' + str(begin))])
    
    print_flush ("Worker begin from ",begin," to ",end, "\n","video length: ", get_length (file_path), "frame count: ",video_length, ", begin and end: ",begin, end)  
    begin_time = time.time()

    global ffmpeg_config
    ffmpeg_proc = None
    last_i = begin
    FrameList = []
    profiler = None
    last_img = None
    #profiler = LineProfiler()
    #profiler.add_function(sbsObj.get_depth)
    #profiler.add_function(sbsObj.left_side_sbs)
    #profiler.enable()        
    try:
        for i in range (begin, min (end, video_length)):
            _, raw_img = cap.read()
            if (i == begin): #Initialization
                sbsObj.add_frame(raw_img[:,:,[2,1,0]], job_queue, result_queue)
                last_img = raw_img
            else:  #Normal run
                if (raw_img is not None): 
                    sbsObj.add_frame(raw_img[:,:,[2,1,0]], job_queue, result_queue) #Add frame after append is worse
                    FrameList.append(sbsObj.left_side_sbs(last_img[:,:,[2,1,0]], job_queue, result_queue))
                    last_img = raw_img
                else: #Frame read error?
                    FrameList.append(np.zeros((height, 2*width, 3), dtype = np.uint8))
                    print_flush ("Frame read error at i = ",i,", adding black frame to compensate.")
            

            if (i == min (end, video_length)-1): #Final Run
                FrameList.append(sbsObj.left_side_sbs(raw_img[:,:,[2,1,0]], job_queue, result_queue))
            torch.cuda.empty_cache() #Memory issue
            #gc.collect() #Memory issue
            if (len (FrameList) == max_frame_count) or (i == (min (end-1, video_length-1))):
                step_taken = i - begin
                print_flush ("Time elapsed (minutes):", ((time.time() - begin_time))/60.0,", ETA:", ((time.time() - begin_time) / step_taken * (total_step - step_taken))/60.0,", Estimated Total Time (minutes):", ((time.time() - begin_time) / step_taken * total_step)/60.0)
                print_flush (str(int(step_taken / total_step * 10000)/100), " %")            

                if (ffmpeg_proc is not None): #If the previous ffmpeg subprocess did not finished, wait till it's done before generating new one
                    t = time.time()
                    ffmpeg_proc.wait()
                    if (time.time() - t) > 0.0:
                        print ("Wait for ffmpeg video write for",time.time() - t,"seconds")
                ffmpeg_proc = subprocess.Popen(ffmpeg_config + [f'{SubClipDir}{last_i}_{i}.mp4'],
                                           stdin=subprocess.PIPE) #ffmpeg write time should only takes about 0.5 seconds, tiny amount
                for frame in FrameList:
                    ffmpeg_proc.stdin.write(frame.tobytes())
                ffmpeg_proc.stdin.close()
                
                last_i = i+1
                FrameList = []
                #torch.cuda.synchronize()
                gc.collect()
        print_flush ("Worker ending")
        if profiler is not None:
            dump_line_profile_to_csv(profiler, filename=DebugDir + str (begin//(end-begin))+'_' + str(begin)+'_Profiler.csv')
        try:
            ffmpeg_proc.wait()
        except:
            pass
        return 0
    except Exception as e:
        print_flush(f"[ERROR] Segment {begin} failed: {e}")
        print_flush(f"[ERROR] {begin} failed at frame {i}")
        print_flush(traceback.format_exc())
        redirrect_stdout(DebugDir + 'ERROR.txt')
        print_flush (str(begin)+'.txt error')
        print_flush(f"[ERROR] Segment {begin} failed: {e}")
        print_flush(f"[ERROR] {begin} failed at frame {i}")
        print_flush(traceback.format_exc())
        #raise e
        dump_line_profile_to_csv(profiler, filename=DebugDir + str (begin//(end-begin))+'_' + str(begin)+'_Profiler.csv')
        random_sleep ((9,10), "Worker error, sleep then exit")
        return 0
def we_ballin ():
    step = math.ceil((min(end_frame, video_length) - start_frame)/Num_Workers)
    frame_indices = range(start_frame, min(end_frame, video_length), step)
    #m = mp.Manager()
    notify_queue_list = [Queue() for gpu_worker_number in range (0, Num_GPU_Workers)] #To notify gpu worker to check for new queue item
    job_queue_list = [Queue() for i in range (0, Num_Workers)]
    result_queue_list = [Queue() for i in range (0, Num_Workers)]

    #Create and assign input/output queue to gpu_worker
    inference_job_queue_list = [[] for i in range (Num_GPU_Workers)]
    inference_result_queue_list = [[] for i in range (Num_GPU_Workers)]
    for i in range (0, Num_Workers):
        inference_job_queue_list[i%Num_GPU_Workers].append (job_queue_list[i])
        inference_result_queue_list[i%Num_GPU_Workers].append (result_queue_list[i])

    #Initialize Gpu workers and start them
    inference_workers = [Process(target=inference_worker, args=(inference_job_queue_list[i], inference_result_queue_list[i], notify_queue_list[i], torch.device('cuda', (i%num_gpu))))
                             for i in range (0, Num_GPU_Workers)]    
    for j in range (0, Num_GPU_Workers):
        inference_workers[j].start()
        random_sleep ((1,3), "staggered model load")

    #Initialize SBS workers and start them
    workers = []
    for idx, i in enumerate(frame_indices):
        gpu_worker_number = idx % Num_GPU_Workers
        within_gpu_worker_inference_idx = int(idx/Num_GPU_Workers)
        worker = Process(target = nibba_woka, args = (i, min(end_frame, i + step), job_queue_list[idx], result_queue_list[idx], [notify_queue_list[gpu_worker_number], within_gpu_worker_inference_idx]))
        print ("idx: ", idx,", assigned to gpu_worker: ", gpu_worker_number, ", worker: ", worker)
        random_sleep ((0.5, 1.5), "Staggered worker start") #There is a reason for this, don't delete this
        worker.start()
        workers.append(worker)

    #Wait for them worker to finish
    for w in workers:
        w.join()
    for job_queue in job_queue_list:
        job_queue.put(None) #Signal inference worker to stop.
    for notify_queue in notify_queue_list:
        notify_queue.put(None) #Signal inference worker to stop.
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
    
