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
from SupportFunction import get_cutoff, load_model, load_and_set_video, random_sleep, redirrect_stdout, print_flush, remove_all_file
import SupportFunction as SpF
DebugDir = "Debug/"
#SubClipDir = "SubclipOutput/"
SubClipDir = "D:/TEMP/JAV Subclip/"
# Initialize logging
logging.basicConfig(level=logging.DEBUG, filename=DebugDir+'logging.txt', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

VideoDir = "Videos/She s A Beautiful Female Teacher, The Homeroom Teacher, Advisor To Our Team Sports, And My Lover Maria Nagai (1080).mp4"
encoder = 'vitb'
encoder_path = f'depth_anything_v2/checkpoints/depth_anything_v2_vitb.pth'
offset_fg =  0.0225
offset_bg = -0.0125

ffmpeg_preset = 'fast'

Num_Workers = 10
num_gpu = 2
Num_GPU_Workers = 6 #Total
Max_Frame_Count = 25
start_frame = 0
end_frame = 27000 #9999999999999 #if larger than video length, will be clipped off

#Num_Workers = 1
#num_gpu = 1
#Num_GPU_Workers = 1 #Total
#Max_Frame_Count = 60
#start_frame = 0
#end_frame = 78574

#Global Variable
last_depth_flag = True
last_depth = None
last_frame = None
def inference_worker (in_queue, out_queue, DEVICE):
    redirrect_stdout(DebugDir + f"inference_worker_{os.getpid()}.txt")
    print_flush (encoder, encoder_path, DEVICE)
    print_flush ("Torch model loading into device name: ", torch.cuda.get_device_name(DEVICE))
    model = load_model(encoder, encoder_path, DEVICE)
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
    if (last_frame is not None) and (np.sum (cv2.absdiff (cv2.stackBlur(raw_img, (5, 5)), cv2.stackBlur(last_frame, (3, 3)))) < 6000000) and (last_depth_flag == True):
        depth = last_depth
    else:     
        last_frame = raw_img.copy()
        #last_frame = 1
        inference_queue.put((cv2.stackBlur(raw_img, (5,5)),)) #cv2.stackBlur(raw_img, (5,5))
        depth = result_queue.get()
        depth = cv2.stackBlur(depth, (3, 3)).astype(np.float32) #OG la (3, 3)
        #FUCK IT WE NORMALIZE
        depth -= depth.min()
        depth /= depth.max()
        depth *= 15
        if (last_depth_flag == False):
            depth = depth*0.6 + last_depth*0.4
            last_depth = depth.copy()
            last_depth_flag = True
        else:
            last_depth_flag = False
            last_depth = depth.copy()

    #Normal image fill
    result_blank_mask = np.zeros(raw_img.shape[:2], dtype=bool)
    result_img = np.zeros(raw_img.shape, dtype=raw_img.dtype)
    #Edge blurring
    edge_fill_blank_mask = np.zeros(raw_img.shape[:2], dtype=bool)
    #Values idk
    offset_range = [offset_bg * raw_img.shape[0], offset_fg * raw_img.shape[0]]
    limit_step = 15
    max_depth = limit_step
    kernel_size = round(0.0037 * raw_img.shape[0]) #0.0047 is the OG, then 0.0036 works fine, 0.0024 is a bit too low.
    kernel_expand = np.ones ((max(kernel_size-1, 1),  max(kernel_size-1, 1)))
    #Threshold values
    cu = sorted(get_cutoff(depth, last_depth))
    nt = [cu[i+1]-cu[i] for i in range(len(cu)-1)]
    cu.pop() #Remove last element
    
    for i, curr_step in zip(cu, nt):
        #Masked_img va Masked_mask deu bat nguon tu bin_mask,
        #Voi masked_mask chi la binary_mask luu lai pixel nao 
        bin_mask = (((i - 0.05 * curr_step) <= depth) & (depth < i + 1.05 * curr_step)).astype(bool)

        bin_mask_expanded = np.repeat(bin_mask[:, :, None], 3, axis=2)
        masked_img = np.zeros_like(raw_img)
        masked_img[bin_mask_expanded] = raw_img[bin_mask_expanded]

        masked_mask = np.zeros(raw_img.shape[:2], dtype=bool)
        masked_mask[bin_mask] = True
        
        offset_x = int((i + (0.5 * curr_step)) / (limit_step - curr_step) * (offset_range[1] - offset_range[0]) + offset_range[0])      
        #This one is for edge filling for "close-by" objects
        #if (offset_x > 0):
        #    masked_mask_border = cv2.filter2D(masked_mask.astype(np.int16), -1, np.array([[1, -2, 1]], dtype=np.int16))
            #dilated_masked_mask_border = cv2.dilate((masked_mask_border>0).astype(np.uint8), kernel_expand, iterations = 1)
            #del masked_mask_border
            #edge_fill_temp_img = np.zeros_like(raw_img)
            #edge_fill_temp_img[dilated_masked_mask_border] = raw_img[dilated_masked_mask_border]
            #edge_fill_temp_img = np.roll(edge_fill_temp_img, shift=offset_x, axis=1)  # Shift along the width (x-axis)
            #dilated_masked_mask_border = np.roll(dilated_masked_mask_border, shift=offset_x, axis=1).astype (np.bool)
            #edge_fill_img[dilated_masked_mask_border] = edge_fill_temp_img[dilated_masked_mask_border]
            #edge_fill_blank_mask |= dilated_masked_mask_border
            #This one doesn't work
            #result_blank_mask &= np.logical_not (dilated_masked_mask_border)  #A AND (NOT B)
        if offset_x != 0:
            masked_img = np.roll(masked_img, shift=offset_x, axis=1)  # Shift along the width (x-axis)
            masked_mask = np.roll(masked_mask, shift=offset_x, axis=1).astype (np.bool)
        if (offset_x > 0):
            masked_mask_border = cv2.filter2D(masked_mask.astype(np.int16), -1, np.array([[1, -2, 1]], dtype=np.int16))>0
            edge_fill_blank_mask |= masked_mask_border
        #mask_nonzero = masked_mask
        result_img[masked_mask] = masked_img[masked_mask]
        result_blank_mask |= masked_mask

    #edge_fill_temp_img[~edge_fill_blank_mask] = result_img[~edge_fill_blank_mask]
    #result_img = cv2.addWeighted(result_img, 0.5, edge_fill_temp_img, 0.5, 0)
    #return 0

    result_zero_mask = ~result_blank_mask  # inverted boolean mask where no pixel was filled
    result_zero_mask = cv2.dilate(result_zero_mask.astype(np.uint8), kernel_expand,iterations = 1)

    #Fill result_img with blurred value from zero_mask
    result_zero_mask = result_zero_mask.astype(bool)
    result_img[result_zero_mask] = (cv2.stackBlur
                                    (np.roll
                                     (raw_img, shift=round(offset_x/3), axis=1)
                                    ,(limit_step*2 + 3, round(limit_step/8)*2 + 1)
                                    )
                                   )[result_zero_mask]
    edge_fill_blank_mask = cv2.dilate(edge_fill_blank_mask.view(np.uint8), np.ones((1, 3)), iterations = 1).astype(bool)
    #cv2.imwrite ("blank_mask.png", edge_fill_blank_mask*254)
    result_img[edge_fill_blank_mask] = cv2.stackBlur (result_img, (limit_step*2 + 3, round(limit_step/8)*2 + 1))[edge_fill_blank_mask]
    result_img[:, 0:round(offset_x/3), :] = raw_img[:, 0:round(offset_x/3), :]
    return cv2.hconcat([result_img, raw_img])

def nibba_woka(begin, end, inference_queue, result_queue, max_frame_count = Max_Frame_Count, file_path = VideoDir):
    #Silence all output of child process
    redirrect_stdout(DebugDir + str (begin//(end-begin))+'_' + str(begin)+'.txt')
    cap, fps, video_length, width, height = load_and_set_video (file_path, begin)
    print_flush ("Worker begin from ",begin," to ",end)    
    print_flush ("video_length: ",video_length, ", begin and end: ",begin, end)
    #random_sleep ((0, int(30 * (begin/video_length))), "Sleeping some more to diverge them process")
    begin_time = time.time()
    try:
        #PreFilledMemory = [np.zeros((height, 2*width, 3), dtype = np.uint8) for i in range (0, max_frame_count*2)] #Fill memory for initialization
        #PreFilledMemory = []
        ffmpeg_config = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{2*width}x{height}',
            '-r', str(fps),
            '-i', '-',  # stdin
            '-an',
            '-c:v', 'libx264',
            '-preset', ffmpeg_preset,
            '-crf', '19']
        ffmpeg_proc = None
        last_i = begin
        FrameList = []
        for i in range (begin, min (end+1, video_length)):
            _, raw_img = cap.read()
            if (raw_img is not None):
                FrameList.append(left_side_sbs(raw_img[:,:,[2,1,0]], inference_queue, result_queue))
            else:
                FrameList.append(np.zeros((height, 2*width, 3), dtype = np.uint8))
                gc.collect()
                print_flush ("Frame read error at i = ",i,", adding black frame to compensate.")
            if (len (FrameList) == max_frame_count) or (i == (min (end, video_length-1))):
                total_step = (min (end, video_length) - begin)
                step_taken = i - begin
                time_taken = (time.time() - begin_time)
                time_total = time_taken / step_taken * total_step
                time_left = time_taken / step_taken * (total_step - step_taken)
                print_flush ("Writing file ", i, "with length (in frames): ", len(FrameList))
                print_flush ("",str(step_taken / total_step * 100), "%, Time elapsed (minutes):", time_taken/60.0,", ETA:", time_left/60.0,", Estimated Total Time (minutes):", time_total/60.0)
                gc.collect()
                compare_time = time.time()
                """clip = ImageSequenceClip(FrameList, fps=fps)
                clip.write_videofile(SubClipDir+str(last_i)+"_"+str(i)+".mp4", preset = 'slow', audio=False, threads = 4, logger=None)
                del clip
                print_flush ("clip.write_videofile time: ",time.time()-compare_time)
                compare_time = time.time()
                SpF.write_video(SubClipDir+str(last_i)+"_"+str(i)+".mp4", FrameList, fps)
                print_flush ("cv2 write time: ",time.time()-compare_time)
                compare_time = time.time()"""
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
                    temp_cap = cv2.VideoCapture(SubClipDir+str(last_i)+"_"+str(i)+".mp4")
                    print ("FrameList length: ",len(FrameList),", Actual length: ", temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    assert (len(FrameList) == temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    temp_cap.release()
                last_i = i+1
                FrameList = []
                gc.collect()                
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
    #inference_proc = Process(target=inference_worker, args=(inference_queue, result_queue))
    #inference_proc.start()
    _, _, video_length, _, _ = load_and_set_video (VideoDir, 0)
    step = math.ceil((min(end_frame, video_length) - start_frame)/Num_Workers)
    frame_indices = range(start_frame, min(end_frame, video_length), step)
    
    inference_queue_list = [Queue() for i in range (0, Num_GPU_Workers)]
    result_queue_list = [Queue() for i in range (0, Num_GPU_Workers)]
    inference_workers = [Process(target=inference_worker, args=(inference_queue_list[i], result_queue_list[i], torch.device('cuda', (i%num_gpu))))
                             for i in range (0, Num_GPU_Workers)]
    
    for j in range (0, Num_GPU_Workers):
        inference_workers[j].start()
        random_sleep ((1,2), "staggered model load")
    #for i in frame_indices:
    workers = []
    random_sleep ((0, 1), "Random sleep for model loading")
    for idx, i in enumerate(frame_indices):
        inference_workers_idx = idx % Num_GPU_Workers
        worker = Process(target = nibba_woka, args = (i, i + step, inference_queue_list[inference_workers_idx], result_queue_list[inference_workers_idx]))
        print ("idx: ", idx,", GPU Worker: ",inference_workers_idx, ", worker: ", worker)
        random_sleep ((0, 1), "staggered worker start")
        worker.start()
        workers.append(worker)
    for w in workers:
        w.join()
    for inference_queue in inference_queue_list:
        inference_queue.put(None)        
    for w in inference_workers:
        w.join()
    #nibba_woka (15550, 15610)
    asdasd
    from Combine_Clips import combine_clips
    combine_clips (SubClipDir, VideoDir, "output_vid.mp4")
