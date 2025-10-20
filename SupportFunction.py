import numpy as np
import torch
import cv2
import random, time, sys, shutil, os
from depth_anything_v2.dpt import DepthAnythingV2
import subprocess
from line_profiler import LineProfiler
import os
import csv
import inspect

def dump_line_profile_to_csv(profiler, filename="line_profile_output.csv"):
    file_exists = os.path.exists(filename)
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Function", "Line #", "Line Content",
                "Hits", "Total Time (µs)", "Per Hit (µs)"
            ])
        FUCK  = profiler.get_stats()
        for (filename, first_lineno, function_name), timings in FUCK.timings.items():
            #func_name = code_obj.co_name
            try:
                #src_lines, start_line = inspect.getsourcelines(timings.code)
                #line_dict = {start_line + i: line.rstrip() for i, line in enumerate(src_lines)}
                with open(filename, "r", encoding="utf-8") as f:
                    src_lines = f.readlines()
                line_dict = {1 + i: line.rstrip() for i, line in enumerate(src_lines)}
            except Exception:
                line_dict = {}

            for line_no, hits, total_time in timings:
                line_content = line_dict.get(line_no, "")
                per_hit = total_time / hits if hits else 0
                writer.writerow([
                    function_name, line_no, line_content,
                    hits, total_time, f"{per_hit:.2f}"
                ])


def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    return float(result.stdout.decode().strip())

def remove_all_file (dir_path):
    if os.path.isdir(dir_path) and os.listdir(dir_path):  # check not empty
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                try:
                    os.unlink(file_path)
                except PermissionError:
                    pass
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            
def random_sleep (sleep_length, message = ""):
    sleep_length = random.uniform(sleep_length[0], sleep_length[1])
    print (message + " : " + str(sleep_length) +  " seconds.")
    try:
        sys.stdout.flush()
    except:
        pass
    time.sleep(sleep_length)

def redirrect_stdout (out_path):
    if True:
	#if sys.stdout is None:
        out_file = open(out_path, 'w')
        sys.stdout = out_file
        sys.stderr = out_file
        sys.stdout.flush()

def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
def write_video (file_path, frames, fps):
    #frames is a list of numpy frames
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'h264'), fps, (width, height))
    for frame in frames:
        out.write(frame[:,:,::-1])
    out.release()
last_cutoff = None
def get_cutoff (depth_img, last_depth):
    #Getting cutoff locations from depth_img    
    #DO NOT FUCKING MODIFY THIS FUNCTION, WE HAVE TESTED IT A LOT IN THE PAST AND FUCKING LEAVE IT BE
    step_width = 0.125
    bin_range = np.arange(0, depth_img.max(), step_width)
    profile_temp_var = np.digitize (depth_img, bin_range)
    a, bin_count = np.unique(profile_temp_var, return_counts = True)
    bin_count_avg = np.zeros (bin_count.shape)

    for j in range (1, len(bin_count)-1):
        bin_count_avg [j] = 0.33 * bin_count[j] + 0.33 * bin_count[j-1] + 0.33 * bin_count[j+1]
    bin_count_avg [0] = 0.5 * bin_count[0] + 0.5 * bin_count[j+1]
    bin_count_avg [len(bin_count)-1] = 0.5 * bin_count[len(bin_count)-1] + 0.5 * bin_count[len(bin_count)-2]
    bin_count = bin_count_avg
    
    bin_label = bin_range[a-1]
    
    Result_Cutoff_List = []
    Max = -1
    MaxIdx = -1
    assert (len(bin_label) == len(bin_count))
    for i in range (1, len(bin_label)-1):
        label = bin_label[i+1]
        count = bin_count[i]
        if (Max == -1):
            Max = count
            MaxIdx = i
        else:
            if (Max < count):
                Max = count
                MaxIdx = i
            else:
                if (#Ở xa, lax hơn
                    ((count *(1 - 0.0125*(Max/count)) < bin_count[i+1] * 0.96) 
                         and (count *(1 - 0.0125*(Max/count)) < bin_count[i-1] * 0.99)
                      and (i - MaxIdx >= round(0.5/step_width - 1)))
                    #Ở gần, gắt hơn
                    or ((count *(1.01 - 0.01*(Max/count)) < bin_count[i+1] * 0.93)
                         and (count *(1.01 - 0.01*(Max/count)) < bin_count[i-1] * 0.8)
                         )
                    ):
                    Result_Cutoff_List.append (label)
                    Max = count
                    MaxIdx = i

    #Result_Cutoff_List = sorted (Result_Cutoff_List)
    Result_Cutoff_List.append(depth_img.max())
    Result_Cutoff_List.insert (0, 0)
    Result_Cutoff_List = sorted (Result_Cutoff_List)
    
    for i in range (len(Result_Cutoff_List)-2, 0, -1):
        if abs(Result_Cutoff_List[i] - Result_Cutoff_List[i+1])<1:
            del Result_Cutoff_List[i]
    global last_cutoff
    if (last_cutoff is not None):
        if (np.linalg.norm(depth_img - last_depth) < 500):
            if (len(last_cutoff) >= len(Result_Cutoff_List)):
                return last_cutoff
            else:
                pass
        else:
            pass
            #print ("Norm Declined: ", np.linalg.norm(depth_img - last_depth))
    last_cutoff = Result_Cutoff_List
    return Result_Cutoff_List

def load_model(encoder, encoder_path, DEVICE):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(device = DEVICE, **model_configs[encoder])
    model.load_state_dict(torch.load(encoder_path, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    return model

def load_and_set_video (file_path, begin): #Function used only in nibba_woka
    cap  = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # get total number of frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, begin) # set frame position
    return cap, fps, video_length, width, height

#Yes, order of inputs is important: ffmpeg [global options] [input options] -i input [output options] output.
#Options in [input options] are applied to the input. Options in [output options] are applied to the output.
def get_ffmpeg_config (VideoDir, ffmpeg_device = 'cpu'):
    global video_length
    _, fps, video_length, width, height = load_and_set_video (VideoDir, 0)
    ffmpeg_config = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s',
        f'{2*width}x{2*height}', '-r', str(fps), #DEBUG DEBUG DEBUG
        '-i', '-',  # stdin
        '-an',
        '-pix_fmt', 'yuv420p'
        ]

    if (ffmpeg_device == 'cpu'):
        ffmpeg_encoder = 'libx264' #'libx264' for cpu
        ffmpeg_config += ['-c:v', ffmpeg_encoder]
        ffmpeg_crf = '21'
        ffmpeg_preset = 'veryfast' #ultrafast
        ffmpeg_config += ['-crf', ffmpeg_crf, '-preset', ffmpeg_preset]  # use crf with libx264
    elif (ffmpeg_device == 'nvidia'):
        ffmpeg_encoder = 'hevc_nvenc' #h264_nvenc for h264, hevc_nvenc for h265.
        ffmpeg_config += ['-c:v', ffmpeg_encoder]
        ffmpeg_cq = '29'
        ffmpeg_tune = '5'# -(default hq)  hq:1 uhq:5         
        ffmpeg_preset = 'p7' #Lower is faster
        ffmpeg_multipass = '0' #disabled:0, qres:1, fullres:2, original 2 but chatgpt told 0 lol
        ffmpeg_config = ffmpeg_config + ['-cq', ffmpeg_cq,
                                         '-rc', 'vbr',
                                         '-preset', ffmpeg_preset,
                                         '-multipass', ffmpeg_multipass,
                                         '-tune', ffmpeg_tune,
                                         #'-rc-lookahead', '4'
                                         ]
    return video_length, ffmpeg_config
