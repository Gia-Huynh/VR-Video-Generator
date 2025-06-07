import numpy as np
import torch
import cv2
import random, time, sys
from depth_anything_v2.dpt import DepthAnythingV2

def random_sleep (sleep_length, message):
    sleep_length = random.uniform(sleep_length[0], sleep_length[1])
    print (message + " : " + str(sleep_length) +  " seconds.")
    try:
        sys.stdout.flush()
    except:
        pass
    time.sleep(sleep_length)
    
last_cutoff = None
def get_cutoff (depth_img, last_depth):
    #Getting cutoff locations from depth_img    
    #DO NOT FUCKING MODIFY THIS FUNCTION, WE HAVE TESTED IT A LOT IN THE PAST AND FUCKING LEAVE IT BE
    step_width = 0.125
    bin_range = np.arange(0, depth_img.max(), step_width)
    a, bin_count = np.unique(np.digitize (depth_img, bin_range), return_counts = True)
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

    Result_Cutoff_List = sorted (Result_Cutoff_List)
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
    model = DepthAnythingV2(**model_configs[encoder])
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
