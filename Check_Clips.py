import os, time, argparse, glob
import subprocess
from SupportFunction import get_length
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--SubClipDir', type=str)  # Folder containing sub-clips
parser.add_argument('--repair_mode',type=int,  default=0)   #Repair mode 0: Just checking, no repair
                                                            #Repair mode 1: Check and repair
args = parser.parse_args()
SubClipDir = args.SubClipDir
repair_mode = args.repair_mode

# Step 1: Get all sub-clip filenames and sort them numerically
def Checkin (SubClipDir, repair_mode = 0):
    print ("'We are checking'")
    files = [i for i in os.listdir (SubClipDir) if i[-1]=='4']
    numeric_files = sorted(files, key=lambda x: int(os.path.splitext(x)[0])) 
    for i in range (0, len(numeric_files)-1):
        cap  = cv2.VideoCapture(SubClipDir+numeric_files[i])
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        a = int(numeric_files[i].split('_')[1].split(".")[0])
        b = int(numeric_files[i+1].split('_')[0])
        if (i%100 == 0):
            print (SubClipDir+numeric_files[i], video_length)
        if video_length!=a+1-int(numeric_files[i].split('_')[0]):
            print ("Length Issue on file", numeric_files[i],', a = ',a, #' int(nume...:', int(numeric_files[i].split('_')[0]),
                   ',\n   Expected length:',a+1-int(numeric_files[i].split('_')[0]),', True Length: ',video_length)
            if (repair_mode == 1):
                subprocess.run(["python", "PredictAndGenerate.py", "--SubClipDir", "D:/TEMP/FixxingSubclip/",
                                "--Num_Workers", "2", "--start_frame", numeric_files[i].split('_')[0], "--end_frame", str(a+1), "--repair_mode", "1"])
            os.remove(SubClipDir+numeric_files[i])
        if ((a != b) and (a!=b-1)):
            print ("Issue in continuity: ",a, b,', Difference: ', b-a ,", File name: ",numeric_files[i]," ",numeric_files[i+1])
            if (repair_mode == 1):
                subprocess.run(["python", "PredictAndGenerate.py", "--SubClipDir", "D:/TEMP/FixxingSubclip/",
                                "--Num_Workers", "2", "--start_frame", str(a+1), "--end_frame", str(b), "--repair_mode", "1"])
if __name__ == "__main__":
    Checkin (SubClipDir, repair_mode)
