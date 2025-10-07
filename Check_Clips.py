import os, time
import subprocess
from SupportFunction import get_length
#import fixxingFunc
import glob
dirpath = "D:/TEMP/JAV Subclip/"
#dirpath = "SubclipOutput/GPT/"  # Folder containing sub-clips
import cv2

# Step 1: Get all sub-clip filenames and sort them numerically
#files = [os.path.normpath(i) for i in glob.glob(dirpath + "*.mp4")]
files = [i for i in os.listdir (dirpath) if i[-1]=='4']
numeric_files = sorted(files, key=lambda x: int(os.path.splitext(x)[0])) 
#print (numeric_files)
print ("We are checking")

for i in range (0, len(numeric_files)-1):
    cap  = cv2.VideoCapture(dirpath+numeric_files[i])
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    a = int(numeric_files[i].split('_')[1].split(".")[0])
    b = int(numeric_files[i+1].split('_')[0])
    if (i%100 == 0):
        print (dirpath+numeric_files[i], video_length)
    if video_length!=a+1-int(numeric_files[i].split('_')[0]):
        print ("Length Issue on file", numeric_files[i],', a = ',a, #' int(nume...:', int(numeric_files[i].split('_')[0]),
               ',\n   Expected length:',a+1-int(numeric_files[i].split('_')[0]),', True Length: ',video_length)
        subprocess.run(["python", "PredictAndGenerate.py", "--SubClipDir", "D:/TEMP/FixxingSubclip/",
                        "--Num_Workers", "2", "--start_frame", numeric_files[i].split('_')[0], "--end_frame", str(a+1), "--repair_mode", "1"])
        os.remove(dirpath+numeric_files[i])
    if ((a != b) and (a!=b-1)):
        print ("Issue in continuity: ",a, b,', Difference: ', b-a ,", File name: ",numeric_files[i]," ",numeric_files[i+1])
        subprocess.run(["python", "PredictAndGenerate.py", "--SubClipDir", "D:/TEMP/FixxingSubclip/",
                        "--Num_Workers", "2", "--start_frame", str(a+1), "--end_frame", str(b), "--repair_mode", "1"])
        
#Code mình đang bị 3 lỗi:
# Lỗi 1: Nếu sub-process bị đói ram, nó sẽ ngưng chạy và không tạo file
# Lỗi 2: Đôi lúc file tạo ra bị thiếu frame? (Nhưng chỉ bị với file mình chạy tay, không biết tự động thì có bị không
# Lỗi 3: Không tạo được file cuối cùng (Có lẽ là đã sửa xong?)
#fixxingFunc.nibba_woka (
