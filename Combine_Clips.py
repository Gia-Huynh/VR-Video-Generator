import os
import subprocess
import glob
subclip_path = "D:/TEMP/JAV Subclip"  # Folder containing sub-clips
original_path = "Videos/Shoplifter Aryana Amatista.mp4" # Full video with audio
output_path = "SBS Aryana Amatista.mp4"

def combine_clips (subclip_path, original_path, output_path, just_combine = 0):
    # Step 1: Get all sub-clip filenames and sort them numerically
    #files = os.listdir(subclip_path)
    files = [i for i in os.listdir (subclip_path) if i[-1]=='4']
    numeric_files = sorted(files, key=lambda x: int(os.path.splitext(x)[0])) 
    # Step 2: Create a list file for FFmpeg
    file_list_path = os.path.join(".", "input_list.txt")
    with open(file_list_path, "w+") as f:
        for name in numeric_files:
            fullfile = os.path.join(subclip_path, name)
            f.write(f"file '{fullfile}'\n")
    # Step 3: Concatenate video files
    concat_cmd = [
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", file_list_path, "-c", "copy", "temp_video.mp4"
    ]
    subprocess.run(concat_cmd, check=True)
    if (just_combine == 1):
        os.remove(file_list_path)
        return 0
    # Step 4: Extract original audio
    #audio_path = "original_audio.aac"
    #subprocess.run(["ffmpeg", "-y", "-i", original_path,
    #                "-q:a", "0", "-map", "a", audio_path], check=True)
    audio_path = "original_audio.mka"
    subprocess.run([
                    "ffmpeg", "-y", "-i", original_path,
                    "-map", "0:a", "-c:a", "aac", "-b:a", "192k",
                    audio_path
                    ], check=True)

    # Step 5: Merge video with extracted audio
    #final_cmd = [
    #    "ffmpeg", "-i", "temp_video.mp4", "-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", output_path
    #]
    #subprocess.run(final_cmd, check=True)
    # Step 5: Merge video with extracted audio (all tracks)
    final_cmd = [
        "ffmpeg", "-i", "temp_video.mp4", "-i", audio_path,
        "-map", "0:v", "-map", "1:a",  # map video + all audio streams
        "-c:v", "copy", "-c:a", "copy",
        output_path
    ]
    subprocess.run(final_cmd, check=True)
    # Step 6: Cleanup
    os.remove("temp_video.mp4")
    os.remove(audio_path)
    os.remove(file_list_path)
    print(f"Final video saved as {output_path}")
if __name__ == "__main__":
    combine_clips (subclip_path, original_path, output_path, just_combine = 0)
    import cv2
    original_cap = cv2.VideoCapture (original_path)
    original_frame_count = original_cap.get (cv2.CAP_PROP_FRAME_COUNT)
    output_cap = cv2.VideoCapture (output_path)
    output_frame_count = output_cap.get (cv2.CAP_PROP_FRAME_COUNT)
    print ("These frames count should be the same")
    print ("Original: ", original_frame_count,' output: ', output_frame_count,', difference: ', abs(output_frame_count - original_frame_count))
