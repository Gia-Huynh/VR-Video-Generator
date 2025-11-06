# VR Video Generator
Create Side-by-side (SBS) video from normal video to give you an illusion of depth, for use in VR glasses.  
Available in source code and compiled exe file for ease of use, with Graphic User Interface.
  
Total number of nut busted:  ![Total number of nut busted: ](https://img.shields.io/github/stars/Gia-Huynh/VR-Video-Generator)  
  
<img src="GithubResources/ezgif.com-webp-maker.webp" alt="Demo gif with left and right eye image alternating" width="40%" height="auto;">

Photo by [Fadhil Abhimantra](https://unsplash.com/@fabhimantra?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/photos/eagle-hunters-in-traditional-attire-converse-on-horseback-vV3ovPQAlmc?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)
      
## System Requirements

| 			 | Theoretical Bare Minimum   | Tested System 	| Note per components |
|:---------|:-----------------------|:-------------|:------------------|
| Gpu 		 | Nvidia 2060, 1650Ti, T400  | 2080 Ti		 	        | Need half-precision (fp16) support, so Pascal and older gpu is out of the question|
| Cpu 		 | Potato  				            | Amd Ryzen 9 5950X   | Used for video encoding only, so no real "bare minimum" here|
| Ram 		 | 16Gb Ram 				          | 64Gb 		 	          | Check FAQ section below for Ram and Gpu Vram issue  |
| HDD		   | 5Gb per hour of 1080p video| No ssd needed       | Prefer hdd than ssd because this writes a lot and may cause ssd wear  |

## Quick Install 
Download the lastest [release](https://github.com/Gia-Huynh/VR-Video-Generator/releases), then extract with archiving programs like 7-zip or WinRar (don't use Windows default one), then run UserInterface.exe.
The released version only comes with the Small model due to licensing, the other two requires manual downloading from step 3 below and placing into the ```_internal\depth_anything_v2\checkpoints``` folder.  
## Running from source [Not fully tested]
1. Clone the repository:
```bash
git clone https://github.com/Gia-Huynh/VR-Video-Generator
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Download Depth-Anything-V2 model weights manually and place in the ```depth_anything_v2\checkpoints``` folder, from these link:

| Model | Params | Checkpoint |
|:-|-:|:-:|
| Depth-Anything-V2-Small | 24.8M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true) |
| Depth-Anything-V2-Base | 97.5M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true) |
| Depth-Anything-V2-Large | 335.3M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) |

Generally, the larger the model, the more consistent it will be frame-to-frame, though will take more vram and slightly longer to run, Large model 2.5x the total time compared to Base from my testing.  

4. Download the latest ffmpeg from [ffmpeg release link](https://github.com/BtbN/FFmpeg-Builds/releases), look out for the ```ffmpeg-master-latest-win64-gpl.zip``` file, extract ffmpeg.exe and ffprobe.exe into ```ffmpeg``` folder.
5. Run: ```python UserInterface.py```
## Parameter Changing
**Offset foreground:** Bigger positive value means popping object closer to you, default to 0.025-0.03.  
**Offset foreground:** Larger negative value means background feels further from you, default to -0.015.  
Try increasing one value, increasing both value, or increasing one while decreasing the other, just mess around  with it and have fun, other people's repo have stuff like Convergence and something else idk lol this is way simpler than that.  
**Offset step size:** Increasing from 1 to 2 will spedup conversion speed, may cause undesirable banding effect.  

## Common Issue/FAQ
### Q: Issue with Ram / Out of Memory
A: Reduce the ```Batch Frame Count``` and ```Workers Count``` value if you run out of ram, increase pagefile if it takes only less than 50% ram but you still got OOM issue.
### Q: Gpu not enough Vram
A: Drop Gpu Worker Count -> Choose smaller model -> Drop Workers Count -> Increase ```offset step size``` to 2 (which may create undesirable banding effect).  
Gpu Vram usage will increase with higher video resolution, higher "offset foreground", "offset background" values, if it skyrocketed, consider Increase ```offset step size``` to 2 first to see if it's manageable before testing anything else.  
### Q: Performance Tuning Note:
A: From my quick testing I found that increasing Worker Count larger than 6-8 does not help my running time much (maybe cpu sucks, or ffmpeg config is too hard for my cpu), for 4 workers or below you only need 1 gpu worker, for 6 workers having only 1 gpu workers compared to 2 does make a small difference in time.
### Q: Debugging Issue:
A: All of subprocess's output and error are located in the Debug folder, there's a button to open debug folder from the UI, open and check them for me, especially the ERROR.txt file if it exist.
## License
This project is CC BY 4.0.  
MIT components: FileDialog.  
Apache components: Depth-Anything-V2 source code (slightly modified), and Depth-Anything-V2-small weight.  
Depth-Anything-V2-base/large weights: CC BY-NC 4.0 (user downloads separately).

Attribution required if you use or redistribute this project:
```
Author: Gia Huynh
Author contact: giahuynh.thiet@gmail.com, https://gia-huynh.github.io/
Source: https://github.com/Gia-Huynh/VR-Video-Generator/tree/master
```
