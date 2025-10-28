# VR Video Generator
Create Side-by-side (SBS) video from normal video, for use in VR glasses.  
Total number of nut busted:  ![Total number of nut busted: ](https://img.shields.io/github/stars/Gia-Huynh/VR-Video-Generator)  
![Left and Right alternating](GithubResources/ezgif.com-webp-maker.webp "Demo gif with left and right eye image alternating")

## System Requirements

| 			 | Theoretical Bare Minimum   | Tested System 	| Note per components |
|:---------|:-----------------------|:-------------|:------------------|
| Gpu 		 | Nvidia 2060, 1650Ti, T400  | 2080 Ti		 	| Need half-precision (fp16) support, so Pascal and older gpu is out of the question|
| Cpu 		 | Anything 				  | Amd Ryzen 9 5950X| Used for video encoding only, so no real "bare minimum" here|
| Ram 		 | 16Gb Ram 				  | 64Gb 		 	| Check FAQ section below for Ram and Gpu Vram issue  |
| HDD		 | 5Gb per hour of 1080p video| No ssd needed   | Prefer hdd than ssd because this writes a lot  |

## Quick Install 
Download the lastest [release](https://github.com/Gia-Huynh/VR-Video-Generator/releases), then extract and run UserInterface.exe
## Running from source
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

4. Run: ```python UserInterface.py```

## Common Issue/FAQ
### Q: Issue with Ram / Out of Memory
A: Reduce the ```Batch Frame Count``` value if out of ram, increase pagefile if it takes only less than 50% ram but you still got OOM issue.
### Q: Gpu not enough Vram
A: Drop Gpu Worker Count -> Choose smaller model -> Drop Workers Count -> Increase ```offset step size``` to 2 (which may create undesirable banding effect).
### Q: Bigger offset foreground/background value
A: May increase vram usage by quite a lot, so you may need to Increase ```offset step size``` to 2.
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
Author: [Your Name]  
Source: [Repo URL]
