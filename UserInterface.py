import dearpygui.dearpygui as dpg
from file_dialog.fdialog import FileDialog
import subprocess
import os,  sys, psutil, webbrowser

OG_cwd = os.getcwd()
state = {
    "proc" : None
}
if getattr(sys, "frozen", False):  # running in PyInstaller bundle
    base_path = sys._MEIPASS
    sys.path.append(os.path.join(base_path, "_internal"))
    IsFrozen = True
else: #Running from .py file
    base_path = os.path.dirname(__file__)
    IsFrozen = False
encoder_path = {
                    "vits": os.path.join(base_path, "depth_anything_v2", "checkpoints", "depth_anything_v2_vits.pth"),
                    "vitb": os.path.join(base_path, "depth_anything_v2", "checkpoints", "depth_anything_v2_vitb.pth"),
                    "vitl": os.path.join(base_path, "depth_anything_v2", "checkpoints", "depth_anything_v2_vitl.pth"),
                }
Skipped_Param = ["OutputDirectory", "OutputName", "encoder_selection"]
"""
    "VideoDir": "Videos/Input/Maria Nagai [Trimmed].mp4", #"Original video here"
    "OutputDirectory": "./Videos/Output", #"Output video folder here"
    "OutputName": "KillMe.mkv", #"Output video name here"
    "OutputDir": "Will Be Auto Calculated", #"Output video name here"
    "SubClipDir": "D:/TEMP/Subclip/",
"""
args = {
    "VideoDir": "Path to input video",
    "OutputDirectory": "Enter output video folder path",
    "OutputName": "Output file name [auto end with .mkv]",
    "OutputDir": "Will Be Auto Calculated.mkv",
    "SubClipDir": "./Subclip/",
    "DebugDir": "./Debug/",
    "encoder_selection": "vits: Small model, faster",
    "encoder": "vitb",
    "encoder_path": "./depth_anything_v2/checkpoints/depth_anything_v2_vitb.pth",
    "offset_fg": 0.025,
    "offset_bg": -0.01,
    "offset_step_size": 1,
    "Num_Workers": 6,
    "num_gpu": 1,
    "Num_GPU_Workers": 2,
    "Max_Frame_Count": 30,
    "start_frame": 0,
    "end_frame": 999999999999999,
    "repair_mode": 0
}

# --- CALLBACKS ---
def select_dir_callback(sender, app_data, user_data):
    args[user_data] = app_data['file_path_name']
    dpg.set_value(user_data, app_data['file_path_name'])
    update_preview()

def select_file_callback(sender, app_data, user_data):
    args[user_data] = app_data['file_path_name']
    dpg.set_value(user_data, app_data['file_path_name'])
    update_preview()

def open_dir_dialog(user_data):
    dpg.show_item("dir_dialog")
    dpg.set_item_user_data("dir_dialog", user_data)

def open_file_dialog(user_data):
    dpg.show_item("file_dialog")
    dpg.set_item_user_data("file_dialog", user_data)

def update_value(sender, app_data, user_data):
    args[user_data] = app_data
    update_preview()

def update_value_video(sender, app_data, user_data):
    # Auto-append .mkv for video/output paths
    if not (app_data.lower().endswith(".mkv") or app_data.lower().endswith(".mp4")):
        app_data += ".mkv"
        dpg.set_value(sender, app_data)  # update the input box immediately
    update_value(sender, app_data, user_data)
    
def update_preview():
    args["OutputDir"] = os.path.join(args["OutputDirectory"], args["OutputName"])
    if "vits" in args["encoder_selection"]:
        args["encoder"] = "vits"
        args["encoder_path"] = encoder_path["vits"]
    elif "vitb" in args["encoder_selection"]:
        args["encoder"] = "vitb"
        args["encoder_path"] = encoder_path["vitb"]
    elif "vitl" in args["encoder_selection"]:
        args["encoder"] = "vitl"
        args["encoder_path"] = encoder_path["vitl"] 
    else:
        print ("ERROR: Encoder not found")
        print (args["encoder_selection"])
    cmd = "python PredictAndGenerate.py " + " ".join(
        [f'--{k} "{v}"' for k, v in args.items() if k not in Skipped_Param]
    )
    dpg.set_value("preview_text", cmd)
def auto_update_filename (update_target):
    OG_Filename = os.path.splitext(os.path.basename(args["VideoDir"]))[0]
    args["OutputName"] = OG_Filename + f" [SBS {args['offset_fg']:.3f} {args['offset_bg']:.3f} {args['offset_step_size']}].mkv"
    dpg.set_value(update_target,  OG_Filename + f" [SBS {args['offset_fg']:.3f} {args['offset_bg']:.3f} {args['offset_step_size']}].mkv")
    update_preview()
    
def run_script(sender, app_data):
    update_preview()
    #print("Running with arguments:")
    ##for k, v in args.items():
    ##    print(f"--{k} {v}")
    #print (args)
    print("Command Preview: ")
    if IsFrozen: #Don't drop the white space
        cmd = "PredictAndGenerate.exe " + " ".join(
            [f'--{k} "{v}"' for k, v in args.items() if k not in Skipped_Param]
        )
    else:
        #cmd = ["python", "PredictAndGenerate.py", arg1, arg2]
        cmd = "python PredictAndGenerate.py " + " ".join(
            [f'--{k} "{v}"' for k, v in args.items() if k not in Skipped_Param]
        )
    print (cmd)
    print ("At directory: ", base_path)
    proc = subprocess.Popen(cmd , cwd = base_path)
    state["proc"] = proc

def stop_script():
    if state["proc"] and (state["proc"].poll() is None):  # Still running
        """state["proc"].terminate()  # Graceful
        try:
            state["proc"].wait(timeout=10)
        except subprocess.TimeoutExpired:
            state["proc"].kill()  # Force kill if not exiting"""
        parent = psutil.Process(state["proc"].pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
        print("Process stopped.")
    else:
        print("No process running.")
        
# --- UI START ---
dpg.create_context()
dpg.create_viewport(title="PredictAndGenerate UI", width=1400, height=800)

# Register a BIGGER font
with dpg.font_registry():
    big_font = dpg.add_font("C:/Windows/Fonts/seguiemj.ttf", 20)  # change to another TTF if missing

def InputFileDialogueCallback(selected_files):
    dpg.set_value("VideoDir", selected_files[0])
    args["VideoDir"] = selected_files[0]
    update_preview()
    auto_update_filename("OutputName")
InputFileDialogue = FileDialog(callback=InputFileDialogueCallback, tag="InputFileDialogue",
                               filter_list = [".*", ".mkv", ".mp4", ".avi", ".ts"],
                               modal=True, multi_selection=False, no_resize=False, default_path=".")

def OutputFolderDialogueCallback(selected_files):
    dpg.set_value("OutputDirectory", selected_files[0])
    args["OutputDirectory"] = selected_files[0]
    update_preview()
OutputFolderDialogue = FileDialog(callback=OutputFolderDialogueCallback, tag="OutputFolderDialogue",
                                  dirs_only = True, show_shortcuts_menu = True, modal=True,  multi_selection=False, no_resize=False, default_path=".")
#ádasd
#with dpg.window(label="hi", height=100, width=100):
#    dpg.add_button(label="InputFileDialogue", callback=InputFileDialogue.show_file_dialog)
    
# Directory dialog
with dpg.file_dialog(directory_selector=True, show=False, callback=select_dir_callback, tag="dir_dialog", width=600, height=400):
    dpg.add_file_extension(".*", color=(255, 255, 255, 255))

# File dialog
with dpg.file_dialog(directory_selector=False, show=False, callback=select_file_callback, tag="file_dialog", width=600, height=400):
    #dpg.add_file_extension("", color=(255, 255, 255, 255))
    dpg.add_file_extension(".mp4,.mkv", color=(255, 255, 255, 255))
    dpg.add_file_extension(".mov", color=(255, 255, 255, 255))
    dpg.add_file_extension(".avi", color=(255, 255, 255, 255))
    dpg.add_file_extension(".*", color=(255, 255, 255, 255))
    dpg.add_file_extension(".pth", color=(255, 255, 255, 255))

button_list = []
non_important_button_list = []
with dpg.window(label="PredictAndGenerate", tag="main_window", width=1380, height=780):
    dpg.add_text(" VR SBS Video Generator UI")
    dpg.add_separator()

    with dpg.table(header_row=False, resizable=True, policy=dpg.mvTable_SizingStretchProp, row_background=False):
        dpg.add_table_column(width_stretch=True, init_width_or_weight=0.5)  # left column 40%
        dpg.add_table_column(width_stretch=True, init_width_or_weight=0.5)  # right column 60%

        with dpg.table_row():
            # LEFT COLUMN
            with dpg.group(horizontal=False):
                dpg.add_spacer(width=0, height=10)  # 10 pixels vertical space
                
                dpg.add_text("Directories / Files")
                with dpg.group(horizontal=True):
                    dpg.add_text("Input Video Path")
                    dpg.add_spacer(width=100)
                    #button_list.append(dpg.add_button(label="Select Video", callback=lambda: open_file_dialog("VideoDir")))
                    button_list.append(dpg.add_button(label="Select Video", callback=InputFileDialogue.show_file_dialog))
                dpg.add_input_text(tag="VideoDir", default_value=args["VideoDir"], callback=update_value, user_data="VideoDir", width=-1)
                dpg.add_spacer(width=0, height=10)  # 10 pixels vertical space
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Output Folder")
                    dpg.add_spacer(width=128)
                    button_list.append(dpg.add_button(label="Select Destination Folder", callback=OutputFolderDialogue.show_file_dialog)) #lambda: open_dir_dialog("OutputDirectory")
                dpg.add_input_text(tag="OutputDirectory", default_value=args["OutputDirectory"], callback=update_value, user_data="OutputDirectory", width=-1)
                dpg.add_spacer(width=0, height=10)  # 10 pixels vertical space
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Output Video Name")
                    dpg.add_spacer(width=70)
                    button_list.append(dpg.add_button(label="Auto Generate Output Name", callback=lambda: auto_update_filename("OutputName")))
                dpg.add_input_text(tag="OutputName", default_value=args["OutputName"], callback=update_value_video, user_data="OutputName", width=-1)
                dpg.add_spacer(width=0, height=10)  # 10 pixels vertical space
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Intermediate Subclip Folder")
                    dpg.add_spacer(width=3)
                    non_important_button_list.append(dpg.add_button(label="Select 'Scratch Disk' Folder", callback=lambda: open_dir_dialog("SubClipDir")))
                dpg.add_input_text(tag="SubClipDir", default_value=args["SubClipDir"], callback=update_value, user_data="SubClipDir", width=-1)
                dpg.add_spacer(width=0, height=10)  # 10 pixels vertical space
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Model Selection")
                    dpg.add_spacer(width=108)
                    non_important_button_list.append(dpg.add_button(label="Open Model Folder", callback=lambda: os.startfile(os.path.join(base_path, "depth_anything_v2", "checkpoints"))))
                    #non_important_button_list.append(dpg.add_button(label="Select Encoder Path", callback=lambda: open_file_dialog("encoder_path")))
                dpg.add_combo(tag="encoder_selection", items=["vits: Small model, faster", "vitb: Base model, more consistent [Needs download]", "vitl: Gigantic model [Needs download]"],
                              default_value=args["encoder_selection"], callback=update_value, user_data="encoder_selection", width=-1)                
                #dpg.add_input_text(tag="encoder", default_value=args["encoder"], callback=update_value, user_data="encoder", width=-1)
                #dpg.add_input_text(tag="encoder_path", default_value=args["encoder_path"], callback=update_value, user_data="encoder_path", width=-1)
                

            # RIGHT COLUMN
            with dpg.group(horizontal=False):
                dpg.add_spacer(width=0, height=10)  # 10 pixels vertical space
                dpg.add_text("Video Parameter")
                dpg.add_input_float(label="offset foreground", default_value=args["offset_fg"], callback=update_value, user_data="offset_fg")
                dpg.add_input_float(label="offset background", default_value=args["offset_bg"], callback=update_value, user_data="offset_bg")
                dpg.add_input_int(label="offset step size", default_value=args["offset_step_size"], callback=update_value, user_data="offset_step_size")

                dpg.add_text("Performance Parameter")
                dpg.add_input_int(label="Workers Count", default_value=args["Num_Workers"], callback=update_value, user_data="Num_Workers")
                dpg.add_input_int(label="Gpu Count", default_value=args["num_gpu"], callback=update_value, user_data="num_gpu")
                dpg.add_input_int(label="GPU Workers Count", default_value=args["Num_GPU_Workers"], callback=update_value, user_data="Num_GPU_Workers")
                dpg.add_input_int(label="Batch Frame Count", default_value=args["Max_Frame_Count"], callback=update_value, user_data="Max_Frame_Count")
                dpg.add_spacer(width=0, height=10)
                dpg.add_text("Debug Parameter, don't touch unless you need it.")
                dpg.add_combo(label="repair_mode", items=["0 - Full, Default", "1 - Rerun from start_frame to end_frame, don't combine", "2 - Combine and export full video with audio", "3 - [Debug] Combine video only, temp.mp4"],
                          default_value="0 - Full",
                          callback=lambda s,a,u: update_value(s,int(a[0]),"repair_mode"), user_data="repair_mode")
                dpg.add_input_text(label="start_frame", default_value=args["start_frame"], callback=update_value, user_data="start_frame")
                dpg.add_input_text(label="end_frame", default_value=args["end_frame"], callback=update_value, user_data="end_frame")

                dpg.add_spacer(width=0, height=10)  # 10 pixels vertical space
                with dpg.group(horizontal=True):
                    dpg.add_text("Debug Directory")
                    non_important_button_list.append(dpg.add_button(label="Select Debug Folder", callback=lambda: open_dir_dialog("DebugDir")))
                    non_important_button_list.append(dpg.add_button(label="View Debug Folder", callback=lambda: os.startfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), args["DebugDir"]))))
                    
                dpg.add_input_text(tag="DebugDir", default_value=args["DebugDir"], callback=update_value, user_data="DebugDir", width=-1)

    dpg.add_separator()
    with dpg.group(horizontal=True):
        dpg.add_text("Command Preview:\n                [No edit]")
        dpg.add_input_text(multiline=True, readonly=True, auto_select_all = True, tag="preview_text", width=-1, height=50)
    with dpg.group(horizontal=True):
        with dpg.group(horizontal=False):
            with dpg.group(horizontal=True):
                green_button = dpg.add_button(label="Run Script", callback=run_script, width=200, height=50)
                red_button = dpg.add_button(label="Stop Script", callback=stop_script, width=200, height=50)
            dpg.add_button(label="Verify Integrity (Work In Progress)", callback=None, width=408, height=50)
        with dpg.group(horizontal=False):
            dpg.add_text("To check progress,\nview files in debug folder.")
            button_list.append(dpg.add_button(label="View Debug Folder", callback=lambda: os.startfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), args["DebugDir"]))))
                
        vertical_line = dpg.add_slider_float(vertical  = True, no_input = True, width = 2, height = -1, default_value=0.0, format=" ") #dearpygui have no vertical line lol
        with dpg.group(horizontal=False):
            dpg.add_text("This video is brought to you by...\nOur sponsors:")
            with dpg.group(horizontal=True):
                button_list.append(dpg.add_button(label="Gimme a job", callback=lambda:webbrowser.open("https://gia-huynh.github.io/"), width=200, height=50))
                button_list.append(dpg.add_button(label="Steam Items\n   Donation", callback=lambda:webbrowser.open("https://steamcommunity.com/tradeoffer/new/?partner=243084914&token=Wd4eyX_8"), width=200, height=50))
    dpg.add_separator()
    dpg.add_text("Attribution: FileDialog was used for the UI, Depth-Anything-V2 and their pretrain small models for the depth estimation.")
    
vertical_line_theme_parent = dpg.add_theme()
with dpg.theme_component(dpg.mvAll, parent=vertical_line_theme_parent):
    dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (128,128,128,255))
    dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (128,128,128,255))
    dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (128,128,128,255))
    dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (128,128,128,255))
dpg.bind_item_theme(vertical_line, vertical_line_theme_parent)

# Apply BIG font to everything
dpg.bind_font(big_font)

#Normal button theme
with dpg.theme() as button_theme:
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (100, 140, 220))        # normal
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (65, 105, 225))  # hover
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (25, 25, 112))    # pressed
for btn in button_list:
    dpg.bind_item_theme(btn, button_theme)

#Unimportant button theme
with dpg.theme() as non_important_button_theme:
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (80, 100, 130))        # normal
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (65, 105, 225))  # hover
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (25, 25, 112))    # pressed
for btn in non_important_button_list:
    dpg.bind_item_theme(btn, non_important_button_theme)
    
#Red + Green button theme
with dpg.theme() as green_button_theme:
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (20, 140, 20))    # normal
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (20, 200, 20))    # hover  
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 140, 0))    # pressed
dpg.bind_item_theme(green_button, green_button_theme)
with dpg.theme() as red_button_theme:
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (140, 20, 20))    # normal
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (200, 20, 20))    # hover  
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (140, 0, 0))    # pressed
dpg.bind_item_theme(red_button, red_button_theme)

update_preview()
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("main_window", True)
dpg.start_dearpygui()
dpg.destroy_context()
