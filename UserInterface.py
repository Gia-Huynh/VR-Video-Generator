import dearpygui.dearpygui as dpg
from file_dialog.fdialog import FileDialog
import subprocess
import os, psutil
OG_cwd = os.getcwd()
state = {
    "proc" : None
}
args = {
    "VideoDir": "Videos/Input/Maria Nagai [Trimmed].mp4", #"Original video here"
    "OutputDirectory": "./Videos/Output", #"Output video folder here"
    "OutputName": "KillMe.mkv", #"Output video name here"
    "OutputDir": "Will Be Auto Calculated", #"Output video name here"
    "SubClipDir": "D:/TEMP/JAV Subclip/",
    #"VideoDir": "Path to input video",
    #"OutputDirectory": "Enter output video folder path",
    #"OutputName": "Output file name [auto end with .mkv]",
    #"OutputDir": "Will Be Auto Calculated.mkv",
    #"SubClipDir": "./Subclip",
    "DebugDir": "./Debug/",
    "encoder": "vitb",
    "encoder_path": "depth_anything_v2/checkpoints/depth_anything_v2_vitb.pth",
    "offset_fg": 0.025,
    "offset_bg": -0.004,
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
    cmd = "python PredictAndGenerate.py " + " ".join(
        [f'--{k} "{v}"' for k, v in args.items() if k not in ["OutputDirectory", "OutputName"]]
    )
    dpg.set_value("preview_text", cmd)
def auto_update_filename (update_target):
    print ("FUCK")
    OG_Filename = os.path.splitext(os.path.basename(args["VideoDir"]))[0]
    args["OutputName"] = OG_Filename + "Test.mkv"
    dpg.set_value(update_target,  OG_Filename + f" [SBS {args['offset_fg']} {args['offset_bg']} {args['offset_step_size']}].mkv")
    update_preview()
    
def run_script(sender, app_data):
    args["OutputDir"] = os.path.join(args["OutputDirectory"], args["OutputName"])
    print("Running with arguments:")
    for k, v in args.items():
        print(f"--{k} {v}")
    cmd = "python PredictAndGenerate.py " + " ".join(
        [f'--{k} "{v}"' for k, v in args.items() if k not in ["OutputDirectory", "OutputName"]]
    )
    print (cmd)
    proc = subprocess.Popen(cmd , cwd = OG_cwd)
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
dpg.create_viewport(title="PredictAndGenerate UI", width=1600, height=800)

# Register a BIGGER font
with dpg.font_registry():
    big_font = dpg.add_font("C:/Windows/Fonts/seguiemj.ttf", 20)  # change to another TTF if missing

def InputFileDialogueCallback(selected_files):
    dpg.set_value("VideoDir", selected_files[0])
    args["VideoDir"] = selected_files[0]
    update_preview()
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

with dpg.window(label="PredictAndGenerate", tag="main_window", width=1580, height=780):
    dpg.add_text("Configure & Run", bullet=True)
    dpg.add_separator()

    with dpg.table(header_row=False, resizable=True, policy=dpg.mvTable_SizingStretchProp, row_background=False):
        dpg.add_table_column(width_stretch=True, init_width_or_weight=0.6)  # left column 40%
        dpg.add_table_column(width_stretch=True, init_width_or_weight=0.4)  # right column 60%

        with dpg.table_row():
            # LEFT COLUMN
            with dpg.group(horizontal=False):
                dpg.add_text("Directories / Files")
                dpg.add_spacer(width=0, height=10)  # 10 pixels vertical space
                with dpg.group(horizontal=True):
                    dpg.add_text("Input Video Path")
                    #button_list.append(dpg.add_button(label="Select Video", callback=lambda: open_file_dialog("VideoDir")))
                    button_list.append(dpg.add_button(label="Select Video", callback=InputFileDialogue.show_file_dialog))
                dpg.add_input_text(tag="VideoDir", default_value=args["VideoDir"], callback=update_value, user_data="VideoDir", width=-1)

                dpg.add_spacer(width=0, height=10)  # 10 pixels vertical space
                with dpg.group(horizontal=True):
                    dpg.add_text("Output Folder")
                    #button_list.append(dpg.add_button(label="Select Output Folder", callback=lambda: open_dir_dialog("OutputDirectory")))
                    button_list.append(dpg.add_button(label="Select Output Folder", callback=OutputFolderDialogue.show_file_dialog))
                    
                dpg.add_input_text(tag="OutputDirectory", default_value=args["OutputDirectory"], callback=update_value, user_data="OutputDirectory", width=-1)
                with dpg.group(horizontal=True):
                    dpg.add_text("Output Video Name")
                    button_list.append(dpg.add_button(label="Get Auto Output Name", callback=lambda: auto_update_filename("OutputName")))
                dpg.add_input_text(tag="OutputName", default_value=args["OutputName"], callback=update_value_video, user_data="OutputName", width=-1)
                
                dpg.add_spacer(width=0, height=10)  # 10 pixels vertical space
                with dpg.group(horizontal=True):
                    dpg.add_text("Intermediate (Temporarily) subclip directory")
                    button_list.append(dpg.add_button(label="Select SubClipDir", callback=lambda: open_dir_dialog("SubClipDir")))
                dpg.add_input_text(tag="SubClipDir", default_value=args["SubClipDir"], callback=update_value, user_data="SubClipDir", width=-1)

                dpg.add_spacer(width=0, height=10)  # 10 pixels vertical space
                dpg.add_text("Model / Encoder Path")
                dpg.add_input_text(tag="encoder", default_value=args["encoder"], callback=update_value, user_data="encoder", width=-1)
                dpg.add_input_text(tag="encoder_path", default_value=args["encoder_path"], callback=update_value, user_data="encoder_path", width=-1)
                button_list.append(dpg.add_button(label="Select Encoder Path", callback=lambda: open_file_dialog("encoder_path")))

            # RIGHT COLUMN
            with dpg.group(horizontal=False):
                dpg.add_text("Video Parameter")
                dpg.add_spacer(width=0, height=10)  # 10 pixels vertical space
                dpg.add_input_float(label="offset foreground", default_value=args["offset_fg"], callback=update_value, user_data="offset_fg")
                dpg.add_input_float(label="offset background", default_value=args["offset_bg"], callback=update_value, user_data="offset_bg")
                dpg.add_input_int(label="offset step size", default_value=args["offset_step_size"], callback=update_value, user_data="offset_step_size")

                dpg.add_text("Performance Parameter")
                dpg.add_input_int(label="Workers Count", default_value=args["Num_Workers"], callback=update_value, user_data="Num_Workers")
                dpg.add_input_int(label="Gpu Count", default_value=args["num_gpu"], callback=update_value, user_data="num_gpu")
                dpg.add_input_int(label="GPU Workers Count", default_value=args["Num_GPU_Workers"], callback=update_value, user_data="Num_GPU_Workers")
                dpg.add_input_int(label="Batch Frame Count", default_value=args["Max_Frame_Count"], callback=update_value, user_data="Max_Frame_Count")
                dpg.add_text("Debug Parameter, don't touch unless you need it.")
                dpg.add_combo(label="repair_mode", items=["0 - Full", "1 - Rerun no combine", "2 - Combine - Export video", "3 - [Debug] Combine video only, temp.mp4"],
                          default_value="0 - Full",
                          callback=lambda s,a,u: update_value(s,int(a[0]),"repair_mode"), user_data="repair_mode")
                dpg.add_input_text(label="start_frame", default_value=args["start_frame"], callback=update_value, user_data="start_frame")
                dpg.add_input_text(label="end_frame", default_value=args["end_frame"], callback=update_value, user_data="end_frame")

                dpg.add_spacer(width=0, height=10)  # 10 pixels vertical space
                with dpg.group(horizontal=True):
                    dpg.add_text("Debug Directory")
                    button_list.append(dpg.add_button(label="Select DebugDir", callback=lambda: open_dir_dialog("DebugDir")))
                dpg.add_input_text(tag="DebugDir", default_value=args["DebugDir"], callback=update_value, user_data="DebugDir", width=-1)

    dpg.add_separator()
    dpg.add_text("Command Preview (No edit):")
    dpg.add_input_text(multiline=True, readonly=True, auto_select_all = True, tag="preview_text", width=-1, height=50)
    with dpg.group(horizontal=True):
        dpg.add_button(label="Run Script", callback=run_script, width=200, height=50)
        dpg.add_button(label="Stop Script", callback=stop_script, width=200, height=50)
        with dpg.group(horizontal=False):
            dpg.add_text("To view progress, open the debug folder and R E A D")
            dpg.add_progress_bar(tag="progress", default_value=0.0, width=300)

# Apply BIG font to everything
dpg.bind_font(big_font)
# --- create button theme ---
with dpg.theme() as button_theme:
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (100, 149, 237))        # normal
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (65, 105, 225))  # hover
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (25, 25, 112))    # pressed

# --- apply the theme to all buttons ---
for btn in button_list:
    dpg.bind_item_theme(btn, button_theme)
    
update_preview()
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("main_window", True)
dpg.start_dearpygui()
dpg.destroy_context()
