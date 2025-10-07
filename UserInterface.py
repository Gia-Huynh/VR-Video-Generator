from dearpygui.dearpygui import *
import subprocess

args = {
    "DebugDir": "Debug/",
    "SubClipDir": "D:/TEMP/JAV Subclip/",
    "VideoDir": "Videos/Drive.2011.1080p.BluRay.DDP5.1.x265.10bit-GalaxyRG265.mkv",
    "OutputDir": "SBS Drive.mkv",
    "encoder": "vits",
    "encoder_path": "depth_anything_v2/checkpoints/depth_anything_v2_vits.pth",
    "offset_fg": 0.04,
    "offset_bg": -0.04,
    "Num_Workers": 10,
    "num_gpu": 1,
    "Num_GPU_Workers": 2,
    "Max_Frame_Count": 15,
    "start_frame": 0,
    "end_frame": 999999999999999,
    "repair_mode": 0
}

# --- CALLBACKS ---
def select_dir_callback(sender, app_data, user_data):
    args[user_data] = app_data['file_path_name']
    set_value(user_data, app_data['file_path_name'])
    update_preview()

def select_file_callback(sender, app_data, user_data):
    args[user_data] = app_data['file_path_name']
    set_value(user_data, app_data['file_path_name'])
    update_preview()

def open_dir_dialog(user_data):
    show_item("dir_dialog")
    set_item_user_data("dir_dialog", user_data)

def open_file_dialog(user_data):
    show_item("file_dialog")
    set_item_user_data("file_dialog", user_data)

def update_value(sender, app_data, user_data):
    args[user_data] = app_data
    update_preview()

def update_preview():
    cmd = "python PredictAndGenerate.py " + " ".join(
        [f'--{k} "{v}"' for k, v in args.items()]
    )
    set_value("preview_text", cmd)

def run_script(sender, app_data):
    print("Running with arguments:")
    for k, v in args.items():
        print(f"--{k} {v}")
    cmd = "python PredictAndGenerate.py " + " ".join(
        [f'--{k} "{v}"' for k, v in args.items()]
    )
    #subprocess.run(cmd, check=True)
    proc = subprocess.Popen(cmd)
    while proc.poll() is None:
        time.sleep(0.1)
        increment_progress()
    set_value("progress", 1.0)
# --- UI START ---
create_context()
create_viewport(title="PredictAndGenerate UI", width=1600, height=800)

# Register a BIGGER font
with font_registry():
    big_font = add_font("C:/Windows/Fonts/seguiemj.ttf", 20)  # change to another TTF if missing

# Directory dialog
with file_dialog(directory_selector=True, show=False, callback=select_dir_callback, tag="dir_dialog", width=600, height=400):
    add_file_extension(".*", color=(255, 255, 255, 255))

# File dialog
with file_dialog(directory_selector=False, show=False, callback=select_file_callback, tag="file_dialog", width=600, height=400):
    #add_file_extension("", color=(255, 255, 255, 255))
    add_file_extension(".mp4,.mkv", color=(255, 255, 255, 255))
    add_file_extension(".mov", color=(255, 255, 255, 255))
    add_file_extension(".avi", color=(255, 255, 255, 255))
    add_file_extension(".*", color=(255, 255, 255, 255))
    add_file_extension(".pth", color=(255, 255, 255, 255))
    #pass

button_list = []

with window(label="PredictAndGenerate", tag="main_window", width=1580, height=780):
    add_text("Configure & Run", bullet=True)
    add_separator()

    with table(header_row=False, resizable=True, policy=mvTable_SizingStretchProp, row_background=False):
        add_table_column(width_stretch=True, init_width_or_weight=0.7)  # left column 40%
        add_table_column(width_stretch=True, init_width_or_weight=0.3)  # right column 60%

        with table_row():
            # LEFT COLUMN
            with group(horizontal=False):
                add_text("Directories / Files")
                add_spacer(width=0, height=10)  # 10 pixels vertical space
                with group(horizontal=True):
                    add_text("Input (Original) Video Path")
                    button_list.append(add_button(label="Select Video", callback=lambda: open_file_dialog("VideoDir")))
                add_input_text(tag="VideoDir", default_value=args["VideoDir"], callback=update_value, user_data="VideoDir", width=-1)

                add_spacer(width=0, height=10)  # 10 pixels vertical space
                with group(horizontal=True):
                    add_text("Output (Result) Video Directory")
                    button_list.append(add_button(label="Select Output File", callback=lambda: open_file_dialog("OutputDir")))
                add_input_text(tag="OutputDir", default_value=args["OutputDir"], callback=update_value, user_data="OutputDir", width=-1)
                
                add_spacer(width=0, height=10)  # 10 pixels vertical space
                with group(horizontal=True):
                    add_text("Intermediate (Temporarily) subclip directory")
                    button_list.append(add_button(label="Select SubClipDir", callback=lambda: open_dir_dialog("SubClipDir")))
                add_input_text(tag="SubClipDir", default_value=args["SubClipDir"], callback=update_value, user_data="SubClipDir", width=-1)

                add_spacer(width=0, height=10)  # 10 pixels vertical space
                add_text("Model / Encoder Path")
                add_input_text(tag="encoder", default_value=args["encoder"], callback=update_value, user_data="encoder", width=-1)
                add_input_text(tag="encoder_path", default_value=args["encoder_path"], callback=update_value, user_data="encoder_path", width=-1)
                button_list.append(add_button(label="Select Encoder Path", callback=lambda: open_file_dialog("encoder_path")))

            # RIGHT COLUMN
            with group(horizontal=False):
                add_text("Video Parameter")
                add_spacer(width=0, height=10)  # 10 pixels vertical space
                add_input_float(label="offset_fg", default_value=args["offset_fg"], callback=update_value, user_data="offset_fg")
                add_input_float(label="offset_bg", default_value=args["offset_bg"], callback=update_value, user_data="offset_bg")

                add_text("Performance Parameter")
                add_input_int(label="Num_Workers", default_value=args["Num_Workers"], callback=update_value, user_data="Num_Workers")
                add_input_int(label="num_gpu", default_value=args["num_gpu"], callback=update_value, user_data="num_gpu")
                add_input_int(label="Num_GPU_Workers", default_value=args["Num_GPU_Workers"], callback=update_value, user_data="Num_GPU_Workers")
                add_input_int(label="Max_Frame_Count", default_value=args["Max_Frame_Count"], callback=update_value, user_data="Max_Frame_Count")
                add_combo(label="repair_mode", items=["0 - Full", "1 - Rerun no combine", "2 - Combine - Export video", "3 - [Debug] Combine video only, temp.mp4"],
                          default_value="0 - Full",
                          callback=lambda s,a,u: update_value(s,int(a[0]),"repair_mode"), user_data="repair_mode")
                add_text("Debug Parameter, don't touch")
                add_input_text(label="start_frame", default_value=args["start_frame"], callback=update_value, user_data="start_frame")
                add_input_text(label="end_frame", default_value=args["end_frame"], callback=update_value, user_data="end_frame")

                add_spacer(width=0, height=10)  # 10 pixels vertical space
                with group(horizontal=True):
                    add_text("Debug Directory")
                    button_list.append(add_button(label="Select DebugDir", callback=lambda: open_dir_dialog("DebugDir")))
                add_input_text(tag="DebugDir", default_value=args["DebugDir"], callback=update_value, user_data="DebugDir", width=-1)

    add_separator()
    add_text("Command Preview:")
    add_input_text(multiline=True, readonly=True, tag="preview_text", width=950, height=80)
    add_button(label="Run", callback=run_script, width=200, height=50)
    add_progress_bar(tag="progress", default_value=0.0, width=300)

# Apply BIG font to everything
bind_font(big_font)
# --- create button theme ---
with theme() as button_theme:
    with theme_component(mvButton):
        add_theme_color(mvThemeCol_Button, (100, 149, 237))        # normal
        add_theme_color(mvThemeCol_ButtonHovered, (65, 105, 225))  # hover
        add_theme_color(mvThemeCol_ButtonActive, (25, 25, 112))    # pressed

# --- apply the theme to all buttons ---
for btn in button_list:
    bind_item_theme(btn, button_theme)
    
update_preview()
setup_dearpygui()
show_viewport()
set_primary_window("main_window", True)
start_dearpygui()
destroy_context()
