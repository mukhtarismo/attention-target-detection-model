# import sys
# sys.path.insert(0,'./ScaledYOLOv4')
# sys.path.insert(1,'./yolo_head')
from yolo_head.detect import det_head
## Build a dictionary and iterate through each image
import cv2
import os
from pathlib import Path


def delete_files_in_folder(folder_path):
    #Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"this folder path '{folder_path}' does not exit")
        return

    # Get all files and subfolders in a folder
    files = os.listdir(folder_path)

    for file in files:
        file_path = os.path.join(folder_path, file)

        if os.path.isfile(file_path):
            # If it's a file, delete it
            os.remove(file_path)
            print(f"del_file: {file_path}")
        elif os.path.isdir(file_path):
            # If it's a folder, delete it recursively
            delete_files_in_folder(file_path)
    
    # Delete empty folders


def generator(video_file_name,is_delete=False,fps=1):
    base_path = os.path.dirname(os.path.abspath(__file__))    
    imgset =os.path.join(base_path,'data/demo/Frames2/*.jpg') 
    vid_dir_name = video_file_name.split('.')[0]
    if is_delete:
        delete_files_in_folder("data/demo/Frames2/")
        delete_files_in_folder("data/demo/output_frames/"+vid_dir_name+"/")
        delete_files_in_folder("data/demo/annotated_frames/")
    
    save_dir = Path(base_path+'/data/demo/output_frames')  
    (save_dir / vid_dir_name).mkdir(parents=True, exist_ok=True)  # make dir
    # Initialize
    cap = cv2.VideoCapture('./'+video_file_name)
    frame_id = 0
    frame_count = 0
    while   True:
        ret, frame = cap.read()
        if ret:
            frame_count = frame_count +1
            if frame_count % fps == 0:
                cv2.imwrite('data/demo/Frames2/%d.jpg' % frame_id, frame)
                frame_id += 1
        else:
            break
    frames, data_dict = det_head(imgset,base_path)

    return frames, data_dict



