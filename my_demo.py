import argparse, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from scipy.spatial import distance
import cv2
import sys

from pathlib import Path # if you haven't already done so
# sys.path.append(str(Path(__file__).resolve().parents[1]))
# print(sys.path)

from model import ModelSpatial
from utils import imutils, evaluation
from config import *
from dataset import AttentionFlow
from head_det import generator


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="gpu id")
parser.add_argument("--model_weights", type=str, default="pretrained-models/model_demo.pt", help="model weights")
parser.add_argument("--batch_size", type=int, default=40, help="batch size")
parser.add_argument('--vis_mode', type=str, help='heatmap or arrow', default='arrow')
parser.add_argument('--out_threshold', type=int, help='out-of-frame target dicision threshold', default=200)
args = parser.parse_args()

def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)


base_path = os.path.dirname(os.path.abspath(__file__)) 

def find_closest_object_of_focus(gaze_point, other_objects_on_frame):
    min_distance = float("inf")
    object_of_index = None
    for index, objects in enumerate(other_objects_on_frame):
        object_centroid = (int((objects[3]+objects[5])/2),int((objects[4]+objects[6])/2))
        euclid_dist = distance.euclidean(object_centroid,gaze_point)
        if euclid_dist < min_distance:
            min_distance =  euclid_dist
            object_of_index = index
    
    return object_of_index
        
    ###### my  version
def run(video_file,is_delete,fps):
    #generate data
    frames, data_dict = generator(video_file+'.mp4',is_delete=is_delete,fps=fps)
    # Define device
    device = torch.device('cuda', args.device)

    # Load model
    print("Constructing model")
    model = ModelSpatial()
    model.cuda().to(device)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.model_weights)
    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print('Evaluation in progress ...')
    model.train(False)

    transform = _get_transform()

    for person_id, person_detail in data_dict['head'].items():

        # if the person does not appear in 70% and above of the frames, then skip.
        # This is neccessary to weed out false head detection from the detection model
        # if int((len(head_bboxes)/len(frames))*100) < 40:
        #     continue
        
        frame_index = 0

        #make person output dir
        save_dir = Path(base_path+'/data/demo/output_frames/'+video_file+"/")  
        person_dir = 'person_'+str(person_id)
        (save_dir / person_dir).mkdir(parents=True, exist_ok=True)  # make dir
        # Prepare data
        print("Loading Data for person ", person_id)
        val_dataset = AttentionFlow(frames, person_detail,transform, input_size=input_resolution, output_size=output_resolution)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=0)
    
    
        with torch.no_grad():
            for val_batch, (val_img, val_face, val_head_channel, headbox, imsize, frame_id,track_id) in enumerate(val_loader):
                
                val_images = val_img.cuda().to(device)
                val_head = val_head_channel.cuda().to(device)
                val_faces = val_face.cuda().to(device)
                val_gaze_heatmap_pred, val_attmap, val_inout_pred = model(val_images, val_head, val_faces)

                for j in range(len(val_gaze_heatmap_pred)):
                    #heatmap modulation
                    raw_hm = val_gaze_heatmap_pred[j].cpu().detach().numpy() * 255
                    raw_hm = raw_hm.squeeze()
                    inout = val_inout_pred[j].cpu().detach().numpy()
                    inout = 1 / (1 + np.exp(-inout))
                    inout = (1 - inout) * 255

                    width, height = int(imsize[j][0]), int(imsize[j][1])
                    x1,y1,x2,y2 = int(headbox[j][0]), int(headbox[j][1]), int(headbox[j][2]), int(headbox[j][3])
            
                    norm_map = cv2.resize(raw_hm,(height, width)) - inout # imresize(raw_hm, (height, width)) - inout
                    
            
                    frame_raw = frames[frame_index].detach().clone().numpy()
                    
                    
                    if inout < args.out_threshold: # in-frame gaze
                        pred_x, pred_y = evaluation.argmax_pts(raw_hm)
                        norm_p = [pred_x/output_resolution, pred_y/output_resolution]
                        
                        # get all other objects in the frame
                        other_objects = data_dict['other_objects'][int(frame_id[j])]
                        #find the closest object of focus
                        object_index = find_closest_object_of_focus((int(norm_p[0]*width), int(norm_p[1]*height)), other_objects)
                        if args.vis_mode == 'arrow':
                            frame_raw = cv2.circle(frame_raw, (int(norm_p[0]*width), int(norm_p[1]*height)), int(height/50.0), (255, 0, 0), 2) 
                            cv2.arrowedLine(frame_raw,(int((x1+x2)/2),int((y1+y2)/2)),(int(norm_p[0]*width),int(norm_p[1]*height)), (230,253,11),thickness=3)
                            #     #write the last processed frame to file
                            cv2.imwrite(os.path.join(base_path,'data/demo/output_frames/'+video_file+"/"+person_dir, str(frame_index)+'.jpg'), frame_raw) 
                            cv2.imwrite(os.path.join(base_path,'data/demo/output_frames/'+video_file+"/"+person_dir, str(frame_index)+'_'+str(other_objects[object_index][1])+'.jpg'), frame_raw[other_objects[object_index][4]:other_objects[object_index][6], other_objects[object_index][3]:other_objects[object_index][5]] ) 
                    # else: #out of frame gaze
                                

                    frame_index+=1        


if __name__ == "__main__":
    run('video_17_1',True,30)
