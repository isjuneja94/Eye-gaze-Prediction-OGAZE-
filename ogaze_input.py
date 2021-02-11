import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import numpy as np
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd




def get_frames(vidpath = None,overlapping = False,train=True):
    frlen=[]
    vlen =  []
    #print(os.listdir(vidpath))
    if train:
        
        frames = []

        for vid in os.listdir(vidpath):
            
            if vid.endswith('.avi'):
                #print('vid=',vid)
                cap = cv2.VideoCapture(vidpath+'/'+vid)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frames.append(cv2.cvtColor(cv2.resize(frame,(640,640)), cv2.COLOR_BGR2RGB))
               
         
            frames.pop()
            #blob = cv2.dnn.blobFromImages(frames, 1/255.0, (416, 416), swapRB=True, crop=False)
    cap.release()
    cv2.destroyAllWindows()
    return np.array(frames)


def get_grids(labels_path = None,vidpath=None):
    frlen=[]
    vlen =  []
    inpgrid = []
    gridarr=[]
    coordinates=[]

    gridlist = os.listdir(labels_path)
    vidlist = os.listdir(vidpath)
    newgridlist=[]
    for vid in vidlist:
        if vid.endswith('.avi'):
     
            file = vid[:-4] + '.txt'
            #print(file,'read')
           
            heatarr = []
            with open(labels_path+'/'+file, "r") as f:
            
                #print(file)
                content = f.readlines()
                for i in range(len(content)):
            
                    grid = np.zeros((16,16))
                    x,y = content[i].strip().split(',')[0:2]
                    coords = np.abs(np.array(([round(int(x)/2),round(int(y)/1.125)]))) 
                    coordinates.append(coords)
                    grid[int(int(y)//45),int(int(x)//80)]=1
                    gridarr.append(grid)
    gridarray = np.array(gridarr)
    coordinates = np.array(coordinates)
                    
            

           
    return gridarray,coordinates


if __name__ == '__main__':
    
   
    inp_array = get_frames(vidpath='../gaze_data/val/videos')
    print('shape:',inp_array.shape)
    inp_grid,hm = get_grids(labels_path='../gaze_data/val/labels',vidpath='../gaze_data/val/videos')
    print('heatmap = ',hm.shape,inp_grid.shape)

