import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import imageio

import matplotlib.pyplot as plt
import nibabel as nib
import h5py
import numpy as np
import itertools
import matplotlib.image as mpimg
import argparse
from keras.models import *
from scipy.stats import pearsonr, linregress
from keras import backend as K
import sys



def data_generation_stimulus(movie_idx = '4'):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        movie_files = ['7T_MOVIE1_CC1_v2.mp4', '7T_MOVIE2_HO1_v2.mp4', '7T_MOVIE3_CC2_v2.mp4', '7T_MOVIE4_HO2_v2.mp4']
        root_data='../preprocess/Post_20140821_version/' 
        videos = []
        for movie in movie_files:
            path = os.path.join(root_data, movie)
            videos.append(imageio.get_reader(path,  'ffmpeg'))
     
        
        clips = np.load('../preprocess/clip_times_24.npy')
        
        video = videos[int(movie_idx)-1]
        
        idxs = clips.item().get(movie_idx)
        frame_idx = []
        for c in range(len(idxs)-1): ## Get rid of the last segment (it is repeated in the first 3 movies)
                frame_idx.append(np.arange(idxs[c,0]/24, idxs[c,1]/24).astype('int'))
        frame_idx = np.array(list(itertools.chain(*frame_idx)))*24
         
       
              
        x = [video.get_data(int(frame)) for frame in frame_idx]
                    
        return np.asarray(x)

    
def main():
    parser = argparse.ArgumentParser(description='Model evaluation')
    parser.add_argument('--model_file', default=None, type = str, help='Model')
    parser.add_argument('--predictions_file', default=None, type = str, help='File for saving predictions')
    args = parser.parse_args()

    model = load_model(args.model_file) 
    X = data_generation_stimulus(movie_idx = '4') 
   
    print('Predicting..')    
    Y = model.predict(X, batch_size = 1)
    print('Saving..')
    np.save(args.predictions_file, Y)
    
if __name__ == '__main__':
    main()
    