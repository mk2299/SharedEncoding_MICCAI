import os
from nibabel import cifti2
import numpy as np
import itertools
import nibabel as nib
import argparse


def normalize_activations(root = '../data/HCP_7TMovies/', data_loc = '../data/HCP_7TMovies/preprocessed/MinMax', movie_idx):

    subject_IDs = np.genfromtxt('../lists/subject_IDs_train.txt', delimiter=',',dtype=str)
    
    names = {'1':'AP', '2':'PA', '3':'PA', '4':'AP'}
    clips = np.load('./clip_times_24.npy').item().get(movie_idx)  ## Valid 'movie' clip periods manually extracted 
    
    frame_idx = []
    for c in range(len(clips)): 
        frame_idx.append(np.arange(clips[c,0]/24, clips[c,1]/24).astype('int'))
    frame_idx = np.array(list(itertools.chain(*frame_idx)))
  
    for sub_folder in subject_IDs:
        print('Processing data for subject : ', sub_folder)
        
        sub_file = os.path.join(root,  sub_folder,  'MNINonLinear/Results/', 'tfMRI_MOVIE' + movie_idx + '_7T_' + names[movie_idx], 'tfMRI_MOVIE' + movie_idx + '_7T_' + names[movie_idx] + '_hp2000_clean.nii.gz')

        if os.path.exists(sub_file):

            data = nib.load(sub_file).get_fdata()  

            mins = data.min(axis=3)[:,:,:,np.newaxis]
            maxs = data.max(axis=3)[:,:,:,np.newaxis]
            
            ######### Min-max normalization ###########
            
            data = (data - mins)/(maxs - mins) 
            data[np.isnan(data)] = 0               
            if not os.path.isdir(os.path.join(data_loc, sub_folder)):
                os.mkdir(os.path.join(data_loc, sub_folder))                 
            out_file = os.path.join(data_loc, sub_folder, 'MOVIE'+ movie_idx +'_MNI.npy') 
            
            np.save(out_file, data) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'preprocess volumetric data')
    parser.add_argument('--movie', default = '4', type = str, help= 'Movie index')
    parser.add_argument('--root', default = '../data/HCP_7TMovies/', type = str, help= 'Path to downloaded HCP volumetric data (MNI 1.6mm)')
    parser.add_argument('--data_loc', default = '../data/HCP_7TMovies/preprocessed/MinMax', type = str, help= 'Path for saving normalized fMRI data')
    args = parser.parse_args()
    normalize_activations(args.root, args.data_loc, args.movie) 
    