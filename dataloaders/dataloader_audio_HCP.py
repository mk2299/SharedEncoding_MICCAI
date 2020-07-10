import numpy as np
import keras
import imageio
from skimage.transform import resize
import nibabel as nib
import os
import cv2


    
class DataGenerator_shared(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,  train= True, split = [0.9,0.1], batch_size=32,  dim=(96,64),  num_subs = 10, volume = (111, 127, 111), n_channels = 1, delay = None, shuffle=True):
        'Initialization'
        self.subjects = np.genfromtxt('../lists/subject_IDs_train.txt', delimiter = ',', dtype = str)[:num_subs]
        
        self.root = '../preprocess/audio_feats/'
        self.vol_root = '../data/HCP_7TMovies/preprocessed/MinMax/' 
        self.delay = delay
        
        self.num_subs = num_subs
        self.audio_files = ['audio1_audioset.npy', 'audio2_audioset.npy', 'audio3_audioset.npy']
        self.audios = []
        for audio in self.audio_files:
            path = os.path.join(self.root, audio)
            self.audios.append(np.load(path))
            
        self.dim = dim
        self.volume = volume 
        self.batch_size = batch_size
        self.total_size = len(list_IDs)
        self.list_IDs = list_IDs

                 
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.list_IDs) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

   
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.volume, self.num_subs))

        # Generate data
        # Generate data
        for i,idx in enumerate(list_indexes):
            # Store sample
            movie, frame = self.list_IDs[idx]
           
            
            X[i,:,:,0] = self.audios[int(movie)-1][int(int(frame)/24)] 
            for j in range(self.num_subs):
                y[i,:,:,:,j] = np.load(os.path.join(self.vol_root, self.subjects[j], 'MOVIE'+ movie+'_MNI.npy'), mmap_mode='r')[2:,4:-5,:-2,int(int(frame)/24)+self.delay]
            

        return X, y 

  
    
class DataGenerator_individual(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,  train= True, split = [0.9,0.1], subject = None, batch_size=32,  dim=(96,64),  num_subs = 10, volume = (111, 127, 111), n_channels = 1, delay = None, shuffle=True):
        'Initialization'
        
        self.subject = subject
        self.root = '../preprocess/audio_feats'
        self.vol_root = '../data/HCP_7TMovies/preprocessed/MinMax/' 
        self.delay = delay
        
        self.num_subs = num_subs
        self.audio_files = ['audio1_audioset.npy', 'audio2_audioset.npy', 'audio3_audioset.npy']
        self.audios = []
        for audio in self.audio_files:
            path = os.path.join(self.root, audio)
            self.audios.append(np.load(path))
            
        self.dim = dim
        self.volume = volume 
        self.batch_size = batch_size
        self.total_size = len(list_IDs)
        self.list_IDs = list_IDs
         
        
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.list_IDs) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

   
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.volume, 1))

        # Generate data
        # Generate data
        for i,idx in enumerate(list_indexes):
            # Store sample
            movie, frame = self.list_IDs[idx]
           
            
            X[i,:,:,0] = self.audios[int(movie)-1][int(int(frame)/24)] 
          
            y[i,:,:,:,0] = np.load(os.path.join(self.vol_root, self.subject, 'MOVIE'+ movie+'_MNI.npy'), mmap_mode='r')[2:,4:-5,:-2,int(int(frame)/24)+self.delay]
            

        return X, y 

