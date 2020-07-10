import numpy as np
import keras
import imageio
from skimage.transform import resize
import nibabel as nib
import os

       
class DataGenerator_individual(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,  train= True, split = [0.9,0.1], subject = None, batch_size = 16, dim=(720,1024),  volume = (111, 127, 111), n_channels = 3, delay = None, shuffle=True):
        'Initialization'
        
        self.movie_files = ['7T_MOVIE1_CC1_v2.mp4', '7T_MOVIE2_HO1_v2.mp4', '7T_MOVIE3_CC2_v2.mp4', '7T_MOVIE4_HO2_v2.mp4']

        self.root_data='../preprocess/Post_20140821_version/' 
        self.vol_root = '../data/HCP_7TMovies/preprocessed/MinMax/' 
        self.subject = subject
        self.delay = delay
     
        self.videos = []
        for movie in self.movie_files:
            path = os.path.join(self.root_data, movie)
            self.videos.append(imageio.get_reader(path,  'ffmpeg'))
            
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
        X = np.zeros((self.batch_size, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size, *self.volume, 1))

        # Generate data
        for i,idx in enumerate(list_indexes):
            # Store sample
            movie, frame = self.list_IDs[idx]
            
            X[i] = np.array(self.videos[int(movie)-1].get_data(int(frame)))
            y[i,:,:,:,0] = np.load(os.path.join(self.vol_root, self.subject, 'MOVIE'+ movie+'_MNI.npy'), mmap_mode='r')[2:,4:-5,:-2,int(int(frame)/24)+self.delay] 

        return X, y    
    
    
class DataGenerator_shared(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,  train= True, split = [0.9,0.1], batch_size = 16, num_subs = 10, dim=(720,1024),  volume = (111, 127, 111), n_channels = 3, delay = None, shuffle=True):
        'Initialization'
        self.subjects = np.genfromtxt('../lists/subject_IDs_train.txt', delimiter = ',', dtype = str)[:num_subs]
        self.movie_files = ['7T_MOVIE1_CC1_v2.mp4', '7T_MOVIE2_HO1_v2.mp4', '7T_MOVIE3_CC2_v2.mp4', '7T_MOVIE4_HO2_v2.mp4']

       
        self.root_data='../preprocess/Post_20140821_version/' 
        self.vol_root = '../data/HCP_7TMovies/preprocessed/MinMax/' 
        
        self.delay = delay
        self.num_subs = num_subs
        self.videos = []
        for movie in self.movie_files:
            path = os.path.join(self.root_data, movie)
            self.videos.append(imageio.get_reader(path,  'ffmpeg'))
            
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
        X = np.zeros((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.volume, self.num_subs))
        # Generate data
        for i,idx in enumerate(list_indexes):
            # Store sample
            movie, frame = self.list_IDs[idx]
            
            X[i] = np.array(self.videos[int(movie)-1].get_data(int(frame)))
            for j in range(self.num_subs):
                y[i,:,:,:,j] = np.load(os.path.join(self.vol_root, self.subjects[j], 'MOVIE'+ movie+'_MNI.npy'), mmap_mode='r')[2:,4:-5,:-2,int(int(frame)/24)+self.delay]
        return X, y        

    