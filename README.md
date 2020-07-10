# SharedEncoding_MICCAI
Source code for the following paper: https://arxiv.org/pdf/2006.15802.pdf
(To appear in the proceedings of MICCAI 2020)

# Background
In the present study, we propose a shared convolutional neural encoding method that accounts for individual-level differences. Our method leverages multi-subject data to improve the prediction of subject-specific responses evoked by visual or auditory stimuli. We showcase our approach on high-resolution 7T fMRI
data from the Human Connectome Project movie-watching protocol and demonstrate significant improvement over single-subject encoding models. 

# Getting Started 
_Data organization_ <br>
All experiments in this study are based on the Human Connectome Project movie-watching database. The dataset is publically available for download through the ConnectomeDB software [https://db.humanconnectome.org/]. Here, we utilized 7T fMRI data from the 'Movie Task fMRI 1.6mm/59k FIX-Denoised' package. Training models using the code provided herein will be easiest if data is organized according to the file structure within the data folder of this repo. 
Once the data is downloaded, run "preprocess_fMRI.py  --movie #index" to normalize the range of fMRI data for all 4 movies (index 1-4). 

_Base models_ <br>
Pre-trained models for audio classification were obtained from a large-scale audio classification study [Hershey et al., ICASSP 2017]. Clone the following repository before proceeding:  https://github.com/tensorflow/models/tree/master/research/audioset. 
The preprocessing of raw audio waveforms is based on code from the above repository. Once cloned, use the notebook 'preprocess_audio.ipynb' within the preprocess folder to extract mel-spectrograms from the HCP video files.  

Note that the VGGish model checkpoint on the audioset repo was converted into keras using the following code. Please follow their instructions to get VGGish keras weights and store it in the 'base' folder under root directory with the name 'vggish_weights_keras.h5'. 
https://github.com/antoinemrcr/vggish2Keras/blob/master/convert_ckpt.py

_Training_ <br>
To train the shared auditory encoding model, run the following script from the scripts folder: <br>
python train_audio_shared_HCP.py --lrate 0.0001 --epochs 50 --model_file model_best_path --lastckpt_file last_ckpt_path --log_file log_path --delay 4 --gpu_device 0 --batch_size 1

To train the shared visual encoding model, run the following script: <br>
python train_visual_shared_HCP.py --lrate 0.0001 --epochs 20 --model_file model_best_path --lastckpt_file last_ckpt_path --log_file log_path --delay 4 --gpu_device 0 --batch_size 1

To run individual level (non-shared) encoding models for subjects, run the bash script provided in the scripts folder after possibly changing the locations for saving models:
1. For audio models, run "./train_audio_individual_models.sh"
2. For visual models, run "./train_visual_individual_models.sh"

# References
* Meenakshi Khosla, Gia H. Ngo, Keith Jamison, Amy Kuceyeski and Mert R. Sabuncu. A shared neural encoding model for the prediction of subject-specific fMRI response. Tech report, arXiv, July 2020. 
* Hershey, S. et. al., CNN Architectures for Large-Scale Audio Classification, ICASSP 2017
# Bugs and Questions 
Please contact Meenakshi Khosla at mk2299@cornell.edu if you have any questions.  
