#!/bin/bash


subjectlist=subject_IDs_train.txt

while read -r subject;
do
    python train_visual_individual_HCP.py --lrate 0.0001 --epochs 20 --model_file '../saved_models/Individual_models/singleframe_'$subject'_vol_FPN_visual_best.h5' --lastckpt_file '../saved_models/Individual_models/singleframe_'$subject'_vol_FPN_visual_last.h5' --log_file '../saved_models/Individual_models/singleframe_'$subject'_vol_FPN_visual.log' --delay 4 --gpu_device 0 --batch_size 1 --subject_id $subject
done < $subjectlist    



 