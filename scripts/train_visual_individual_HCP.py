import os

import argparse


def main():
    parser = argparse.ArgumentParser(description='Single frame model')
    
    parser.add_argument('--lrate', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default = 16, type=int)
    parser.add_argument('--subject_id', default = None, type=str)
    parser.add_argument('--model_file', default = None, help = 'Location for saving model')
    parser.add_argument('--lastckpt_file', default = None, help = 'Location for saving last model')
    parser.add_argument('--log_file', default = None, help = 'Location for saving logs')
    parser.add_argument('--gpu_devices', default = "0", type = str, help = 'Device IDs')
    parser.add_argument('--pretrained', default = 1, type = int, help = 'Freeze ResNet weights')
    parser.add_argument('--delay', default = None, type = int, help = 'HR')
    parser.add_argument('--list_IDs', default = '../lists/', type = str, help = 'Path to training and validation ID files')
    
    args = parser.parse_args()
   
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    import numpy as np
    import sys
    sys.path.append('../models/')
    sys.path.append('../dataloaders/')
    from models_visual_HCP import pretrained_resnet_individual
    from dataloader_visual_HCP import DataGenerator_individual
    from keras.callbacks import ModelCheckpoint, EarlyStopping,  CSVLogger
    from keras.models import *
    from keras import optimizers
    from losses import LossHistory
    
    IDs_train = np.genfromtxt(os.path.join(args.list_IDs, 'ListIDs_mean_train.txt'), dtype = 'str') 
    IDs_val = np.genfromtxt(os.path.join(args.list_IDs, 'ListIDs_mean_val.txt'), dtype = 'str') 

    train_generator = DataGenerator_individual(IDs_train, subject = args.subject_id, batch_size = args.batch_size, train = True, delay = args.delay)
    val_generator = DataGenerator_individual(IDs_val,  subject = args.subject_id, batch_size = args.batch_size, train = False, delay = args.delay)

    history = LossHistory()

    callback_save = ModelCheckpoint(args.model_file, monitor="val_mean_squared_error", save_best_only=True)
    saver = CSVLogger(args.log_file)
   
    model = pretrained_resnet_individual(pretrained = args.pretrained)

    print(model.summary())
    model.compile(optimizer=optimizers.Adam(lr=args.lrate, amsgrad=True), loss='mean_squared_error',metrics=['mean_squared_error'])
    model.fit_generator(
        train_generator,
        validation_data = val_generator,
        callbacks = [history, saver, callback_save], steps_per_epoch = 2000, validation_steps = 500,
        epochs = args.epochs)
    model.save(args.lastckpt_file)
if __name__ == '__main__':
    main()
