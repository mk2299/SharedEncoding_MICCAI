from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
from keras.utils import multi_gpu_model
from custom_layers import CrowdsRegression_SRM_conv_blocks
import sys

def pretrained_vggish_individual(model_path = '../base/vggish_weights_keras.h5'):
 
    # Arguments: 
    ## model_path: Path to the pretrained VGGish model in keras 
    
    def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

    
    def _upsample_add( x, y, crop = 0):
        
        out = UpSampling2D(size=(2,2), interpolation='bilinear')(x)
        if crop==1:
            ch, cw = get_crop_shape(out, y)
            out = Cropping2D(cropping=(ch,cw))(out)
       
        return Add()([out, y]) 
    
    base_model = load_model(model_path)
                 
    
    layer1 =  base_model.layers[3].output 
    layer2 =  base_model.layers[6].output 
    layer3 =  base_model.layers[9].output 
    layer4 =  base_model.get_output_at(-1)
    
    smooth1 = Conv2D(128, kernel_size=3, strides=1, padding="same")
    smooth2 = Conv2D(128, kernel_size=3, strides=1, padding="same")
    
    # Lateral layers
    toplayer  = Conv2D(128, kernel_size=1, strides=1) 
    latlayer1 = Conv2D(128, kernel_size=1, strides=1) 
    latlayer2 = Conv2D(128, kernel_size=1, strides=1) 
    
    
    p5 = toplayer(layer3)
    p4 = _upsample_add(p5, latlayer1(layer2), crop = 1)
    p4 = smooth1(p4)  
    p3 = _upsample_add(p4, latlayer2(layer1))
    p3 = smooth2(p3)
    
    ####### Concatenate features from all layers ##############
    z = concatenate([layer4, GlobalAveragePooling2D()(p3), GlobalAveragePooling2D()(p4), GlobalAveragePooling2D()(p5)])
    
    ##### Convolutional response model ###################
    x = Dense(6*7*6*1024, activation='elu')(z)
    y = Reshape((6,7,6,1024))(x)
    
    y = Conv3DTranspose(512,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(256,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(128,(3,3,3), (2,2,2), activation='elu')(y)
    predictions = Conv3DTranspose(num_features,(3,3,3), (2,2,2), activation='elu')(y)
    
    model = Model(inputs = base_model.input, outputs = predictions)
    return model, base_model.input, predictions

def pretrained_vggish_shared(model_path = '../base/vggish_weights_keras.h5', num_subs = 10):
 
    # Arguments: 
    ## num_subs: Number of unique subjects in the fMRI dataset  
    
    def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

    
    def _upsample_add( x, y, crop = 0):
        
        out = UpSampling2D(size=(2,2), interpolation='bilinear')(x)
        if crop==1:
            ch, cw = get_crop_shape(out, y)
            out = Cropping2D(cropping=(ch,cw))(out)
       
        return Add()([out, y]) 
    
    base_model = load_model(model_path)
                 
    
    layer1 =  base_model.layers[3].output 
    layer2 =  base_model.layers[6].output 
    layer3 =  base_model.layers[9].output 
    layer4 =  base_model.get_output_at(-1)
    
    smooth1 = Conv2D(128, kernel_size=3, strides=1, padding="same")
    smooth2 = Conv2D(128, kernel_size=3, strides=1, padding="same")
    
    # Lateral layers
    toplayer  = Conv2D(128, kernel_size=1, strides=1) 
    latlayer1 = Conv2D(128, kernel_size=1, strides=1) 
    latlayer2 = Conv2D(128, kernel_size=1, strides=1) 
    
    
    p5 = toplayer(layer3)
    p4 = _upsample_add(p5, latlayer1(layer2), crop = 1)
    p4 = smooth1(p4)  
    p3 = _upsample_add(p4, latlayer2(layer1))
    p3 = smooth2(p3)
    
    ######## Concatenate features from all layers #########
    z = concatenate([layer4, GlobalAveragePooling2D()(p3), GlobalAveragePooling2D()(p4), GlobalAveragePooling2D()(p5)])
    
    #### Convolutional response model #####
    x = Dense(6*7*6*1024, activation='elu')(z)
    y = Reshape((6,7,6,1024))(x)
    
    y = Conv3DTranspose(512,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(256,(3,3,3), (2,2,2), activation='elu')(y)
    predictions = CrowdsRegression_SRM_conv_blocks( num_subs = num_subs)(y) 

    model = Model(inputs = base_model.input, outputs = predictions)
    return model, base_model.input, predictions


