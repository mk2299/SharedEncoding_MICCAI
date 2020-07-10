from keras import layers
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
from keras.applications.resnet50 import ResNet50
import sys
from custom_layers import CrowdsRegression_SRM_conv_blocks

  

def pretrained_resnet_shared(pretrained = True, num_subs = 10):
    
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
        #print(x.shape, y.shape)
        out = UpSampling2D(size=(2,2), interpolation='bilinear')(x)
        if crop==1:
            ch, cw = get_crop_shape(out, y)
            out = Cropping2D(cropping=(ch,cw))(out)
       
        return Add()([out, y]) 
    
    base_model = ResNet50(include_top= False, weights="imagenet", input_shape=(720,1024,3))
    if pretrained:
           for layer in base_model.layers:
                   layer.trainable = False
                    
    base_model.layers.pop()
    base_model.layers[-1].outbound_nodes = []
    base_model.outputs = [base_model.layers[-1].output]                
    
    layer1 =  base_model.layers[38].output #base_model.get_layer('activation_10').output
    layer2 =  base_model.layers[80].output #base_model.get_layer('activation_22').output 
    layer3 =  base_model.layers[142].output #base_model.get_layer('activation_40').output
    layer4 =  base_model.get_output_at(-1)
    
    smooth1 = Conv2D(256, kernel_size=3, strides=1, padding="same")
    smooth2 = Conv2D(256, kernel_size=3, strides=1, padding="same")
    smooth3 = Conv2D(256, kernel_size=3, strides=1, padding="same")
    # Lateral layers
    toplayer  = Conv2D(256, kernel_size=1, strides=1) #(layer4)
    latlayer1 = Conv2D(256, kernel_size=1, strides=1) #(layer3)
    latlayer2 = Conv2D(256, kernel_size=1, strides=1) #(layer2)
    latlayer3 = Conv2D(256, kernel_size=1, strides=1) #(layer1)
    
    p5 = toplayer(layer4)
    p4 = _upsample_add(p5, latlayer1(layer3), crop = 1)
    p4 = smooth1(p4)  
    p3 = _upsample_add(p4, latlayer2(layer2))
    p3 = smooth2(p3)
    p2 = _upsample_add(p3, latlayer3(layer1))
    p2 = smooth3(p2)
    
    z = concatenate([GlobalAveragePooling2D()(p2), GlobalAveragePooling2D()(p3), GlobalAveragePooling2D()(p4), GlobalAveragePooling2D()(layer4)])
    x = Dense(1024, activation='elu')(z)  
    x = Dense(6*7*6*1024, activation='elu')(x)
    y = Reshape((6,7,6,1024))(x)
    
    y = Conv3DTranspose(512,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(256,(3,3,3), (2,2,2), activation='elu')(y)
    predictions = CrowdsRegression_SRM_conv_blocks(num_subs = num_subs)(y) 
    model = Model(inputs = base_model.input , outputs = predictions)
    return model, base_model.input, predictions   

def pretrained_resnet_individual(pretrained = True):
    
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
        #print(x.shape, y.shape)
        out = UpSampling2D(size=(2,2), interpolation='bilinear')(x)
        if crop==1:
            ch, cw = get_crop_shape(out, y)
            out = Cropping2D(cropping=(ch,cw))(out)
       
        return Add()([out, y]) 
    
    base_model = ResNet50(include_top= False, weights="imagenet", input_shape=(720,1024,3))
    if pretrained:
           for layer in base_model.layers:
                   layer.trainable = False
                    
    base_model.layers.pop()
    base_model.layers[-1].outbound_nodes = []
    base_model.outputs = [base_model.layers[-1].output]                
    
    layer1 =  base_model.layers[38].output #base_model.get_layer('activation_10').output 
    layer2 =  base_model.layers[80].output #base_model.get_layer('activation_22').output 
    layer3 =  base_model.layers[142].output #base_model.get_layer('activation_40').output 
    layer4 =  base_model.get_output_at(-1)
    
    smooth1 = Conv2D(256, kernel_size=3, strides=1, padding="same")
    smooth2 = Conv2D(256, kernel_size=3, strides=1, padding="same")
    smooth3 = Conv2D(256, kernel_size=3, strides=1, padding="same")
    # Lateral layers
    toplayer  = Conv2D(256, kernel_size=1, strides=1) #(layer4)
    latlayer1 = Conv2D(256, kernel_size=1, strides=1) #(layer3)
    latlayer2 = Conv2D(256, kernel_size=1, strides=1) #(layer2)
    latlayer3 = Conv2D(256, kernel_size=1, strides=1) #(layer1)
    
    p5 = toplayer(layer4)
    p4 = _upsample_add(p5, latlayer1(layer3), crop = 1)
    p4 = smooth1(p4)  
    p3 = _upsample_add(p4, latlayer2(layer2))
    p3 = smooth2(p3)
    p2 = _upsample_add(p3, latlayer3(layer1))
    p2 = smooth3(p2)
    
    z = concatenate([GlobalAveragePooling2D()(p2), GlobalAveragePooling2D()(p3), GlobalAveragePooling2D()(p4), GlobalAveragePooling2D()(layer4)]) ## Note the change: layer4 is used instead of p5 for the last input to concatenation
    z = Dense(1024, activation = 'elu')(z)    
    x = Dense(6*7*6*1024, activation='elu')(z)
    y = Reshape((6,7,6,1024))(x)
    
    y = Conv3DTranspose(512,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(256,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(128,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(1,(3,3,3), (2,2,2), activation='elu')(y)
    model = Model(inputs = base_model.input , outputs = y)
    return model   








