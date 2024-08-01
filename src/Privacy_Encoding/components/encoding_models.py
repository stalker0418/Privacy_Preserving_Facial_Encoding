import keras
from keras import layers
from src.Privacy_Encoding.logging import logger

class encodingModels:
    def __init__(self, x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train

    def patching_encoder(self):
        # Model used to perform Patching on Images
        patching_model = keras.Sequential([
            layers.Conv2D(filters=32,kernel_size= (7,7), strides= (7,7), activation= 'relu', input_shape = (self.x_train.shape[1], self.x_train.shape[2], 1)),
            layers.MaxPooling2D(pool_size=(2,2))
        ])

        encode_model = keras.models.Model(inputs = patching_model.inputs, outputs = patching_model.layers[0].output)
        # print(encode_model.summary())
        logger.info(f"Patching Encoding Model is initialized. It's summary is as follows: {encode_model.summary()}")
        return encode_model
    
    def convolution_encoder(self, layer= 1):
        # Model used for feature extraction for Single Conv and Double Conv
        conv_model = keras.Sequential([
            layers.Conv2D(filters=32,kernel_size= (7,7), strides=(1,1), activation= 'relu', input_shape = (self.x_train.shape[1], self.x_train.shape[2], 1)),
            layers.MaxPooling2D(pool_size=(2,2)),
            layers.Conv2D(filters=64,kernel_size= (5,5), activation= 'relu', input_shape = (self.x_train.shape[1], self.x_train.shape[2], 1))
        ])

        conv_encode_model_2 = keras.models.Model(inputs = conv_model.inputs, outputs = conv_model.layers[2].output)
        conv_encode_model_1 = keras.models.Model(inputs = conv_model.inputs, outputs = conv_model.layers[0].output)

        # print(conv_encode_model_2.summary(), conv_encode_model_1.summary())
        return conv_encode_model_1 if layer == 1 else conv_encode_model_2
        
