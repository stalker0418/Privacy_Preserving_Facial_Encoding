from src.Privacy_Encoding.components.encoding_models import encodingModels
from src.Privacy_Encoding.logging import logger
import numpy as np
from src.Privacy_Encoding.config.configuration import ConfigurationManager
from src.Privacy_Encoding.utils.common import *
from src.Privacy_Encoding.utils.imageHandling import *


class dataEncodingPipeline():
    def __init__(self,x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    
    def single_convolution_encoding(self, data):
        encoding_models = encodingModels(self.x_train, self.y_train)
        convolutional_encoder = encoding_models.convolution_encoder(layer = 1)
        logger.info(f"Single Convolutional Encoder initialized successfully")

        x_conv_train_1 = convolutional_encoder(self.x_train)
        x_conv_train_1 = np.sum(x_conv_train_1, axis = 3)

        # print(self.x_train[0], x_conv_train_1[0])

        logger.info(f"Completed Single Convolution ENcoding successfully. The shape of x_conv_train is {x_conv_train_1.shape}")

        config = ConfigurationManager()
        save_images_config = config.get_save_encoding_images_config()

        if save_images_config.enabled:
            create_directories([save_images_config.output_dir])
            output_dir = os.path.join(Path(save_images_config.output_dir), Path(f"Single_Convolution/{data}"))
            save_images_to_directory(output_dir, x_conv_train_1, self.y_train)

            logger.info(f"Successfully Completed saving Single COnvolution Images to the directory {output_dir}")

        return x_conv_train_1
    

