from src.Privacy_Encoding.components.encoding_models import encodingModels
from src.Privacy_Encoding.logging import logger
import numpy as np
from src.Privacy_Encoding.config.configuration import ConfigurationManager
from src.Privacy_Encoding.utils.common import *
from src.Privacy_Encoding.utils.imageHandling import *
from src.Privacy_Encoding.components.transforms import ImageTransformations
from src.Privacy_Encoding.entity import encodingsConfig
import torch
from torch.utils.data import TensorDataset, DataLoader


class dataEncodingPipeline():
    def __init__(self,x_train, y_train, x_test, y_test):
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.config = ConfigurationManager()

        self.transform = ImageTransformations().get_transform()
        x_tensor_train, x_tensor_test = torch.stack([self.transform(image) for image in x_train]), torch.stack([self.transform(image) for image in x_test])
        self.train_dataset, self.test_dataset = [x_tensor_train], [x_tensor_test]
    
    def single_convolution_encoding(self):
        encoding_models = encodingModels(self.x_train, self.y_train)
        convolutional_encoder = encoding_models.convolution_encoder(layer = 1)
        logger.info(f"Single Convolutional Encoder initialized successfully")

        x_conv_train_1, x_conv_test_1 = convolutional_encoder(self.x_train), convolutional_encoder(self.x_test)
        x_conv_train_1, x_conv_test_1 = np.sum(x_conv_train_1, axis = 3), np.sum(x_conv_test_1, axis = 3)
        

        # print(self.x_train[0], x_conv_train_1[0])

        logger.info(f"Completed Single Convolution ENcoding successfully. The shape of x_conv_train is {x_conv_train_1.shape}")

        
        save_images_config = self.config.get_save_encoding_images_config()

        if save_images_config.enabled:
            create_directories([save_images_config.output_dir])
            output_dir = os.path.join(Path(save_images_config.output_dir), Path(f"Single_Convolution/Train"))
            save_images_to_directory(output_dir, x_conv_train_1, self.y_train)
            logger.info(f"Successfully Completed saving Single COnvolution Images to the directory {output_dir}")


            output_dir = os.path.join(Path(save_images_config.output_dir), Path(f"Single_Convolution/Test"))
            save_images_to_directory(output_dir, x_conv_test_1, self.y_test)
            logger.info(f"Successfully Completed saving Single COnvolution Images to the directory {output_dir}")


        

        x_tensor_train, x_tensor_test = self.convert_to_tensor(x_conv_train_1), self.convert_to_tensor(x_conv_test_1)
        y_tensor_train, y_tensor_test = torch.LongTensor(self.y_train), torch.LongTensor(self.y_test)


        logger.info(f"Successfully completed converted the single convolution encoed images to tensors")
        return x_tensor_train, x_tensor_test
    
    def convert_to_tensor(self, x_train):
        return torch.stack([self.transform(image) for image in x_train])

    def get_encoded_tensors(self):
        encoding_triggers = self.config.get_encoding_bools()
        if encoding_triggers.single_convolution:
            x_tensor_train, x_tensor_test = self.single_convolution_encoding()
            self.train_dataset.append(x_tensor_train)
            self.test_dataset.append(x_tensor_test)
        

        y_tensor_train, y_tensor_test = torch.LongTensor(self.y_train), torch.LongTensor(self.y_test)
        self.train_dataset, self. test_dataset = [TensorDataset(i, y_tensor_train) for i in self.train_dataset], [TensorDataset(i, y_tensor_test) for i in self.test_dataset]
        
        # print(f"The length of train_dataset is {len(self.train_dataset)} ")
        train_dataloader, test_dataloader = [DataLoader(i,batch_size=4, shuffle=True) for i in self.train_dataset], [DataLoader(i, batch_size=4, shuffle=True) for i in self.test_dataset]
        print(f"The length of train_dataloader is {len(train_dataloader)} ")

        logger.info(f"Successfullly converted the encoded data to tensor data loaders")       
        return train_dataloader, test_dataloader
    
    


