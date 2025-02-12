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
import numpy as np
import copy


class dataEncodingPipeline():
    def __init__(self,x_train, y_train, x_test, y_test):
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.x_patched_train, self.x_patched_test = [],[]
        self.x_conv_train_1, self.x_conv_test_1 = [],[]
        self.config = ConfigurationManager()

        self.transform = ImageTransformations().get_transform()
        x_tensor_train, x_tensor_test = torch.stack([self.transform(image) for image in x_train]), torch.stack([self.transform(image) for image in x_test])
        self.train_dataset, self.test_dataset = [x_tensor_train], [x_tensor_test]
        self.used_techniques = ["Original"]


    def laplace_mechanism(self,data, epsilon, sensitivity):
        np.random.seed(42)
        scale = sensitivity / epsilon
        # noise = np.random.laplace(0.0, scale, data.shape)
        noise = np.random.laplace(0.0, scale, data.shape).astype(np.float32)
        noisy_image =  data + noise
        # print(noise)
        return noisy_image



    def laplace_mechanism_dynamic(self, data, epsilon):
        new_data = copy.deepcopy(data)
        for i in range(1,len(data)):
            prev, pres = data[i-1], data[i]
            prev_set, pres_set = set(prev), set(pres)
            # jaccard_similarity = len(prev_set.intersection(pres_set)) / len(prev_set.union(pres_set))
            sensitivity = (np.linalg.norm(pres - prev))
            # sensitivity = (np.linalg.norm(pres - prev))/(jaccard_similarity + 0.8)
            scale = sensitivity/epsilon
            noise = np.random.laplace(0.0,scale,data[i-1].shape).astype(np.float32)
            # print(sensitivity, noise.shape)
            new_data[i-1] = data[i-1] + noise
        return new_data


    
    
    def single_convolution_encoding(self):
        encoding_models = encodingModels(self.x_train, self.y_train)
        convolutional_encoder = encoding_models.convolution_encoder(layer = 1)
        logger.info(f"Single Convolutional Encoder initialized successfully")

        self.x_conv_train_1, self.x_conv_test_1 = convolutional_encoder(self.x_train), convolutional_encoder(self.x_test)
        self.x_conv_train_1, self.x_conv_test_1 = np.sum(self.x_conv_train_1, axis = 3), np.sum(self.x_conv_test_1, axis = 3)
        

        # print(self.x_train[0], x_conv_train_1[0])

        logger.info(f"Completed Single Convolution ENcoding successfully. The shape of x_conv_train is {self.x_conv_train_1.shape}")

        
        return self.save_and_convert(self.x_conv_train_1, self.x_conv_test_1, "Single_Convolution_Encoding")
    
    def double_convolution_encoding(self):
        encoding_models = encodingModels(self.x_train, self.y_train)
        convolutional_encoder = encoding_models.convolution_encoder(layer = 2)
        logger.info(f"Double Convolutional Encoder initialized successfully")

        x_conv_train_2, x_conv_test_2 = convolutional_encoder(self.x_train), convolutional_encoder(self.x_test)
        x_conv_train_2, x_conv_test_2 = np.sum(x_conv_train_2, axis = 3), np.sum(x_conv_test_2, axis = 3)
        

        # print(self.x_train[0], x_conv_train_1[0])

        logger.info(f"Completed Double Convolution ENcoding successfully. The shape of x_conv_train is {x_conv_train_2.shape}")

        
        return self.save_and_convert(x_conv_train_2, x_conv_test_2, "Double_Convolution_Encoding")
    
    
    def patch_encoding(self):
        encoding_models = encodingModels(self.x_train, self.y_train)
        convolutional_encoder = encoding_models.patching_encoder()
        logger.info(f"Patching Encoder initialized successfully")

        self.x_patched_train, self.x_patched_test = convolutional_encoder(self.x_train), convolutional_encoder(self.x_test)
        self.x_patched_train, self.x_patched_test = np.sum(self.x_patched_train, axis = 3), np.sum(self.x_patched_test, axis = 3)
        

        # print(self.x_train[0], x_conv_train_1[0])

        logger.info(f"Completed Pathced ENcoding successfully. The shape of x_conv_train is {self.x_patched_train.shape}")

        
        return self.save_and_convert(self.x_patched_train, self.x_patched_test, "Patch_Encoding")

    

    def pseudo_differential_privacy(self):
        x_train_only_dp, x_test_only_dp = [],[]
        for i in range(self.x_train.shape[0]):
            x_train_only_dp.append(self.laplace_mechanism(self.x_train[i],8,1))
        for i in range(self.x_test.shape[0]):
            x_test_only_dp.append(self.laplace_mechanism(self.x_test[i],8,1))


        logger.info(f"Completed Pseudo Differential Privacy Encoding successfully. The shape of x_conv_train is {len(x_train_only_dp)}")

        
        return self.save_and_convert(x_train_only_dp, x_test_only_dp, "Pseudo_Dp")
    

    

    def differential_privacy(self):
        x_dp_train, x_dp_test = [],[]
        for i in range(self.x_train.shape[0]):
            x_dp_train.append(self.laplace_mechanism_dynamic(self.x_train[i],8))

        for i in range(self.x_test.shape[0]):
            x_dp_test.append(self.laplace_mechanism_dynamic(self.x_test[i],8))

        logger.info(f"Completed Differential Privacy Encoding successfully. The shape of x_conv_train is {len(x_dp_test)}")

        
        return self.save_and_convert(x_dp_train, x_dp_test, "Differential_Privacy")
    
    def pseudo_dp_patched(self):
        x_train__pseudo_dp_patched, x_test_pseudo_dp_patched = [],[]
        for i in range(self.x_train.shape[0]):
            x_train__pseudo_dp_patched.append(self.laplace_mechanism(self.x_patched_train[i],8,1))
        for i in range(self.x_test.shape[0]):
            x_test_pseudo_dp_patched.append(self.laplace_mechanism(self.x_patched_test[i],8,1))


        logger.info(f"Completed Pseudo Differential Privacy with Patched Encoding successfully. The shape of x_conv_train is {len(x_train__pseudo_dp_patched)}")

        
        return self.save_and_convert(x_train__pseudo_dp_patched, x_test_pseudo_dp_patched, "Pseudo_Dp_Patched")
    
    def dp_patched(self):
        x_train_dp_patched, x_test_dp_patched = [],[]
        for i in range(self.x_train.shape[0]):
            x_train_dp_patched.append(self.laplace_mechanism_dynamic(self.x_patched_train[i],8))
        for i in range(self.x_test.shape[0]):
            x_test_dp_patched.append(self.laplace_mechanism_dynamic(self.x_patched_test[i],8))


        logger.info(f"Completed Differential Privacy with Patching Encoding successfully. The shape of x_conv_train is {len(x_train_dp_patched)}")

        
        return self.save_and_convert(x_train_dp_patched, x_test_dp_patched, "Dp_Patched")
    

    def dp_single_conv(self):
        x_conv_dp_train, x_conv_dp_test = [],[]
        for i in range(self.x_train.shape[0]):
            x_conv_dp_train.append(self.laplace_mechanism_dynamic(self.x_conv_train_1[i],8))
        for i in range(self.x_test.shape[0]):
            x_conv_dp_test.append(self.laplace_mechanism_dynamic(self.x_conv_test_1[i],8))


        logger.info(f"Completed Differential Privacy with Patching Encoding successfully. The shape of x_conv_train is {len(x_conv_dp_train)}")

        
        return self.save_and_convert(x_conv_dp_train, x_conv_dp_test, "Single_convolution_Dp_Patched")
    
    def convert_to_tensor(self, x_train):
        return torch.stack([self.transform(image) for image in x_train])

    def save_and_convert(self, x_train, x_test, dir):
        save_images_config = self.config.get_save_encoding_images_config()

        if save_images_config.enabled:
            create_directories([save_images_config.output_dir])
            output_dir = os.path.join(Path(save_images_config.output_dir), Path(f"{dir}/Train"))
            save_images_to_directory(output_dir, x_train, self.y_train)
            logger.info(f"Successfully Completed saving {dir} Encoding Images to the directory {output_dir}")


            output_dir = os.path.join(Path(save_images_config.output_dir), Path(f"{dir}/Test"))
            save_images_to_directory(output_dir, x_test, self.y_test)
            logger.info(f"Successfully Completed saving {dir} Encoding Images to the directory {output_dir}")


        

        x_tensor_train, x_tensor_test = self.convert_to_tensor(x_train), self.convert_to_tensor(x_test)
        return x_tensor_train, x_tensor_test
    


    
    def get_encoded_tensors(self):
        encoding_triggers = self.config.get_encoding_bools()
        if encoding_triggers.single_convolution:
            x_tensor_train, x_tensor_test = self.single_convolution_encoding()
            self.train_dataset.append(x_tensor_train)
            self.test_dataset.append(x_tensor_test)
            self.used_techniques.append("Single Convolution")

        if encoding_triggers.double_convolution:
            x_tensor_train, x_tensor_test = self.double_convolution_encoding()
            self.train_dataset.append(x_tensor_train)
            self.test_dataset.append(x_tensor_test)
            self.used_techniques.append("Double Convolution")
        
        if encoding_triggers.patch_convolution:
            x_tensor_train, x_tensor_test = self.patch_encoding()
            self.train_dataset.append(x_tensor_train)
            self.test_dataset.append(x_tensor_test)
            self.used_techniques.append("Patched Convolution")
            print(self.used_techniques)

        if encoding_triggers.pseudo_differential_privacy:
            x_tensor_train, x_tensor_test = self.pseudo_differential_privacy()
            self.train_dataset.append(x_tensor_train)
            self.test_dataset.append(x_tensor_test)
            self.used_techniques.append("Pseudo Differential Privacy")
        
        if encoding_triggers.differential_privacy:
            x_tensor_train, x_tensor_test = self.differential_privacy()
            self.train_dataset.append(x_tensor_train)
            self.test_dataset.append(x_tensor_test)
            self.used_techniques.append("Differential Privacy")

        if encoding_triggers.pseudo_differential_privacy_patched:
            x_tensor_train, x_tensor_test = self.pseudo_dp_patched()
            self.train_dataset.append(x_tensor_train)
            self.test_dataset.append(x_tensor_test)
            self.used_techniques.append("Pseudo Differential Privacy Patched")
        
        if encoding_triggers.differential_privacy_patched:
            x_tensor_train, x_tensor_test = self.dp_patched()
            self.train_dataset.append(x_tensor_train)
            self.test_dataset.append(x_tensor_test)
            self.used_techniques.append("Differential Privacy Patched")

        if encoding_triggers.differential_privacy_single_convolution:
            x_tensor_train, x_tensor_test = self.dp_single_conv()
            self.train_dataset.append(x_tensor_train)
            self.test_dataset.append(x_tensor_test)
            self.used_techniques.append("Differential Privacy with Single Convolution")




        

        y_tensor_train, y_tensor_test = torch.LongTensor(self.y_train), torch.LongTensor(self.y_test)
        self.train_dataset, self. test_dataset = [TensorDataset(i, y_tensor_train) for i in self.train_dataset], [TensorDataset(i, y_tensor_test) for i in self.test_dataset]
        
        # print(f"The length of train_dataset is {len(self.train_dataset)} ")
        train_dataloader, test_dataloader = [DataLoader(i,batch_size=4, shuffle=True) for i in self.train_dataset], [DataLoader(i, batch_size=4, shuffle=True) for i in self.test_dataset]
        print(f"The length of train_dataloader is {len(train_dataloader)} ")

        logger.info(f"Successfullly converted the encoded data to tensor data loaders")       
        return train_dataloader, test_dataloader
    

    def get_used_techniques(self):
        logger.info(f"The list is : {self.used_techniques}")
        return self.used_techniques
    

    


