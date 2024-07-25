from src.Privacy_Encoding.logging import logger
from src.Privacy_Encoding.pipeline.stage_01_read_image_data import dataReadingTrainingPipeline
from src.Privacy_Encoding.components.encoding_models import encodingModels
from src.Privacy_Encoding.pipeline.stage_02_image_encoding import dataEncodingPipeline
from src.Privacy_Encoding.pipeline.stage_03_model_initialization import modelInitializationPipeline
from src.Privacy_Encoding.components.model_trainer import modelTrainer
from src.Privacy_Encoding.components.transforms import ImageTransformations
import numpy as np
import torch

train_dataset, test_dataset, no_of_labels = [], [], 0


def data_read():
    pipleline_Stage_name = "Data Reading Stage"
    
    try:
        logger.info(f"Starting {pipleline_Stage_name}")

        data_read = dataReadingTrainingPipeline()
        (x_train,y_train), (x_test,y_test), label_count = data_read.get_data()


        logger.info(f"{pipleline_Stage_name} completed successfully")
        logger.info(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")


        train_dataset.append(x_train)
        test_dataset.append(x_test)
        return x_train, y_train, x_test, y_test, label_count
    except Exception as e:
        logger.exception(e)
        raise e


def data_encoding():
    pipeline_stage_name = "Data Encoding Stage"
    
    try:

        transform = ImageTransformations()
        x_train, y_train, x_test, y_test , label_count = data_read()
        no_of_labels = len(label_count)

        data_encode = dataEncodingPipeline(x_train, y_train, x_test,y_test)
        
        train_dataset, test_dataset = data_encode.get_encoded_tensors()

        

        # data_encode = dataEncodingPipeline(x_train, y_train)
        # x_conv_train_1 = data_encode.single_convolution_encoding("Train")
        # logger.info(f"Completed Single Convolution Encoding of Train Dataset.")


        # data_encode = dataEncodingPipeline(x_test, y_test)
        # x_conv_test_1 = data_encode.single_convolution_encoding("Test")
        # logger.info(f"Completed Single Convolution Encoding of Test Dataset")

        
        # x_tensor_train, x_tensor_test = torch.stack([transform(image) for image in i[0]]), torch.stack([transform(image) for image in i[1]])
        # y_tensor_train, y_tensor_test = torch.LongTensor(y_train), torch.LongTensor(y_test)
        # train_dataset.append()
        # test_dataset.append(x_conv_test_1)

        

        logger.info(f"completed appending the new encoded datasets. The shape is of Train is :{train_dataset[1].shape}\n, Test is :{len(test_dataset)}")
        
    except Exception as e:
        logger.exception(e)
        raise e
    
def model_initialization():
    pipeline_stage_name = "Model Initialization Stage"
    
    try:
        logger.info(f"Starting {pipeline_stage_name}................................................................")
        model_initializer = modelInitializationPipeline(no_of_labels)
        model, lossfun, optimizer = model_initializer.initialize_model()
        logger.info(f"Completed {pipeline_stage_name}................................................................")
        return model, lossfun, optimizer
    
    except Exception as e:
        logger.exception(e)
        raise e
        
# model,lossfun,optimizer = model_initialization()

data_encoding()

