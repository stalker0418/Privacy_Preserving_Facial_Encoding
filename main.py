from src.Privacy_Encoding.logging import logger
from src.Privacy_Encoding.pipeline.stage_01_read_image_data import dataReadingTrainingPipeline
from src.Privacy_Encoding.components.encoding_models import encodingModels
from src.Privacy_Encoding.pipeline.stage_02_image_encoding import dataEncodingPipeline
import numpy as np

train_dataset, test_dataset = [], []


def data_read():
    pipleline_Stage_name = "Data Reading Stage"
    
    try:
        logger.info(f"Starting {pipleline_Stage_name}")
        data_read = dataReadingTrainingPipeline()
        (x_train,y_train), (x_test,y_test) = data_read.get_data()
        logger.info(f"{pipleline_Stage_name} completed successfully")
        logger.info(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
        train_dataset.append(x_train)
        test_dataset.append(x_test)
        # train_dataset = np.array(train_dataset)
        return x_train, y_train, x_test, y_test
    except Exception as e:
        logger.exception(e)
        raise e


def data_encoding():
    pipeline_stage_name = "Data Encoding Stage"
    
    try:
        x_train, y_train, x_test, y_test = data_read()
        

        data_encode = dataEncodingPipeline(x_train, y_train)
        x_conv_train_1 = data_encode.single_convolution_encoding("Train")
        logger.info(f"Completed Single Convolution Encoding of Train Dataset.")


        data_encode = dataEncodingPipeline(x_test, y_test)
        x_conv_test_1 = data_encode.single_convolution_encoding("Test")
        logger.info(f"Completed Single Convolution Encoding of Test Dataset")

        train_dataset.append(x_conv_train_1)
        test_dataset.append(x_conv_test_1)

        logger.info(f"completed appending the new encoded datasets. The shape is of Train is :{len(train_dataset)}\n, Test is :{len(test_dataset)}")
        
    except Exception as e:
        logger.exception(e)
        raise e
    


data_encoding()

