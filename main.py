from src.Privacy_Encoding.logging import logger
from src.Privacy_Encoding.pipeline.stage_01_read_image_data import dataReadingTrainingPipeline
from src.Privacy_Encoding.components.encoding_models import encodingModels
from src.Privacy_Encoding.pipeline.stage_02_image_encoding import dataEncodingPipeline
from src.Privacy_Encoding.pipeline.stage_03_model_initialization import modelInitializationPipeline
from src.Privacy_Encoding.pipeline.stage_04_model_training import modelTrainerPipeline


train_dataloader, test_dataloader, no_of_labels = [], [], 0
used_techniques = ["Original"]


def data_read():
    pipleline_Stage_name = "Data Reading Stage"
    
    try:
        logger.info(f"Starting {pipleline_Stage_name}")

        data_read = dataReadingTrainingPipeline()
        (x_train,y_train), (x_test,y_test), label_count = data_read.get_data()


        logger.info(f"{pipleline_Stage_name} completed successfully")
        logger.info(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

        return x_train, y_train, x_test, y_test, label_count
    except Exception as e:
        logger.exception(e)
        raise e


def data_encoding():
    pipeline_stage_name = "Data Encoding Stage"
    
    try:

        global train_dataloader 
        global test_dataloader
        global no_of_labels
        global used_techniques

        x_train, y_train, x_test, y_test , label_count = data_read()
        no_of_labels = len(label_count)

        data_encode = dataEncodingPipeline(x_train, y_train, x_test,y_test)
        
        train_dataloader, test_dataloader = data_encode.get_encoded_tensors()
        used_techniques = data_encode.get_used_techniques()

        logger.info(f"completed appending the new encoded datasets. ")
        return train_dataloader, test_dataloader
    except Exception as e:
        logger.exception(e)
        raise e
    
def model_initialization():
    pipeline_stage_name = "Model Initialization Stage"
    
    try:
        logger.info(f"Starting {pipeline_stage_name}................................................................")
        print(no_of_labels)
        model_initializer = modelInitializationPipeline(no_of_labels)
        model, lossfun, optimizer = model_initializer.initialize_model()
        logger.info(f"Completed {pipeline_stage_name}................................................................")
        return model, lossfun, optimizer
    
    except Exception as e:
        logger.exception(e)
        raise e
    
def model_trainer():
    pipeline_stage_name = "Model Training Stage"

    logger.info(f"Started Running {pipeline_stage_name}")
    data_encoding()
    model,lossfun, optimizer = model_initialization()
    for i in range(len(train_dataloader)):
        model_trainer = modelTrainerPipeline(model, train_dataloader[i], test_dataloader[i], lossfun, optimizer)
        model_trainer.train_model()
        logger.info(f"Completed Model Training in the technique {used_techniques[i]}-----------------------------------------")
    print(used_techniques)
                


model_trainer()

