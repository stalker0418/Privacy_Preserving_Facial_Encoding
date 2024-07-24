from src.Privacy_Encoding.components.faceRecognitionModel import FRModel
from src.Privacy_Encoding.config.configuration import ConfigurationManager
from src.Privacy_Encoding.logging import logger

class modelInitializationPipeline:
    def __init__(self, no_of_labels):
        self.no_of_labels = no_of_labels

    def initialize_model(self):
        config = ConfigurationManager()
        model_initialization_config = config.get_model_initialization_config()
        model = FRModel(self, model_initialization_config)
        model.set_first_layer()
        model.unfreeze_layers()
        model.create_fc_layer(self.no_of_labels)
        loss, optimizer = model.get_loss_and_optimizer()
        logger.info(f"Facial Recognition Model Intitialized successfully.")
        return model, loss, optimizer
    


        


