from src.Privacy_Encoding.constants import *
from src.Privacy_Encoding.utils.common import create_directories, read_yaml
from src.Privacy_Encoding.entity import *


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

    def get_data_read_config(self) -> DataReadConfig:
        config = self.config.data_read

        data_read_config = DataReadConfig(
            train_dir = config.train_dir,
            test_dir= config.test_dir,
            index_separator= config.index_separator,
            image_id_index = config.image_id_index,
            label_index= config.label_index
        )

        return data_read_config
    
    def get_encoding_bools(self):
        config = self.config.encoding_techniques

        encoding_config = encodingsConfig(
            single_convolution= config.single_convolution,
            double_convolution= config.double_convolution,
            patch_convolution = config.patch_convolution,
            pseudo_differential_privacy = config.pseudo_differential_privacy,
            pseudo_differential_privacy_patched = config.pseudo_differential_privacy_patched,
            differential_privacy = config.differential_privacy,
            differential_privacy_patched = config.differential_privacy_patched,
            differential_privacy_single_convolution = config.differential_privacy_single_convolution
        )

        return encoding_config
    
    def get_save_encoding_images_config(self):
        config  = self.config.save_encoding_images
        save_encoded_images_config = saveEncodedImagesConfig(
            enabled= config.enabled,
            output_dir= config.output_dir
        )
        return save_encoded_images_config
     
    def get_model_initialization_config(self):
        param = self.params
        model_initialization_config = ModelInitializerConfig(
            image_channels= param.image_channels,
            unfreeze_layers= unfreezingModelConfig(
                enabled= param.unfreeze_layers.enabled,
                layers= param.unfreeze_layers.layers
            ),
            learning_rate = param.learning_rate,
            momentum= param.momentum,
            fc_layer_size = param.fc_layer_size
        )
        return model_initialization_config
    
    
