
from src.Privacy_Encoding.config.configuration import ConfigurationManager
from src.Privacy_Encoding.components.data_read import DataRead


class dataReadingTrainingPipeline:
    def __init__(self):
        pass

    def get_data(self):
        config = ConfigurationManager()
        data_read_config = config.get_data_read_config()
        data_read = DataRead(config = data_read_config)
        x_train, y_train, label_count = data_read.read_images(data_read_config.train_dir)
        x_test, y_test, _ = data_read.read_images(data_read_config.test_dir)
        x_train = data_read.normalize_dataset(x_train)
        x_test = data_read.normalize_dataset(x_test)
        return (x_train, y_train), (x_test, y_test), label_count
        