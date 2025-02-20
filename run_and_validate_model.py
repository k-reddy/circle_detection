from torch import load
from model_config import DataParams, TrainingParams, ArchitectureParams
from model_trainer import ModelTrainer
from cnn import CNN

if __name__ == "__main__":
    load_model = True
    if not load_model:
        # use the default settings
        data_params = DataParams()
        cnn = CNN(ArchitectureParams(), img_size=50)
        trainer = ModelTrainer(cnn, TrainingParams(), data_params)
        trainer.train_model()
        # test the model on a new dataset
        trainer.validate_model()
    else:
        cnn = load("models/model_0_epoch_24.pth")
        trainer = load("models/model_trainer_0_epoch_24.pth")
        # further train the model
        trainer.training_params.epochs = 25
        trainer.train_model()
        # test the model on a new dataset
        trainer.validate_model()
