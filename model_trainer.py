import optuna
import os
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.optim import Adam
from cnn import CNN
from model_config import DataParams, TrainingParams
from circle_dataset import CircleDataset
from circle_detection import count_ious_over_thresholds


class ModelTrainer:
    def __init__(
        self,
        model: CNN,
        training_params: TrainingParams,
        data_params: DataParams,
        hyperparam_run: bool = False,
        trial: optuna.Trial = None,
    ):
        """
        Args:
            - training_dataloader: a Dataloader object with our training data
            - training_params: a TrainingParams object that specifies number of
                epochs, weight decay, momentum (for SGD), alpha, and gamma
        """
        if hyperparam_run and trial is None:
            raise ValueError("Hyperparam run requires a trial")

        self.model = model
        self.training_params = training_params
        self.data_params = data_params
        self.hyperparam_run = hyperparam_run
        self.trial = trial
        self.training_losses: list = []
        self.validation_losses: list = []

        self.training_dataloader = None

        self.loss_fn = nn.MSELoss()

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.training_params.alpha,
            weight_decay=self.training_params.weight_decay,
        )

        # self.optimizer = SGD(
        #         self.model.parameters(),
        #         lr=self.training_params.alpha,
        #         momentum=self.training_params.momentum,
        #         weight_decay= self.training_params.weight_decay,
        #     )

        self.learning_rate_scheduler = StepLR(
            self.optimizer, step_size=5, gamma=self.training_params.gamma
        )

    def train_model(
        self,
    ) -> None:
        """
        Trains the CNN according to parameters passed in initialization
        """
        self.model.train()

        for epoch in range(self.training_params.epochs):
            # use new data for every epoch
            training_dataset = CircleDataset(
                self.data_params.num_samples,
                self.data_params.noise_level,
            )

            self.training_dataloader = DataLoader(
                training_dataset,
                batch_size=self.data_params.batch_size,
                shuffle=True,
            )

            avg_loss, pct_iou_over_thresholds = self.train_epoch()
            validation_loss, validation_pct_iou_over_thresholds = self.validate_model()

            self.report_epoch(
                epoch,
                avg_loss,
                pct_iou_over_thresholds,
                validation_loss,
                validation_pct_iou_over_thresholds,
            )
            self.learning_rate_scheduler.step()
            if not self.hyperparam_run:
                self.save_model(epoch)

    def report_epoch(
        self,
        epoch: int,
        avg_loss: float,
        pct_iou_over_thresholds: dict,
        validation_loss: float,
        validation_pct_iou_over_thresholds: dict,
    ) -> None:
        """
        Prints the epoch results
        Reports them to optuna if this is an optuna trial
        Saves them for plotting later
        """
        self.training_losses.append(avg_loss)
        self.validation_losses.append(validation_loss)

        print(
            f"Epoch: {epoch+1}/{self.training_params.epochs}",
            f"Avg Loss: {avg_loss}, Validation loss: {validation_loss}",
        )
        self.print_iou_string(
            pct_iou_over_thresholds, validation_pct_iou_over_thresholds
        )

        if self.hyperparam_run:
            self.trial.report(validation_loss, epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned()

    def print_iou_string(self, training_ious: dict, validation_ious: dict) -> None:
        print("IOUs: TRAINING / VALIDATION")
        for threshold, pct in training_ious.items():
            print(
                f"Over {threshold}: {round(pct*100)}% / {round(validation_ious[threshold]*100)}%"
            )

    def train_epoch(self) -> tuple[float, dict]:
        """
        Runs one rep of training on all the data

        Returns avg_loss, pct_iou_over_threshold for the whole epoch
        """
        epoch_loss = 0.0
        ious_over_thresholds = {0.1: 0, 0.25: 0, 0.5: 0, 0.75: 0, 0.9: 0, 0.95: 0}

        for batch_idx, (inputs, targets) in enumerate(self.training_dataloader):
            batch_loss, batch_iou_over_thresholds = self.train_batch(inputs, targets)
            if batch_idx % 50 == 0:  # Print every 50 batches
                print(f"Batch: {batch_idx}, Avg Loss: {batch_loss}")
            for threshold, count in batch_iou_over_thresholds.items():
                ious_over_thresholds[threshold] += count
            epoch_loss += batch_loss

        avg_loss = epoch_loss / len(self.training_dataloader)
        pct_iou_over_thresholds = {
            k: v / self.data_params.num_samples for k, v in ious_over_thresholds.items()
        }
        return avg_loss, pct_iou_over_thresholds

    def train_batch(self, inputs, targets) -> tuple[float, dict]:
        """
        Runs one rep of training for a batch

        returns loss, n_iou_over_threshold for the batch
        """
        # zero the gradients
        self.optimizer.zero_grad()
        # forward pass
        predictions = self.model.forward(inputs)
        # calculate loss
        loss = self.loss_fn(predictions, targets)
        # backward pass
        loss.backward()
        # update weights
        self.optimizer.step()

        with torch.no_grad():
            batch_iou_over_thresholds = count_ious_over_thresholds(
                predictions,
                targets,
            )
        return loss.item(), batch_iou_over_thresholds

    def validate_model(self) -> tuple[float, dict]:
        """
        Evaluates the model on validation data (20% as much as training data)
        Returns the average validation loss, pct_iou_over_threshold for the batch
        """
        # set up the data
        validation_dataset = CircleDataset(
            round(self.data_params.num_samples * 0.2),
            self.data_params.noise_level,
        )
        validation_dataloader = DataLoader(
            validation_dataset, batch_size=self.data_params.batch_size, shuffle=True
        )

        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0
        ious_over_thresholds = {0.1: 0, 0.25: 0, 0.5: 0, 0.75: 0, 0.9: 0, 0.95: 0}

        with torch.no_grad():
            for inputs, targets in validation_dataloader:
                # Forward pass
                predictions = self.model.forward(inputs)

                # Calculate loss
                loss = self.loss_fn(predictions, targets)
                total_loss += loss.item()
                batch_iou_over_thresholds = count_ious_over_thresholds(
                    predictions,
                    targets,
                )
                for threshold, count in batch_iou_over_thresholds.items():
                    ious_over_thresholds[threshold] += count

        avg_loss = total_loss / len(validation_dataloader)
        pct_iou_over_thresholds = {
            k: v / len(validation_dataset) for k, v in ious_over_thresholds.items()
        }
        self.model.train()  # Set the model back to training mode
        return avg_loss, pct_iou_over_thresholds

    def save_model(self, epoch: int):
        base_path = "models/"
        os.makedirs(base_path, exist_ok=True)
        i = 0
        while os.path.exists(os.path.join(base_path, f"model_{i}_epoch_{epoch}.pth")):
            i += 1

        # Create file paths
        model_path = os.path.join(base_path, f"model_{i}_epoch_{epoch}.pth")
        trainer_path = os.path.join(base_path, f"model_trainer_{i}_epoch_{epoch}.pth")

        torch.save(self.model, model_path)
        torch.save(self, trainer_path)
