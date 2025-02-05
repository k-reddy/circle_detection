import optuna
from typing import Callable

from model_trainer import ModelTrainer
from model_config import DataParams, TrainingParams, ArchitectureParams
from cnn import CNN


def create_objective(param_suggestor: Callable) -> Callable:
    """
    Creates a base objective function for Optuna optimization.

    Args:
        default_hyperparams: Dictionary of default hyperparameters
        param_suggestions: Function that takes (trial, hyperparams) and suggests new parameters
    """

    def objective(trial):
        img_size = 100
        architecture_params, training_params, data_params = param_suggestor(trial)
        # First trials do less data to quickly eliminate bad choices
        if trial.number < 10:
            data_params.num_samples = data_params.num_samples // 2
            training_params.epochs = round(training_params.epochs // 2)
        cnn = CNN(architecture_params, img_size)
        trainer = ModelTrainer(
            cnn, training_params, data_params, hyperparam_run=True, trial=trial
        )
        trainer.train_model()
        # return just the loss, not the IOU
        return trainer.validate_model()[0]

    return objective


def suggest_conv_layer_params(
    trial: optuna.Trial,
) -> tuple[ArchitectureParams, TrainingParams, DataParams]:
    architecture_params = ArchitectureParams(
        cnet_1_channels_out=trial.suggest_categorical(
            "cnet_1_channels_out", [4, 8, 16]
        ),
        cnet_2_channels_out=trial.suggest_categorical("cnet_2_channels_out", [8, 16]),
        cnet_3_channels_out=trial.suggest_categorical(
            "cnet_3_channels_out", [16, 32, 64]
        ),
    )

    # return defaults for training and data
    return architecture_params, TrainingParams(), DataParams()


def suggest_dense_layer_params(
    trial: optuna.Trial,
) -> tuple[ArchitectureParams, TrainingParams, DataParams]:
    architecture_params = ArchitectureParams(
        dense_1_out=trial.suggest_categorical("dense_1_out", [128, 64, 32]),
        dense_2_out=trial.suggest_categorical("dense_2_out", [128, 64, 32]),
    )

    return architecture_params, TrainingParams(), DataParams()


def suggest_batch_size_param(
    trial: optuna.Trial,
) -> tuple[ArchitectureParams, TrainingParams, DataParams]:
    data_params = DataParams(
        batch_size=trial.suggest_categorical("batch_size", [32, 64])
    )
    return ArchitectureParams(), TrainingParams(), data_params


def suggest_learining_params(
    trial: optuna.Trial,
) -> tuple[ArchitectureParams, TrainingParams, DataParams]:
    training_params = TrainingParams(
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        alpha=trial.suggest_float("alpha", 1e-4, 1e-2, log=True),
        gamma=trial.suggest_float("gamma", 0.1, 0.9),
    )
    return ArchitectureParams(), training_params, DataParams()


def suggest_dropout_params(
    trial: optuna.Trial,
) -> tuple[ArchitectureParams, TrainingParams, DataParams]:
    architecture_params = ArchitectureParams(
        p_dropout_1=trial.suggest_float("p_dropout_1", 0.0, 0.3),
        p_dropout_2=trial.suggest_float("p_dropout_2", 0.0, 0.3),
        p_dropout_3=trial.suggest_float("p_dropout_3", 0.0, 0.3),
    )
    return architecture_params, TrainingParams(), DataParams()


def run_study(objective_creator: Callable, param_suggestor: Callable) -> None:
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(),
    )
    objective = objective_creator(param_suggestor)
    study.optimize(objective, n_trials=21, n_jobs=3)
    print("Best trial:")
    print("  IOU: ", study.best_trial.value)
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_param_importances(study)


# dense1, dense2 outs
# run_study(create_objective, suggest_dense_layer_params)
# cnet1, cnet2, cnet3 outs
# run_study(create_objective, suggest_conv_layer_params)

# # batch size
# run_study(create_objective, suggest_batch_size_param)
# # alpha, gamma
# run_study(create_objective, suggest_learining_params)
# # dropouts and weight decay
run_study(create_objective, suggest_dropout_params)

# do these manually at the end
manually_fiddle = ["noise_level", "crop", "epochs", "batch_size"]
