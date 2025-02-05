from dataclasses import dataclass


@dataclass
class ArchitectureParams:
    cnet_1_channels_out: int = 16
    cnet_2_channels_out: int = 8
    cnet_3_channels_out: int = 16
    dense_1_out: int = 128
    dense_2_out: int = 128
    p_dropout_1: float = 0.129
    p_dropout_2: float = 0.176
    p_dropout_3: float = 0.0  # model suggested .015, so removing this entirely


@dataclass
class DataParams:
    noise_level: float = 0.5
    crop: bool = False
    batch_size: int = 32
    num_samples: int = 50000


@dataclass
class TrainingParams:
    weight_decay: float = 1.5e-05
    momentum: float = 0.9
    alpha: float = 0.009
    gamma: float = 0.62
    epochs: int = 5


# Best model so far
# @dataclass
# class ArchitectureParams:
#     cnet_1_channels_out: int = 8
#     cnet_2_channels_out: int = 16
#     cnet_3_channels_out: int = 32
#     dense_1_out: int = 64
#     dense_2_out: int = 128
#     p_dropout_1: float = 0.1
#     p_dropout_2: float = 0.2
#     p_dropout_3: float = 0.3


# @dataclass
# class DataParams:
#     noise_level: float = 0.5
#     crop: bool = False
#     batch_size: int = 64
#     num_samples: int = 50000


# @dataclass
# class TrainingParams:
#     weight_decay: float = 1e-4
#     momentum: float = 0.9
#     alpha: float = 0.003
#     gamma: float = 0.5
#     epochs: int = 5
