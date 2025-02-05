import torch.nn as nn
from torch import Tensor
from model_config import ArchitectureParams


class CNN(nn.Module):
    """
    A pytorch CNN that processes images.
    Architecture notes:
        - 3 convolution blocks (conv layer, batch norm, relu activation)
        - dropout
        - 2 conv blocks with dilation
        - dropout
        - 2 fully connected layers with batch norm and relu activation
        - a final fully connected layer that predicts 3 values
    Model requires image to be 2D, grayscale (1 channel), and square.

    It allows you to specify many hyperparameters in initialization and training
    """

    def __init__(self, params: ArchitectureParams, img_size: int):
        """
        Args:
            - img_size: size of image, model assumes 2D, grayscale (1 channel), and square
            - params: an ArchitectureParams object that specifies
                - num output channels for different layers
                - p(dropout) for dropout layers
        """
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            # kernel size 3 for all conv layers
            self.convolution_block(1, params.cnet_1_channels_out, 3),
            self.convolution_block(
                params.cnet_1_channels_out, params.cnet_1_channels_out, 3
            ),
            self.convolution_block(
                params.cnet_1_channels_out, params.cnet_2_channels_out, 3
            ),
            nn.Dropout(p=params.p_dropout_1),
            self.convolution_block(
                params.cnet_2_channels_out,
                params.cnet_3_channels_out,
                3,
                dilation=2,
            ),
            self.convolution_block(
                params.cnet_3_channels_out,
                params.cnet_3_channels_out,
                3,
                stride=2,
            ),
            nn.Dropout(p=params.p_dropout_2),
            nn.Flatten(),
            self.dense_block(
                img_size * img_size * params.cnet_3_channels_out // 4,
                params.dense_1_out,
            ),
            # removing the 3rd dropout layer based on hyperparameter optimization
            # nn.Dropout(p=params.p_dropout_3),
            self.dense_block(params.dense_1_out, params.dense_2_out),
            nn.Linear(params.dense_2_out, 3),
        )
        self.img_size = img_size
        self.architecture_params = params
        num_parameters = sum(p.numel() for p in self.parameters())
        print(f"Model created with {num_parameters} parameters")

    def convolution_block(
        self,
        convolution_in_channels: int,
        convolution_out_channels: int,
        convolution_kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
    ) -> nn.Sequential:
        """
        Returns a Sequential with the following layers:
            conv -> batchnorm -> ReLU

        args:
            - convolution_in_channels: num channels going into the conv layer
            - convolution_out_channels: num channels going out of the conv layer
            - convolution_kernel_size: kernel size for the conv layer
            - dilation: dilation for the conv layer, defaults to no dilation
        """
        return nn.Sequential(
            nn.Conv2d(
                convolution_in_channels,
                convolution_out_channels,
                kernel_size=(convolution_kernel_size, convolution_kernel_size),
                stride=stride,
                dilation=dilation,
                padding=dilation,
            ),
            nn.BatchNorm2d(convolution_out_channels),
            nn.ReLU(),
        )

    def dense_block(self, features_in: int, features_out: int) -> nn.Sequential:
        """
        Returns a Sequential with the following layers:
            linear -> batchnorm -> relu

        args:
            - features_in: number of input features for the linear layer
            - features_out: number of output features for the linear layer
        """
        return nn.Sequential(
            nn.Linear(features_in, features_out),
            nn.BatchNorm1d(features_out),
            nn.ReLU(),
        )

    def forward(self, data: Tensor):
        # ensure the data is single channel and image size is as expected
        assert data.shape[1] == 1
        assert data.shape[2:] == (self.img_size, self.img_size)

        return self.model(data)
