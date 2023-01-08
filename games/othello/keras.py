import os

import numpy as np
from alpha_zero_general import DotDict
from alpha_zero_general import NeuralNet

import tensorflow as tf
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Reshape
from keras.models import Model
from keras.optimizers import Adam

args = DotDict(
    {
        "lr": 0.001,
        "dropout": 0.3,
        "epochs": 1,
        "batch_size": 64,
        "cuda": tf.test.is_gpu_available(),
        "num_channels": 512,
    }
)


class KerasNetWrapper(NeuralNet):
    def __init__(self, game):
        self.args = args
        self.model = self.getModel(
            game.getBoardSize(), game.getActionSize(), self.args,
        )

    @staticmethod
    def getModel(board_size, action_size, args):
        """
        Return compiled tensorflow.keras.models.Model.
        """
        raise NotImplementedError

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        history = self.model.fit(
            x=input_boards,
            y=[target_pis, target_vs],
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=0,
        )
        return history.history["pi_loss"][-1], history.history["v_loss"][-1]

    def predict(self, board):
        """
        board: np array with board
        """
        # preparing input
        board = board[np.newaxis, :, :]

        # run
        pi, v = self.model.predict(board)
        return pi[0], v[0]

    def saveCheckpoint(
        self, folder="checkpoint", filename="checkpoint.pth.tar"
    ):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(
                    folder
                )
            )
            os.mkdir(folder)
        self.model.save_weights(filepath)

    def loadCheckpoint(
        self, folder="checkpoint", filename="checkpoint.pth.tar"
    ):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        self.model.load_weights(filepath)

    def getWeights(self):
        return self.model.get_weights()

    def setWeights(self, weights):
        self.model.set_weights(weights)

    def requestGPU(self):
        return self.args.cuda


class OthelloNNet(KerasNetWrapper):
    @staticmethod
    def getModel(board_size, action_size, args):
        # game params
        board_x, board_y = board_size
        action_size = action_size

        # Neural Net

        # s: batch_size x board_x x board_y
        input_boards = Input(shape=(board_x, board_y))
        # batch_size  x board_x x board_y x 1
        x_image = Reshape((board_x, board_y, 1))(input_boards)
        # batch_size  x board_x x board_y x num_channels
        h_conv1 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(args.num_channels, 3, padding="same", use_bias=False)(
                    x_image
                )
            )
        )
        # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(args.num_channels, 3, padding="same", use_bias=False)(
                    h_conv1
                )
            )
        )
        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv3 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(args.num_channels, 3, padding="valid", use_bias=False)(
                    h_conv2
                )
            )
        )
        # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv4 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(args.num_channels, 3, padding="valid", use_bias=False)(
                    h_conv3
                )
            )
        )
        h_conv4_flat = Flatten()(h_conv4)
        # batch_size x 1024
        s_fc1 = Dropout(args.dropout)(
            Activation("relu")(
                BatchNormalization(axis=1)(
                    Dense(1024, use_bias=False)(h_conv4_flat)
                )
            )
        )
        # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(
            Activation("relu")(
                BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))
            )
        )
        # batch_size x action_size
        pi = Dense(action_size, activation="softmax", name="pi")(s_fc2)
        # batch_size x 1
        v = Dense(1, activation="tanh", name="v")(s_fc2)
        model = Model(inputs=input_boards, outputs=[pi, v])
        model.compile(
            loss=["categorical_crossentropy", "mean_squared_error"],
            optimizer=Adam(args.lr),
        )
        return model
