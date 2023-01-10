"""An abstract base class for the neural network to be used by alpha zero."""

from abc import ABC
from abc import abstractmethod


class NeuralNet(ABC):
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board.
    """

    @abstractmethod
    def __init__(self, game):
        """
        Initialize the net with the game.
        """

    @abstractmethod
    def train(self, examples):
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples has
                      board in its canonical form.

        Returns:
            pi_loss, v_loss: The losses after training the model using the examples.
        """

    @abstractmethod
    def predict(self, board):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.get_action_size
            v: a float in [-1,1] that gives the value of the current board
        """

    @abstractmethod
    def saveCheckpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """

    @abstractmethod
    def loadCheckpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """

    @abstractmethod
    def getWeights(self):
        """
        Returns the weights of the neural network.
        """

    @abstractmethod
    def setWeights(self, weights):
        """
        Sets the given weights to the neural network.
        """

    @abstractmethod
    def requestGPU(self):
        """
        Returns True if a gpu should be used, otherwise False.
        """
