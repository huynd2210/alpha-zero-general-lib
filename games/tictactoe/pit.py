from alpha_zero_general.arena import Arena
from alpha_zero_general import HumanPlayer  # noqa
from alpha_zero_general import AlphaZeroPlayer
from alpha_zero_general import GreedyPlayer  # noqa
from alpha_zero_general import RandomPlayer  # noqa
from alpha_zero_general import BareModelPlayer

from .game import TicTacToeGame
from .pytorch import TicTacToeNNet

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

game = TicTacToeGame(6)

folder, filename = "./runs/1", "model_00003"

player1 = RandomPlayer(game)
# player1 = GreedyPlayer(game)
# player1 = HumanPlayer(game)
# player1 = BareModelPlayer(game, TicTacToeNNet, folder, filename)
player2 = AlphaZeroPlayer(game, TicTacToeNNet, folder, filename)

arena = Arena(player1, player2, game, display=TicTacToeGame.display)

print(arena.playGames(6, verbose=False))
