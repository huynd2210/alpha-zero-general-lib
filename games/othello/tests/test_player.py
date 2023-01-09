from unittest.mock import patch

import pytest
from alpha_zero_general import AlphaZeroPlayer
from alpha_zero_general import BareModelPlayer
from alpha_zero_general import GreedyPlayer
from alpha_zero_general import HumanPlayer
from alpha_zero_general import RandomPlayer

from ..game import OthelloGame
from ..keras import OthelloNNet


def testAlphaZeroPlayer():
    game = OthelloGame(6)
    player = AlphaZeroPlayer(game, OthelloNNet, num_mcts_sims=4)
    board = game.getInitBoard()
    action = player.play(board)
    assert action


def testAlphaZeroPlayerFromCheckpoint():
    game = OthelloGame(6)

    some_net = OthelloNNet(game)
    folder, filename = "/tmp/", "checkpoint_alpha_zero_player"
    some_net.saveCheckpoint(folder, filename)
    del some_net

    player = AlphaZeroPlayer(
        game, OthelloNNet, folder=folder, filename=filename, num_mcts_sims=4
    )
    board = game.getInitBoard()
    action = player.play(board)
    assert action


def testAlphaZeroPlayerFromModel():
    game = OthelloGame(6)
    some_net = OthelloNNet(game)
    player = AlphaZeroPlayer(game, some_net)
    board = game.getInitBoard()
    action = player.play(board)
    assert action


def testAlphaZeroPlayerReset():
    game = OthelloGame(6)
    player = AlphaZeroPlayer(game, OthelloNNet, num_mcts_sims=4)
    assert not player.mcts.Qsa
    board = game.getInitBoard()
    player.play(board)
    assert player.mcts.Qsa
    player.reset()
    assert not player.mcts.Qsa


def testBareModelPlayer():
    game = OthelloGame(6)
    player = BareModelPlayer(game, OthelloNNet)
    board = game.getInitBoard()
    action = player.play(board)
    assert action


def testBareModelPlayerFromCheckpoint():
    game = OthelloGame(6)

    some_net = OthelloNNet(game)
    folder, filename = "/tmp/", "checkpoint_bare_model_player"
    some_net.saveCheckpoint(folder, filename)
    del some_net

    player = BareModelPlayer(
        game, OthelloNNet, folder=folder, filename=filename
    )
    board = game.getInitBoard()
    action = player.play(board)
    assert action


def testBareModelPlayerFromModel():
    game = OthelloGame(6)
    some_net = OthelloNNet(game)
    player = BareModelPlayer(game, some_net)
    board = game.getInitBoard()
    action = player.play(board)
    assert action


def testBareModelPlayerInvallidNNetParameter():
    neither_nnet_nor_nnet_class = "something_else"
    game = OthelloGame(6)
    with pytest.raises(TypeError) as excinfo:
        BareModelPlayer(game, neither_nnet_nor_nnet_class)
    assert "NeuralNet subclass or instance" in str(excinfo.value)


def testGreedyPlayer():
    game = OthelloGame(6)
    player = GreedyPlayer(game)
    board = game.getInitBoard()
    action = player.play(board)
    assert action


def testRandomPlayer():
    game = OthelloGame(6)
    player = RandomPlayer(game)
    board = game.getInitBoard()
    action = player.play(board)
    assert action


def testHumanPlayerValidAction():
    action_valid = "1,2"
    game = OthelloGame(6)
    player = HumanPlayer(game)
    board = game.getInitBoard()
    with patch("builtins.input", side_effect=[action_valid]):
        action = player.play(board)
    assert action == 8


def testHumanPlayerUnknownAction(capsys):
    action_valid = "1,2"
    action_unknown = "x,x"
    game = OthelloGame(6)
    player = HumanPlayer(game)
    board = game.getInitBoard()
    with patch("builtins.input", side_effect=[action_unknown, action_valid]):
        player.play(board)
    out, _err = capsys.readouterr()
    assert out == "Unknown action.\n"


def testHumanPlayerInvalidAction(capsys):
    action_valid = "1,2"
    action_invalid = "0,0"
    game = OthelloGame(6)
    player = HumanPlayer(game)
    board = game.getInitBoard()
    with patch("builtins.input", side_effect=[action_invalid, action_valid]):
        player.play(board)
    out, _err = capsys.readouterr()
    assert out == "Invalid action.\n"
