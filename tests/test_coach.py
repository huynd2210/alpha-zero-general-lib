import os
import random

import ray
from alpha_zero_general import Coach
from alpha_zero_general import DotDict
from alpha_zero_general.coach import ModelTrainer
from alpha_zero_general.coach import ReplayBuffer
from alpha_zero_general.coach import SelfPlay
from alpha_zero_general.coach import SharedStorage

from games.othello.game import OthelloGame
from games.othello.keras import OthelloNNet

args = DotDict(
    {
        "numIters": 2,
        "numEps": 2,  # Number of complete self-play games to simulate during a new iteration.
        "tempThreshold": 15,  #
        "updateThreshold": 0.6,  # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        "maxlenOfQueue": 10,  # Number of game examples to train the neural networks.
        "numMCTSSims": 2,  # Number of games moves for MCTS to simulate.
        "arenaCompare": 2,  # Number of games to play during arena play to determine if new net will be accepted.
        "cpuct": 1,
        "checkpoint": "/tmp/alpha_zero_general/",
        "load_model": False,
        "load_folder_file": ("/tmp/alpha_zero_general/", "best.pth.tar"),
        "numItersForTrainExamplesHistory": 20,
        "nr_actors": 2,  # Number of self play episodes executed in parallel
    }
)


def testSharedStorage(local_ray):
    init_weights = [0, 0]
    init_revision = 1
    s = SharedStorage.remote(init_weights, revision=init_revision)
    assert ray.get(s.getRevision.remote()) == init_revision
    assert ray.get(s.getWeights.remote()) == (init_weights, init_revision)
    next_weights = [1, 1]
    next_revision = ray.get(s.setWeights.remote(next_weights, 0.5, 0.2))
    assert next_revision == init_revision + 1
    assert ray.get(s.getWeights.remote()) == (next_weights, next_revision)
    assert ray.get(s.getInfos.remote()) == {
        "trained_enough": False,
        "policy_loss": 0.5,
        "value_loss": 0.2,
    }
    assert ray.get(s.getWeights.remote(revision=next_revision + 1)) == (
        None,
        next_revision,
    )
    ray.get(s.setInfo.remote("trained_enough", True))
    assert ray.get(s.trainedEnough.remote()) is True


def testReplayBuffer(local_ray, tmpdir):
    def mock_game_examples(game=1, size=10):
        return [game] * size

    r = ReplayBuffer.remote(games_to_use=5, folder=tmpdir)
    assert ray.get(r.getNumberGamesPlayed.remote()) == 0
    game_1 = mock_game_examples(game=1)
    r.addExamples.remote(game_1)
    assert ray.get(r.getNumberGamesPlayed.remote()) == 1
    assert os.path.isfile(os.path.join(tmpdir, f"game_{1:08d}"))
    assert ray.get(ray.get(r.getExamples.remote())) == [game_1]
    for game in range(2, 7):
        r.addExamples.remote(mock_game_examples(game=game))
    assert ray.get(r.getNumberGamesPlayed.remote()) == 6
    games = ray.get(ray.get(r.getExamples.remote()))
    assert len(games) == 5
    assert games[0][0] == 2
    assert games[-1][0] == 6
    assert os.path.isfile(os.path.join(tmpdir, f"game_{6:08d}"))

    r = ReplayBuffer.remote(games_to_use=5, folder=tmpdir)
    assert ray.get(r.load.remote()) == 6
    games = ray.get(ray.get(r.getExamples.remote()))
    assert len(games) == 5
    assert games[0][0] == 2
    assert games[-1][0] == 6

    r = ReplayBuffer.remote(games_to_play=5, games_to_use=5, folder=tmpdir)
    assert ray.get(r.load.remote()) == 6
    assert ray.get(r.playedEnough.remote()) is True

    r = ReplayBuffer.remote(games_to_play=10, games_to_use=5, folder=tmpdir)
    assert ray.get(r.load.remote()) == 6
    assert ray.get(r.playedEnough.remote()) is False


def testSelfPlay(local_ray, tmpdir):
    game = OthelloGame(6)
    nnet = OthelloNNet(game)
    s = SharedStorage.remote(nnet.getWeights())
    r = ReplayBuffer.remote(games_to_play=1, games_to_use=1, folder=tmpdir)
    assert ray.get(r.getNumberGamesPlayed.remote()) == 0
    self_play = SelfPlay.remote(r, s, game, nnet.__class__, dict(args))
    ray.get(self_play.start.remote())
    assert ray.get(r.getNumberGamesPlayed.remote()) == 1
    assert ray.get(r.playedEnough.remote()) is True
    games = ray.get(ray.get(r.getExamples.remote()))
    assert len(games) == 1
    examples = games[0]
    assert len(examples) > 2
    board, policy, winner = examples[0]
    assert isinstance(board, type(game.getInitBoard()))
    assert len(policy) == game.getActionSize()
    assert all(0 <= value <= 1 for value in policy)
    assert winner in [1, -1]


def mockExampleData(game):
    board = game.getInitBoard()
    pi = [random.random() for _ in range(game.getActionSize())]
    player = random.choice([1, -1])
    return [(b, p, player) for b, p in game.getSymmetries(board, pi)]


@ray.remote
class MockedReplayBuffer(ReplayBuffer.__ray_actor_class__):  # type: ignore
    """A replay buffer that behaves so that we'll go through all branches
    of ModelTrainer.start()."""

    played_enough_return_values = [False, False, False, True]

    def playedEnough(self):
        """Returns preset values useful in this test."""
        return self.played_enough_return_values.pop(0)

    games_played_return_values = [0, 2, 4, 8]

    def getNumberGamesPlayed(self):
        """Returns preset values useful in this test."""
        return self.games_played_return_values.pop(0)


def testModelTrainerLoop(local_ray, tmpdir):
    game = OthelloGame(6)
    nnet = OthelloNNet(game)
    s = SharedStorage.remote(nnet.getWeights())
    assert ray.get(s.getRevision.remote()) == 0
    r = MockedReplayBuffer.remote(
        games_to_play=4, games_to_use=4, folder=tmpdir
    )
    r.addExamples.remote(mockExampleData(game))

    model_trainer = ModelTrainer.options(num_gpus=0).remote(
        r, s, game, nnet.__class__, dict(args), selfplay_training_ratio=1
    )
    ray.get(model_trainer.start.remote())
    assert ray.get(s.getRevision.remote()) > 0
    assert ray.get(s.trainedEnough.remote()) is True


def testModelTrainerPitAcceptModel(capsys, local_ray, tmpdir):
    game = OthelloGame(6)
    nnet = OthelloNNet(game)
    s = SharedStorage.remote(nnet.getWeights())
    assert ray.get(s.getRevision.remote()) == 0
    r = ReplayBuffer.remote(games_to_play=2, games_to_use=2, folder=tmpdir)
    r.addExamples.remote(mockExampleData(game))
    # provoke model acceptance by tweaking updateThreshold to pass
    custom_args = dict(args, updateThreshold=-0.1)
    model_trainer = ModelTrainer.options(num_gpus=0).remote(
        r, s, game, nnet.__class__, custom_args, pit_against_old_model=True
    )
    ray.get(model_trainer.train.remote())
    assert ray.get(s.getRevision.remote()) == 1
    out, _err = capsys.readouterr()
    assert "PITTING AGAINST PREVIOUS VERSION" in out
    assert "ACCEPTING NEW MODEL" in out


def testModelTrainerPitRejectModel(capsys, local_ray, tmpdir):
    game = OthelloGame(6)
    nnet = OthelloNNet(game)
    s = SharedStorage.remote(nnet.getWeights())
    assert ray.get(s.getRevision.remote()) == 0
    r = ReplayBuffer.remote(games_to_play=2, games_to_use=2, folder=tmpdir)
    r.addExamples.remote(mockExampleData(game))
    # provoke model rejection by tweaking updateThreshold to fail
    custom_args = dict(args, updateThreshold=1.1)
    model_trainer = ModelTrainer.options(num_gpus=0).remote(
        r, s, game, nnet.__class__, custom_args, pit_against_old_model=True
    )
    ray.get(model_trainer.train.remote())
    assert ray.get(s.getRevision.remote()) == 0
    out, _err = capsys.readouterr()
    assert "PITTING AGAINST PREVIOUS VERSION" in out
    assert "REJECTING NEW MODEL" in out


def testCoach(capsys, tmpdir):
    args.checkpoint = tmpdir
    game = OthelloGame(6)
    nnet = OthelloNNet(game)
    coach = Coach(game, nnet, args)
    coach.learn()
