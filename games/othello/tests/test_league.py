from alpha_zero_general import League
from alpha_zero_general import RandomPlayer

from ..game import OthelloGame


def testLeagueBasicUsage():
    game = OthelloGame(6)
    rounds = 2
    league = League(game, initial_rating=1500, rounds=rounds, games=4)
    league.addPlayer("random_1", RandomPlayer(game), initial_rating=2000)
    league.addPlayer("random_2", RandomPlayer(game), initial_rating=1750)
    league.addPlayer("random_3", RandomPlayer(game))
    league.addPlayer("random_4", RandomPlayer(game), initial_rating=1000)
    assert league.has_started is False
    initial_ratings = league.ratings()
    assert initial_ratings == [
        (2000, 1, "random_1"),
        (1750, 2, "random_2"),
        (1500, 3, "random_3"),
        (1000, 4, "random_4"),
    ]
    league.start()
    assert league.has_started is True
    assert len(league.history) == sum(range(4)) * rounds
    final_ratings = league.ratings()
    assert len(final_ratings) == 4
    assert final_ratings != initial_ratings


def testLeagueIncrementallyAddPlayersToRunning():
    game = OthelloGame(6)
    rounds = 2
    league = League(game, initial_rating=1500, rounds=rounds, games=4)
    assert league.has_started is False

    league.start()
    assert league.has_started is True
    league.addPlayer("random_1", RandomPlayer(game))
    assert league.ratings() == [(1500, 1, "random_1")]
    assert len(league.history) == 0

    for i in range(2, 6):
        league.addPlayer(f"random_{i}", RandomPlayer(game))
        assert len(league.ratings()) == i
        assert len(league.history) == sum(range(i)) * rounds


def testLeagueLazyLoadingPlayers():
    game = OthelloGame(6)
    league = League(game, rounds=1, games=2, cache_size=2)
    for i in range(10):
        league.addPlayer(f"random_{i}", lambda: RandomPlayer(game))
    league.start()
    final_ratings = league.ratings()
    assert len(final_ratings) == 10
