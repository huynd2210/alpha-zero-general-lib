import pytest
from alpha_zero_general.utils import parseGameFilename
from alpha_zero_general.utils import parseModelFilename


def testParseGameFilename():
    assert parseGameFilename("game_123") == 123
    assert parseGameFilename("game_000123") == 123
    assert parseGameFilename("game_000123.file") == 123
    assert parseGameFilename("game_000123_nice") == 123
    assert parseGameFilename("game_000123_nice.file") == 123
    with pytest.raises(ValueError):
        parseGameFilename("game_lala_123")


def testParseModelFilename():
    assert parseModelFilename("model_123") == 123
    assert parseModelFilename("model_000123") == 123
    assert parseModelFilename("model_000123.file") == 123
    assert parseModelFilename("model_000123_nice") == 123
    assert parseModelFilename("model_000123_nice.file") == 123
    with pytest.raises(ValueError):
        parseModelFilename("model_lala_123")
