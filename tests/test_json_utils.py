import pytest

from deep_researcher_demo.json_utils import JSONParseError, parse_json_object


def test_parse_plain_json_object():
    assert parse_json_object('{"a": 1}') == {"a": 1}


def test_parse_fenced_json_object():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_invalid_json_raises():
    with pytest.raises(JSONParseError):
        parse_json_object("not json")

