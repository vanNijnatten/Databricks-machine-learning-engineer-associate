"""Chapter 13 pitfalls: traffic that does not add up to 100 and request bodies without an input key."""

import pytest

from db_ml_associate.ch13_serving.pitfalls import (
    correct_payload,
    correct_traffic_split,
    payload_problems,
    wrong_payload_without_input_key,
    wrong_traffic_split_not_100,
)


def test_traffic_split_must_total_100() -> None:
    with pytest.raises(ValueError, match="100"):
        wrong_traffic_split_not_100()
    assert correct_traffic_split().traffic_config is not None


def test_payload_needs_one_input_key() -> None:
    assert payload_problems(wrong_payload_without_input_key())
    assert payload_problems(correct_payload()) == []
