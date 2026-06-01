"""Unit tests for the negation parser.

Covers English negation patterns post-translation.
Tests are pure logic — regex + string manipulation, no models.
"""
import pytest

from src.agentic.negation_parser import parse_negations, NegationResult

pytestmark = pytest.mark.unit


class TestNegationDetection:

    def test_no_negation_in_simple_query(self):
        result, _ = parse_negations("dog playing in the garden")
        assert result.has_negation is False
        assert result.negative_attributes == []

    def test_not_pattern(self):
        result, _ = parse_negations("car not red on the street")
        assert result.has_negation is True
        assert "red" in result.negative_attributes

    def test_without_pattern(self):
        result, _ = parse_negations("landscape without people")
        assert result.has_negation is True
        assert "people" in result.negative_attributes

    def test_no_prefix_pattern(self):
        result, _ = parse_negations("no dogs in the photo")
        assert result.has_negation is True
        assert "dogs" in result.negative_attributes

    def test_excluding_pattern(self):
        result, _ = parse_negations("animals excluding cats")
        assert result.has_negation is True
        assert "cats" in result.negative_attributes

    def test_multiple_negations(self):
        result, _ = parse_negations("car not red not blue")
        assert result.has_negation is True
        assert "red" in result.negative_attributes
        assert "blue" in result.negative_attributes

    def test_no_duplicate_attributes(self):
        result, _ = parse_negations("car not red not red")
        assert result.negative_attributes.count("red") == 1

    def test_positive_query_strips_negation(self):
        result, _ = parse_negations("car not red on the street")
        assert "not" not in result.positive_query
        assert "red" not in result.positive_query
        assert "car" in result.positive_query

    def test_positive_query_non_empty_on_full_negation(self):
        # Even if negation strips everything, positive_query should not be empty
        result, _ = parse_negations("not red")
        assert result.positive_query  # must not be empty string

    def test_trace_step_has_correct_structure(self):
        _, step = parse_negations("dog not angry")
        assert step.step == "negation_detection"
        assert "detected" in step.result
        assert "positive_query" in step.result
        assert "negative_attributes" in step.result

    def test_trace_step_detected_matches_result(self):
        result, step = parse_negations("cat without collar")
        assert step.result["detected"] == result.has_negation
        assert step.result["negative_attributes"] == result.negative_attributes

    @pytest.mark.parametrize("query,expected_negation,expected_attr", [
        ("person not running",    True,  "running"),
        ("bird without wings",    True,  "wings"),
        ("no cars visible",       True,  "cars"),
        ("sunny day at the park", False, None),
        ("bicycle and helmet",    False, None),
    ])
    def test_parametrized_patterns(self, query, expected_negation, expected_attr):
        result, _ = parse_negations(query)
        assert result.has_negation == expected_negation
        if expected_attr:
            assert expected_attr in result.negative_attributes
