from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

import evaluation


@pytest.mark.parametrize("answer,expected", [
    ("5", "5"), ("5.0", "5"), ("52,000.00", "52000"),
    ("Toyota", "Toyota"), ("  ontario  ", "Ontario"),
    ("Calculated with SQL.\nFINAL_ANSWER: 5", "5"),
    ("Calculated with SQL.\nFINAL_ANSWER: Toyota", "Toyota"),
])
def test_exact_answers_match(answer, expected):
    assert evaluation.answer_matches(answer, expected)


@pytest.mark.parametrize("answer,expected", [
    ("15", "5"), ("15.0", "5"), ("5.5", "5"), ("-5", "5"),
    ("There are 15 records.", "5"), ("Toyota is not the highest; Honda is.", "Toyota"),
    ("Toyota and Honda", "Toyota"), ("5,0", "50"), ("", "5"),
    ("FINAL_ANSWER: 15", "5"), ("FINAL_ANSWER: 5\nActually, 15.", "5"),
    ("FINAL_ANSWER: Toyota is not the winner", "Toyota"),
])
def test_substrings_and_wrong_answers_do_not_match(answer, expected):
    assert not evaluation.answer_matches(answer, expected)


def test_evaluation_continues_after_a_case_failure(monkeypatch, capsys):
    cases = [{"question": "one", "expected": "5"}, {"question": "two", "expected": "Toyota"}]
    runner = Mock(side_effect=[
        RuntimeError("private exception data"),
        {"messages": [AIMessage(content="FINAL_ANSWER: Toyota")]},
    ])
    monkeypatch.setattr(evaluation, "test_cases", cases)
    monkeypatch.setattr(evaluation, "run_agent", runner)
    assert evaluation.evaluate() == 50.0
    assert runner.call_count == 2
    output = capsys.readouterr().out
    assert "private exception data" not in output
    assert "50.0%" in output
