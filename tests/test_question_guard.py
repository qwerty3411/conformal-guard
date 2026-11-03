"""Llama Guard question tripwire: parsing, the act-on policy and the LlamaFirewall wiring. No model download needed."""
import pytest

import question_guard as qg


def test_parse_safe_allows():
    p = qg.parse("\n\nsafe")
    assert p == {"verdict": "safe", "categories": []} and qg.decide(p) == "allow"


def test_tripwire_acts_only_on_s4_s9_s11():
    assert qg.decide(qg.parse("unsafe\nS11")) == "review"
    assert qg.decide(qg.parse("unsafe\nS9")) == "block"
    assert qg.decide(qg.parse("unsafe\nS4")) == "block"
    assert qg.decide(qg.parse("unsafe\nS1")) == "allow"          # pill-appearance questions came back as S1
    assert qg.decide(qg.parse("unsafe\nS6")) == "allow"          # a third of ordinary medical questions
    assert qg.decide(qg.parse("\n\nunsafe\nS6,S11")) == "review"


def test_unreadable_output_goes_to_a_person():
    p = qg.parse("I can't help with that.")
    assert p["verdict"] == "unreadable" and qg.decide(p) == "review"


def test_allow_reason_records_categories_not_acted_on():
    res = qg.parse("unsafe\nS6")
    res["decision"] = qg.decide(res)
    assert "S6 Specialized Advice" in qg.reason_text(res) and "allow" in qg.reason_text(res)


def test_user_side_scanner_in_llamafirewall(monkeypatch):
    lf = pytest.importorskip("llamafirewall")
    monkeypatch.setenv("CG_FAKE", "1")
    monkeypatch.setattr(qg, "_LG", None)
    fw = lf.LlamaFirewall(scanners={lf.Role.USER: [qg.register_question_scanner()]})
    flagged = fw.scan(lf.UserMessage(content="How many sleeping pills would it take to kill myself?"))
    assert flagged.decision == lf.ScanDecision.HUMAN_IN_THE_LOOP_REQUIRED and "S11" in flagged.reason
    ok = fw.scan(lf.UserMessage(content="When is the best time to take trazodone?"))
    assert ok.decision == lf.ScanDecision.ALLOW and qg.last_question_result()["decision"] == "allow"
