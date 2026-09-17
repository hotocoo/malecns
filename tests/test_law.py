"""Tests for the road-law corpus: what the driver is judged by, and where it came from.

The corpus is the one place a rule may live. These tests keep it honest: every
rule names a real instrument, every quote is the document's own words, and no
speed, fine or tolerance is written into the Python.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from law import (  # noqa: E402
    LawCorpus,
    Rule,
    RuleNotInCorpus,
    load_law,
)

CORPUS = ROOT / "data" / "law" / "my_road_law.json"
needs_corpus = pytest.mark.skipif(not CORPUS.exists(), reason="law corpus not built")


@pytest.fixture(scope="module")
def corpus() -> LawCorpus:
    return load_law(CORPUS)


@needs_corpus
class TestCorpusShape:
    def test_jurisdiction_is_recorded(self, corpus):
        assert corpus.jurisdiction == "MY"

    def test_rules_are_unique(self, corpus):
        ids = [r.rule_id for r in corpus.rules]
        assert len(ids) == len(set(ids))

    def test_every_rule_cites_an_instrument(self, corpus):
        for rule in corpus.rules:
            assert rule.citation.strip(), rule.rule_id
            assert re.search(r"\d", rule.citation), rule.rule_id  # a provision number

    def test_every_rule_names_its_source_document(self, corpus):
        for rule in corpus.rules:
            assert rule.source["url"].startswith("https://"), rule.rule_id
            assert len(rule.source["sha256"]) == 64, rule.rule_id

    def test_status_is_one_of_two_known_kinds(self, corpus):
        assert {r.status for r in corpus.rules} <= {"verified", "cited"}


@needs_corpus
class TestQuotes:
    def test_a_verified_rule_carries_the_documents_own_words(self, corpus):
        for rule in corpus.rules:
            if rule.status == "verified":
                assert rule.quote and len(rule.quote) > 40, rule.rule_id

    def test_an_unverified_rule_makes_no_claim_about_wording(self, corpus):
        for rule in corpus.rules:
            if rule.status == "cited":
                assert rule.quote is None, rule.rule_id

    def test_a_quote_appears_in_its_source_document(self, corpus):
        """Spot-check against the downloaded statute when it is present."""
        text_file = ROOT / "data" / "law" / "sources" / "act333.txt"
        if not text_file.exists():
            pytest.skip("statute text not downloaded")
        text = text_file.read_text(errors="replace")
        for rule in corpus.rules:
            if rule.status != "verified":
                continue
            first = rule.quote.splitlines()[0].strip()
            assert first in text, f"{rule.rule_id}: {first!r} not in the source"

    def test_a_cited_rule_says_which_primary_document_names_it(self, corpus):
        for rule in corpus.rules:
            if rule.status == "cited":
                assert rule.named_by, rule.rule_id


@needs_corpus
class TestLookup:
    def test_rule_by_id(self, corpus):
        rule = corpus.require("speeding")
        assert isinstance(rule, Rule)
        assert "section 40" in rule.citation

    def test_missing_rule_is_an_error_not_a_default(self, corpus):
        with pytest.raises(RuleNotInCorpus):
            corpus.require("no_such_rule")

    def test_rules_group_by_subject(self, corpus):
        assert {r.rule_id for r in corpus.by_subject("speed")}
        assert all(r.subject == "speed" for r in corpus.by_subject("speed"))

    def test_verified_is_a_subset(self, corpus):
        ids = {r.rule_id for r in corpus.rules}
        assert {r.rule_id for r in corpus.verified()} <= ids
        assert all(r.verified for r in corpus.verified())


@needs_corpus
class TestNoHardcodedLaw:
    """The numbers the driver is judged by live in the survey, not in the code."""

    def test_speeding_reads_its_limit_from_the_survey(self, corpus):
        applies = corpus.require("speeding").applies_to
        assert applies["limit_source"].startswith("osm:")

    def test_no_rule_states_a_speed_in_its_parameters(self, corpus):
        for rule in corpus.rules:
            for key, value in rule.applies_to.items():
                assert not isinstance(value, (int, float)), f"{rule.rule_id}.{key} = {value}"

    def test_the_module_contains_no_speed_or_fine_literals(self):
        """A limit or a penalty written in Python would outlive the law that set it."""
        source = (ROOT / "src" / "law.py").read_text()
        code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
        code = re.sub(r'""".*?"""', "", code, flags=re.S)
        for number in re.findall(r"\b\d{2,}\b", code):
            assert int(number) < 10, f"law.py states the literal {number}"

    def test_data_sources_are_named_per_rule(self, corpus):
        """Anything the reward must measure points at where the measurement comes from."""
        sourced = [r for r in corpus.rules if any(k.endswith("_source") for k in r.applies_to)]
        assert len(sourced) >= 5
        for rule in sourced:
            for key, value in rule.applies_to.items():
                if key.endswith("_source"):
                    assert value.startswith(("osm:", "roadlaw:")), f"{rule.rule_id}.{key}"


@needs_corpus
class TestEnforceable:
    def test_enforceable_rules_declare_when_the_offence_happens(self, corpus):
        enforceable = corpus.enforceable()
        assert enforceable
        for rule in enforceable:
            assert rule.applies_to.get("offence_when")

    def test_an_index_entry_is_not_enforceable(self, corpus):
        assert corpus.require("scheduled_offences") not in corpus.enforceable()


class TestLoading:
    def test_a_missing_corpus_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_law(tmp_path / "absent.json")

    def test_a_corpus_without_rules_is_rejected(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"jurisdiction": "MY", "rules": []}))
        with pytest.raises(ValueError):
            load_law(path)

    def test_an_unknown_status_is_rejected(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps(
                {
                    "jurisdiction": "MY",
                    "rules": [
                        {
                            "rule_id": "x",
                            "subject": "speed",
                            "citation": "s 1",
                            "status": "probably",
                            "quote": None,
                            "source": {"url": "https://example.org", "sha256": "0" * 64},
                            "applies_to": {},
                        }
                    ],
                }
            )
        )
        with pytest.raises(ValueError):
            load_law(path)
