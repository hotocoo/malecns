"""The road law the driver is judged by, loaded from the corpus rather than written here.

`fetch_law.py` builds `data/law/my_road_law.json` by slicing provisions
verbatim out of the statutes it downloads. This module reads that file and
hands the trainer, the evaluator and the viewer the same rules.

Deliberately, there is not a single speed, fine or tolerance in this file. A
rule says *where its number comes from* - `osm:maxspeed` for a posted limit,
`osm:driving_side` for which side to keep - and the survey supplies the value
for the road the car is actually on. Change country and the numbers change with
the survey; nothing here needs editing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

DEFAULT_CORPUS = Path("data/law/my_road_law.json")
STATUSES = ("verified", "cited")

# A rule that only points at other rules (a schedule index) is not something a
# driver can breach; it earns its place by naming the instruments.
INDEX_SUBJECT = "index"


class RuleNotInCorpus(KeyError):
    """Asked for a rule the corpus does not carry.

    Raised rather than returning a default: a driver may not be charged under a
    provision nobody wrote down.
    """


@dataclass(frozen=True)
class Rule:
    """One provision, with the words it was read from and where they came from."""

    rule_id: str
    subject: str
    citation: str
    status: str
    quote: str | None
    source: dict
    applies_to: dict = field(default_factory=dict)
    named_by: str | None = None

    @property
    def verified(self) -> bool:
        """True when the corpus holds the instrument's own words for this rule."""
        return self.status == "verified"

    @property
    def enforceable(self) -> bool:
        """True when the rule states the condition that makes an offence."""
        return self.subject != INDEX_SUBJECT and bool(self.applies_to.get("offence_when"))

    def source_of(self, quantity: str) -> str | None:
        """Where the value for `quantity` is measured, e.g. `limit` -> `osm:maxspeed`."""
        return self.applies_to.get(f"{quantity}_source")


@dataclass(frozen=True)
class LawCorpus:
    """Every rule for one jurisdiction."""

    jurisdiction: str
    rules: tuple[Rule, ...]
    sources: tuple[dict, ...] = ()
    enforced_by: str = ""

    def __iter__(self) -> Iterator[Rule]:
        return iter(self.rules)

    def __len__(self) -> int:
        return len(self.rules)

    def require(self, rule_id: str) -> Rule:
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        raise RuleNotInCorpus(f"{rule_id!r} is not in the {self.jurisdiction} corpus")

    def by_subject(self, subject: str) -> tuple[Rule, ...]:
        return tuple(rule for rule in self.rules if rule.subject == subject)

    def verified(self) -> tuple[Rule, ...]:
        return tuple(rule for rule in self.rules if rule.verified)

    def enforceable(self) -> tuple[Rule, ...]:
        return tuple(rule for rule in self.rules if rule.enforceable)

    def sourced_from(self, prefix: str) -> tuple[Rule, ...]:
        """Rules whose measurement comes from a given data source, e.g. `osm:`."""
        return tuple(
            rule
            for rule in self.rules
            if any(k.endswith("_source") and str(v).startswith(prefix) for k, v in rule.applies_to.items())
        )

    def summary(self) -> str:
        verified = len(self.verified())
        return (
            f"{self.jurisdiction}: {len(self.rules)} rules, {verified} verified, "
            f"{len(self.rules) - verified} cited only, {len(self.enforceable())} enforceable"
        )


def _rule_from(raw: dict) -> Rule:
    status = raw.get("status")
    if status not in STATUSES:
        raise ValueError(f"rule {raw.get('rule_id')!r}: status {status!r} is not one of {STATUSES}")
    return Rule(
        rule_id=str(raw["rule_id"]),
        subject=str(raw.get("subject", "")),
        citation=str(raw.get("citation", "")),
        status=status,
        quote=raw.get("quote"),
        source=dict(raw.get("source") or {}),
        applies_to=dict(raw.get("applies_to") or {}),
        named_by=raw.get("named_by"),
    )


def load_law(path: str | Path = DEFAULT_CORPUS) -> LawCorpus:
    """The corpus at `path`.

    Raises rather than degrading: a missing or empty corpus means the run has
    no law to enforce, which is a setup error, not a lenient default.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python3 src/fetch_law.py --extract`")
    data = json.loads(path.read_text())
    rules = tuple(_rule_from(raw) for raw in data.get("rules", []))
    if not rules:
        raise ValueError(f"{path} carries no rules")
    seen: set[str] = set()
    for rule in rules:
        if rule.rule_id in seen:
            raise ValueError(f"{path}: duplicate rule {rule.rule_id!r}")
        seen.add(rule.rule_id)
    return LawCorpus(
        jurisdiction=str(data.get("jurisdiction", "")),
        rules=rules,
        sources=tuple(data.get("sources") or ()),
        enforced_by=str(data.get("enforced_by", "")),
    )


if __name__ == "__main__":
    corpus = load_law()
    print(corpus.summary())
    for rule in corpus:
        mark = "*" if rule.verified else " "
        print(f" {mark} {rule.rule_id:28s} {rule.citation}")
