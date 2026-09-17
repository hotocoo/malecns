"""Fetch the primary road-law documents and extract the rules the driver is judged by.

The reward may not charge a fine that nobody wrote down. This module downloads
the statutes themselves, verifies each file against a recorded SHA-256, and
slices the rule text out of them *verbatim* - no provision is ever retyped
here, so the corpus cannot drift from the law it claims to quote.

  python3 src/fetch_law.py                 # download + verify the sources
  python3 src/fetch_law.py --extract       # rebuild data/law/my_road_law.json

What a rule records:

  * `citation`   - instrument and provision, as the statute itself names it;
  * `quote`      - the exact words, sliced from the downloaded document;
  * `source`     - which document, and its SHA-256;
  * `status`     - `verified` when the quote came out of a downloaded primary
    document, `cited` when only the citation is primary (Act 333 names the
    rule in its Schedules) and the instrument's own text is not yet fetchable.

Malaysian subsidiary legislation of the L.N. and P.U.(A) series is not
published in a machine-readable form on the open web; those rules stay `cited`
until a primary copy is supplied with `--add-source`, and the tests keep that
distinction honest rather than letting an unverified rule pass as law.

Act 333 text (c) Government of Malaysia, published by the Ministry of Transport.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

LAW_DIR = Path("data/law")
SOURCE_DIR = LAW_DIR / "sources"
CORPUS = LAW_DIR / "my_road_law.json"
JURISDICTION = "MY"


@dataclass(frozen=True)
class Source:
    """A primary legal document, pinned by content hash."""

    name: str
    url: str
    sha256: str
    title: str
    publisher: str

    @property
    def pdf(self) -> Path:
        return SOURCE_DIR / f"{self.name}.pdf"

    @property
    def text(self) -> Path:
        return SOURCE_DIR / f"{self.name}.txt"


SOURCES = (
    Source(
        name="act333",
        url="https://www.mot.gov.my/en/Documents/Act%20333%20-%20Road%20Transport%20Act%201987.pdf",
        sha256="fd23f3d020334a071bf0a76ef3842bca745051969821fc08c40c70694a7ddc1f",
        title="Laws of Malaysia, Act 333, Road Transport Act 1987 (reprint)",
        publisher="Ministry of Transport Malaysia",
    ),
)


@dataclass(frozen=True)
class Extract:
    """Where a provision's words start and stop inside a source document."""

    rule_id: str
    source: str
    citation: str
    start: str  # regex matching the first line of the provision
    end: str  # regex matching the first line *after* it
    subject: str
    applies_to: dict = field(default_factory=dict)


# Each entry points at the marginal note the reprint prints above the
# provision, and at the note above the next one. The words in between are the
# law; nothing between these anchors is edited here.
EXTRACTS = (
    Extract(
        rule_id="speeding",
        source="act333",
        citation="Road Transport Act 1987 (Act 333), section 40(1)",
        start=r"^Exceeding speed limit$",
        end=r"^Causing death by reckless",
        subject="speed",
        applies_to={
            # The number itself is never written here: it is whatever the
            # survey posts for the road the car is on.
            "limit_source": "osm:maxspeed",
            "limit_fallback": "roadlaw:class_median",
            "offence_when": "speed > limit",
        },
    ),
    Extract(
        rule_id="speed_limit_authority",
        source="act333",
        citation="Road Transport Act 1987 (Act 333), section 69",
        start=r"^Speed limits$",
        end=r"^Power to restrict use of vehicles",
        subject="speed",
        applies_to={"limit_source": "osm:maxspeed", "posted_by": "traffic sign"},
    ),
    Extract(
        rule_id="traffic_sign",
        source="act333",
        citation="Road Transport Act 1987 (Act 333), section 79(2)",
        start=r"^Penalties for neglect of traffic directions and signs$",
        end=r"^Ropes, etc\., across road$",
        subject="signs",
        applies_to={"sign_source": "osm:traffic_sign", "offence_when": "indication not followed"},
    ),
    Extract(
        rule_id="authorized_left_turn",
        source="act333",
        citation="Road Transport Act 1987 (Act 333), section 76A(1)",
        start=r"^Authorized left turns$",
        end=r"^Erection of traffic signs$",
        subject="signals",
        applies_to={
            "sign_source": "osm:traffic_sign",
            "requires": "stop before the marked stop line",
            "exception_to": "red_signal",
        },
    ),
    Extract(
        rule_id="pedestrian_crossing",
        source="act333",
        citation="Road Transport Act 1987 (Act 333), section 75",
        start=r"^Pedestrian crossings$",
        end=r"^Duty of pedestrians to comply",
        subject="pedestrians",
        applies_to={"crossing_source": "osm:highway=crossing"},
    ),
    Extract(
        rule_id="careless_driving",
        source="act333",
        citation="Road Transport Act 1987 (Act 333), section 43(1)",
        start=r"^Careless and inconsiderate driving$",
        end=r"^Driving while under the influence",
        subject="conduct",
        applies_to={"offence_when": "without due care and attention"},
    ),
    Extract(
        rule_id="reckless_driving",
        source="act333",
        citation="Road Transport Act 1987 (Act 333), section 42",
        start=r"^Reckless and dangerous driving$",
        end=r"^Careless and inconsiderate driving$",
        subject="conduct",
        applies_to={"offence_when": "manner dangerous to the public"},
    ),
    Extract(
        rule_id="duty_to_stop_after_accident",
        source="act333",
        citation="Road Transport Act 1987 (Act 333), section 52",
        start=r"^Duty to stop in case of accidents$",
        end=r"^Power to order appearance in court$",
        subject="conduct",
        applies_to={"offence_when": "collision without stopping"},
    ),
    Extract(
        rule_id="scheduled_offences",
        source="act333",
        citation="Road Transport Act 1987 (Act 333), First and Second Schedules",
        start=r"^\s*FIRST SCHEDULE\s*$",
        end=r"^\s*THIRD SCHEDULE\s*$",
        subject="index",
        applies_to={"role": "names the subsidiary rules that carry the remaining offences"},
    ),
)


# Rules whose citation comes from a primary document (the Act's own Schedules
# name them) but whose text is not published machine-readably. They are carried
# with `status: cited` so the corpus states plainly what it has not verified.
CITED_ONLY = (
    {
        "rule_id": "lane_discipline",
        "citation": "Road Traffic Rules 1959 [L.N. 166/1959], rule 3",
        "named_by": "Road Transport Act 1987 (Act 333), First Schedule item (vi)",
        "subject": "lane",
        "applies_to": {"side_source": "osm:driving_side", "offence_when": "not keeping to the near side"},
    },
    {
        "rule_id": "queue_jumping",
        "citation": "Road Traffic Rules 1959 [L.N. 166/1959], rule 3(2)(b)",
        "named_by": "Road Transport Act 1987 (Act 333), First Schedule item (vi)",
        "subject": "lane",
        "applies_to": {"offence_when": "leaving the queue to pass it"},
    },
    {
        "rule_id": "overtaking",
        "citation": "Road Traffic Rules 1959 [L.N. 166/1959], rule 6",
        "named_by": "Road Transport Act 1987 (Act 333), First Schedule item (v)",
        "subject": "overtaking",
        "applies_to": {"offence_when": "overtaking unsafely or preventing another from overtaking"},
    },
    {
        "rule_id": "emergency_lane",
        "citation": "Road Traffic Rules 1959 [L.N. 166/1959], rule 53",
        "named_by": "Road Transport Act 1987 (Act 333), First Schedule item (iv)",
        "subject": "lane",
        "applies_to": {"lane_source": "osm:shoulder", "offence_when": "driving in an emergency lane"},
    },
    {
        "rule_id": "double_line_overtaking",
        "citation": "Road Transport Act 1987 (Act 333), subsection 79(2)",
        "named_by": "Road Transport Act 1987 (Act 333), First Schedule item (iii)",
        "subject": "overtaking",
        "applies_to": {"marking_source": "osm:overtaking=no", "offence_when": "crossing a double line to overtake"},
    },
    {
        "rule_id": "red_signal",
        "citation": "Road Transport Act 1987 (Act 333), Second Schedule item (iii)",
        "named_by": "Road Transport Act 1987 (Act 333), Second Schedule",
        "subject": "signals",
        "applies_to": {
            "signal_source": "osm:highway=traffic_signals",
            "offence_when": "failing to obey a red signal at the traffic sign",
            "subject_to": "authorized_left_turn",
        },
    },
    {
        "rule_id": "bus_lane",
        "citation": "Road Transport Act 1987 (Act 333), Second Schedule item (i)",
        "named_by": "Road Transport Act 1987 (Act 333), Second Schedule",
        "subject": "lane",
        "applies_to": {"lane_source": "osm:lanes:psv", "offence_when": "driving a non-bus in a bus lane"},
    },
    {
        "rule_id": "speed_limit_rule",
        "citation": "Motor Vehicles (Speed Limit) Rules 1989 [P.U. (A) 25/1989], rule 3",
        "named_by": "Road Transport Act 1987 (Act 333), First Schedule item (i)",
        "subject": "speed",
        "applies_to": {"limit_source": "osm:maxspeed"},
    },
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(source: Source, force: bool = False) -> Path:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    if source.pdf.exists() and not force and sha256_of(source.pdf) == source.sha256:
        print(f"[skip] {source.name} ({source.pdf.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
        return source.pdf
    print(f"[get ] {source.name}", file=sys.stderr)
    result = subprocess.run(
        ["curl", "-sSL", "-m", "180", "-A", "malecns-law/1.0", "-o", str(source.pdf), source.url],
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"download failed for {source.name} (curl {result.returncode})")
    got = sha256_of(source.pdf)
    if got != source.sha256:
        raise SystemExit(
            f"{source.name}: sha256 {got} does not match the pinned {source.sha256}. "
            "The publisher reissued the document; re-read it before pinning the new hash."
        )
    return source.pdf


def to_text(source: Source, force: bool = False) -> Path:
    if source.text.exists() and not force:
        return source.text
    result = subprocess.run(["pdftotext", "-layout", str(source.pdf), str(source.text)], check=False)
    if result.returncode != 0:
        raise SystemExit("pdftotext failed; install poppler (brew install poppler)")
    return source.text


def slice_provision(text: str, start: str, end: str) -> str:
    """The lines from the `start` anchor up to the `end` anchor, verbatim.

    The reprint repeats every marginal note once in the table of contents and
    once above the provision; the last occurrence is the provision itself.
    """
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if re.search(start, line)]
    if not starts:
        raise SystemExit(f"anchor not found: {start!r}")
    begin = starts[-1]
    ends = [i for i, line in enumerate(lines) if i > begin and re.search(end, line)]
    if not ends:
        raise SystemExit(f"closing anchor not found after {start!r}: {end!r}")
    return "\n".join(lines[begin : ends[0]]).strip()


def strip_page_furniture(quote: str) -> str:
    """Drop the running headers the reprint stamps across a provision.

    A page break inserts a line like `62   Laws of Malaysia   ACT 333` in the
    middle of a sentence. Removing it changes no word of the provision.
    """
    keep = []
    for line in quote.splitlines():
        bare = line.strip()
        if re.fullmatch(r"\d*\s*Laws of Malaysia\s+ACT\s+333\s*\d*", bare):
            continue
        if re.fullmatch(r"\d*\s*Road Transport\s*\d*", bare):
            continue
        keep.append(line)
    return "\n".join(keep).strip()


def build_corpus() -> dict:
    by_name = {s.name: s for s in SOURCES}
    texts = {s.name: to_text(s).read_text(errors="replace") for s in SOURCES}
    rules = []
    for item in EXTRACTS:
        source = by_name[item.source]
        quote = strip_page_furniture(slice_provision(texts[item.source], item.start, item.end))
        rules.append(
            {
                "rule_id": item.rule_id,
                "subject": item.subject,
                "citation": item.citation,
                "status": "verified",
                "quote": quote,
                "source": {"name": source.name, "title": source.title, "url": source.url, "sha256": source.sha256},
                "applies_to": item.applies_to,
            }
        )
    for item in CITED_ONLY:
        rules.append(
            {
                "rule_id": item["rule_id"],
                "subject": item["subject"],
                "citation": item["citation"],
                "status": "cited",
                "quote": None,
                "named_by": item["named_by"],
                "source": {"name": "act333", "title": by_name["act333"].title, "url": by_name["act333"].url, "sha256": by_name["act333"].sha256},
                "applies_to": item["applies_to"],
            }
        )
    return {
        "jurisdiction": JURISDICTION,
        "enforced_by": "Jabatan Pengangkutan Jalan (JPJ) and the Royal Malaysia Police",
        "note": (
            "Every quote is sliced verbatim from the pinned source document. A rule with "
            "status 'cited' carries a citation taken from a primary document but not the "
            "instrument's own words; supply the instrument to promote it to 'verified'."
        ),
        "sources": [
            {"name": s.name, "title": s.title, "url": s.url, "sha256": s.sha256, "publisher": s.publisher}
            for s in SOURCES
        ],
        "rules": rules,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true", help="rebuild the rule corpus from the sources")
    parser.add_argument("--force", action="store_true", help="re-download and re-convert")
    parser.add_argument("--out", type=Path, default=CORPUS)
    args = parser.parse_args(argv)

    for source in SOURCES:
        download(source, args.force)
        to_text(source, args.force)
    if not args.extract:
        return 0

    corpus = build_corpus()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(corpus, indent=2, ensure_ascii=False) + "\n")
    verified = sum(1 for r in corpus["rules"] if r["status"] == "verified")
    print(f"[ok] {args.out} {len(corpus['rules'])} rules, {verified} verified", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
