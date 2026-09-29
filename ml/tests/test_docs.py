"""Keeps ml/documentation/ from going stale: links, script names and CLI flags.

"Living docs" are every .md under documentation/ except history/. Each check
takes the docs root (and the scripts root where needed) and returns a list of
problem strings, so the same code is tested on the real docs and a synthetic tree.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

import config

DOCS = config.ML_ROOT / "documentation"
SCRIPTS = config.ML_ROOT / "scripts"

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
SCRIPT = re.compile(r"scripts/(\w+)\.py")
FLAG = re.compile(r"`(--[\w-]+)")
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")


def living_docs(docs: Path) -> list[Path]:
    return sorted(p for p in docs.rglob("*.md") if p.relative_to(docs).parts[0] != "history")


def outside_fences(text: str) -> str:
    kept, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        elif not fenced:
            kept.append(line)
    return "\n".join(kept)


def slug(heading: str) -> str:
    kept = re.sub(r"[^\w\s-]", "", heading.lower())
    return kept.replace(" ", "-")


def anchors(md: Path) -> set[str]:
    lines = outside_fences(md.read_text(encoding="utf-8")).splitlines()
    return {slug(m.group(1)) for line in lines if (m := HEADING.match(line))}


def check_links(docs: Path) -> list[str]:
    problems = []
    living = {p.resolve() for p in living_docs(docs)}
    for md in living_docs(docs):
        for target in LINK.findall(outside_fences(md.read_text(encoding="utf-8"))):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path, _, anchor = unquote(target).partition("#")
            dest = (md.parent / path).resolve() if path else md
            where = f"{md.relative_to(docs)}: ({target})"
            if not dest.exists():
                problems.append(f"{where} does not exist")
            elif anchor and dest.suffix == ".md" and dest.is_relative_to(docs.resolve()) \
                    and "history" not in dest.relative_to(docs.resolve()).parts[:1] \
                    and anchor not in anchors(dest):
                problems.append(f"{where} has no heading with that anchor")
    return problems


def check_script_mentions(docs: Path, scripts: Path) -> list[str]:
    return [
        f"{md.relative_to(docs)}: scripts/{name}.py does not exist"
        for md in living_docs(docs)
        for name in sorted(set(SCRIPT.findall(md.read_text(encoding="utf-8"))))
        if not (scripts / f"{name}.py").exists()
    ]


def check_flags(docs: Path, scripts: Path) -> list[str]:
    reference = docs / "reference" / "scripts-and-flags.md"
    if not reference.exists():
        return [f"{reference.relative_to(docs)} is missing"]
    problems = []
    for section in re.split(r"^## ", reference.read_text(encoding="utf-8"), flags=re.M)[1:]:
        title = re.match(r"`(\w+\.py)`", section)
        if not title:
            continue
        script = scripts / title.group(1)
        if not script.exists():
            problems.append(f"scripts-and-flags.md: section {title.group(1)} has no script")
            continue
        source = script.read_text(encoding="utf-8")
        problems += [
            f"scripts-and-flags.md: {title.group(1)} has no flag {flag}"
            for flag in sorted(set(FLAG.findall(section)))
            if f'"{flag}"' not in source
        ]
    return problems


def test_links_resolve():
    assert check_links(DOCS) == []


def test_script_mentions_exist():
    assert check_script_mentions(DOCS, SCRIPTS) == []


def test_documented_flags_exist_in_scripts():
    assert check_flags(DOCS, SCRIPTS) == []


def test_checkers_report_each_kind_of_problem(tmp_path):
    docs, scripts = tmp_path / "documentation", tmp_path / "scripts"
    (docs / "reference").mkdir(parents=True)
    (docs / "history").mkdir()
    scripts.mkdir()
    (scripts / "tool.py").write_text('parser.add_argument("--known")\n', encoding="utf-8")
    (docs / "history" / "old.md").write_text("[gone](nowhere.md) scripts/ghost.py\n", encoding="utf-8")
    (docs / "other.md").write_text("# Other Page\n\n## Real `Heading`!\n", encoding="utf-8")
    (docs / "page.md").write_text(
        "# Page\n\n"
        "[ok](other.md#real-heading) [web](https://example.com/x.md)\n"
        "[broken](missing.md) [bad anchor](other.md#no-such-heading)\n"
        "Run `ml/scripts/absent.py` and `scripts/tool.py`.\n"
        "```\n[fenced](ignored.md)\n```\n",
        encoding="utf-8",
    )
    (docs / "reference" / "scripts-and-flags.md").write_text(
        "# Flags\n\n## `tool.py`\n\n`--known` and `--unknown`\n",
        encoding="utf-8",
    )

    links = check_links(docs)
    assert len(links) == 2
    assert any("missing.md" in p and "does not exist" in p for p in links)
    assert any("no-such-heading" in p for p in links)
    assert check_script_mentions(docs, scripts) == ["page.md: scripts/absent.py does not exist"]
    assert check_flags(docs, scripts) == ["scripts-and-flags.md: tool.py has no flag --unknown"]
