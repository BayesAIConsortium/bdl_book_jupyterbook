from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import chapter_downloads as downloads  # noqa: E402


def test_metadata_is_native_and_preserves_body(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = Path("content/vi/intro.md")
    body = "(chap:vi_intro)=\n# Introduction\n\nText and $x$.\n"
    result = downloads.with_chapter_downloads(body, path, 'Introduction: "VI"')
    assert result.split("---\n\n", 1)[1] == body
    assert 'title: "Introduction: \\"VI\\""'.replace('\\\\', '\\') in result
    assert "template: ../_templates/chapter" in result
    assert "output: ../_build/exports/chapters/vi-intro.pdf" in result
    assert "output: ../_build/tex/vi-intro/vi-intro.tex" in result
    assert "id: pdf-vi-intro" in result
    assert "title: Download chapter PDF" in result
    assert "file: intro.md\n    title: Download Markdown source" in result
    assert downloads.with_chapter_downloads(result, path, 'Introduction: "VI"') == result


def test_diagnostic_exports_outside_content_remain_unchanged(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    body = "# Introduction\n"
    assert downloads.with_chapter_downloads(body, tmp_path / "trial.md", "Intro") == body


def test_unmigrated_chapters_do_not_get_downloads(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    migrated = Path("content/vi/intro.md")
    pending = Path("content/vi/mean_field_vi.md")
    migrated.parent.mkdir(parents=True)
    migrated.write_text("(chap:intro)=\n# Introduction\n\nContent.\n")
    pending.write_text("(chap:mean_field)=\n# Mean field\n\n_Chapter migration pending._\n")
    Path("content/myst.yml").write_text(
        "project:\n  toc:\n    - file: vi/intro.md\n    - file: vi/mean_field_vi.md\n"
    )
    monkeypatch.setenv("BDL_TEX_ROOT", "/nonexistent-tex-checkout")
    assert downloads.configure_downloads() == [(migrated, "Introduction")]
    assert migrated.read_text().startswith("---\n")
    assert not pending.read_text().startswith("---\n")
