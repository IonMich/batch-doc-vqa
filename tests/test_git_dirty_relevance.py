#!/usr/bin/env python3
"""Tests for git-dirty relevance classification and warning messages."""

import unittest

from batch_doc_vqa.core.run_manager import (
    _extract_git_dirty_paths,
    _is_reproducibility_relevant_dirty_path,
    build_git_dirty_warning_lines,
)


class _DummyConfig:
    def __init__(
        self,
        *,
        git_dirty_raw,
        git_dirty_relevant,
        relevant_paths,
        relevant_count=None,
    ):
        self.git_dirty_raw = git_dirty_raw
        self.git_dirty = git_dirty_raw
        self.git_dirty_relevant = git_dirty_relevant
        self.git_dirty_relevant_paths = relevant_paths
        self.git_dirty_relevant_count = relevant_count


class TestGitDirtyHelpers(unittest.TestCase):
    def test_extract_git_dirty_paths_handles_rename(self):
        porcelain = " M README.md\nR  src/old.py -> src/new.py\n?? tests/data/new.csv\n"
        paths = _extract_git_dirty_paths(porcelain)
        self.assertEqual(paths, ["README.md", "src/new.py", "tests/data/new.csv"])

    def test_repro_relevant_path_rules(self):
        self.assertTrue(_is_reproducibility_relevant_dirty_path("src/batch_doc_vqa/openrouter/inference.py"))
        self.assertTrue(_is_reproducibility_relevant_dirty_path("pyproject.toml"))
        self.assertTrue(_is_reproducibility_relevant_dirty_path("imgs/q11/doc_info.csv"))
        self.assertFalse(_is_reproducibility_relevant_dirty_path("README.md"))
        self.assertFalse(_is_reproducibility_relevant_dirty_path("tests/output/runs/foo/results.json"))

    def test_warning_lines_for_relevant_dirty(self):
        cfg = _DummyConfig(
            git_dirty_raw=True,
            git_dirty_relevant=True,
            relevant_paths=["src/a.py", "src/b.py"],
            relevant_count=2,
        )
        lines = build_git_dirty_warning_lines(cfg, max_paths=1)
        self.assertTrue(any("Reproducibility warning" in line for line in lines))
        self.assertTrue(any("src/a.py" in line for line in lines))
        self.assertTrue(any("and 1 more" in line for line in lines))

    def test_warning_lines_for_non_relevant_dirty(self):
        cfg = _DummyConfig(
            git_dirty_raw=True,
            git_dirty_relevant=False,
            relevant_paths=[],
            relevant_count=0,
        )
        lines = build_git_dirty_warning_lines(cfg)
        self.assertEqual(len(lines), 1)
        self.assertIn("none matched reproducibility-relevant", lines[0])

    def test_warning_lines_for_clean_tree(self):
        cfg = _DummyConfig(
            git_dirty_raw=False,
            git_dirty_relevant=False,
            relevant_paths=[],
            relevant_count=0,
        )
        lines = build_git_dirty_warning_lines(cfg)
        self.assertEqual(lines, [])


if __name__ == "__main__":
    unittest.main()
