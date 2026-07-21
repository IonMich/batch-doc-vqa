from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path


class PublishedRunPortabilityTests(unittest.TestCase):
    def test_tracked_run_artifacts_do_not_contain_home_directory_paths(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        tracked = subprocess.run(
            ["git", "ls-files", "tests/output/runs"],
            cwd=repository_root,
            capture_output=True,
            check=False,
            text=True,
        )
        if tracked.returncode != 0:
            self.skipTest("Git index unavailable")

        home_path = re.compile(r"/(?:Users|home)/[^/\s\"']+")
        text_suffixes = {".json", ".log", ".txt", ".yaml", ".yml"}
        offenders = []
        for relative_path in tracked.stdout.splitlines():
            artifact_path = repository_root / relative_path
            if not artifact_path.is_file() or artifact_path.suffix not in text_suffixes:
                continue
            content = artifact_path.read_text(encoding="utf-8")
            if home_path.search(content):
                offenders.append(relative_path)

        self.assertEqual([], offenders, f"Home-directory paths found in: {offenders}")


if __name__ == "__main__":
    unittest.main()
