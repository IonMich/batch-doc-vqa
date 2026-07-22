import tempfile
import unittest
from pathlib import Path

from PIL import Image, PngImagePlugin

from batch_doc_vqa.benchmarks.pareto_plot import (
    PARETO_FINGERPRINT_KEY,
    create_pareto_plot,
)
from batch_doc_vqa.update_benchmarks import _artifacts_match


class UpdateBenchmarksTests(unittest.TestCase):
    def _write_png(
        self,
        path: Path,
        *,
        color: str,
        fingerprint: str,
        size: tuple[int, int] = (4, 3),
    ) -> None:
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text(PARETO_FINGERPRINT_KEY, fingerprint)
        Image.new("RGBA", size, color=color).save(path, pnginfo=metadata)

    def test_png_comparison_uses_semantic_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            generated = root / "generated.png"
            committed = root / "committed.png"
            self._write_png(generated, color="red", fingerprint="same-data")
            # Different image dimensions are expected when the same plot is
            # rendered with different platform fonts or rasterizers.
            self._write_png(committed, color="blue", fingerprint="same-data", size=(7, 5))

            self.assertTrue(_artifacts_match(generated, committed))

            self._write_png(committed, color="blue", fingerprint="different-data")
            self.assertFalse(_artifacts_match(generated, committed))

    def test_generated_pareto_png_contains_data_fingerprint(self) -> None:
        run_stats = {
            "provider/example": {
                "run_info": {
                    "config": {
                        "model": {
                            "org": "provider",
                            "model": "example",
                            "variant": None,
                        }
                    }
                },
                "stats": {
                    "total_cost": 0.01,
                    "cost_status": "precise",
                    "id_top1": 75.0,
                },
            }
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "pareto.png"
            create_pareto_plot(run_stats, str(output))
            with Image.open(output) as image:
                fingerprint = image.info.get(PARETO_FINGERPRINT_KEY)
        self.assertRegex(str(fingerprint), r"^[0-9a-f]{64}$")

    def test_non_png_comparison_remains_byte_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            generated = root / "generated.md"
            committed = root / "committed.md"
            generated.write_text("same\n", encoding="utf-8")
            committed.write_text("same\n", encoding="utf-8")
            self.assertTrue(_artifacts_match(generated, committed))
            committed.write_text("different\n", encoding="utf-8")
            self.assertFalse(_artifacts_match(generated, committed))


if __name__ == "__main__":
    unittest.main()
