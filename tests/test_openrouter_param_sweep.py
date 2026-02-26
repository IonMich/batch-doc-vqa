#!/usr/bin/env python3
"""Tests for OpenRouter parameter sweep parsing and grid construction."""

import unittest

from batch_doc_vqa.openrouter.param_sweep import (
    SweepAxis,
    build_condition_grid,
    build_sweep_axes,
    parse_sweep_axis,
)


class TestParamSweepParsing(unittest.TestCase):
    def test_parse_sweep_axis_temperature(self):
        axis = parse_sweep_axis("temperature=0, 1, 1")
        self.assertEqual(axis.name, "temperature")
        self.assertEqual(axis.values, (0.0, 1.0))

    def test_parse_sweep_axis_int_param(self):
        axis = parse_sweep_axis("top_k=10,20")
        self.assertEqual(axis.name, "top_k")
        self.assertEqual(axis.values, (10, 20))

    def test_parse_sweep_axis_rejects_unknown_param(self):
        with self.assertRaises(ValueError):
            parse_sweep_axis("seed=42")

    def test_parse_sweep_axis_requires_values(self):
        with self.assertRaises(ValueError):
            parse_sweep_axis("temperature= , ")

    def test_build_sweep_axes_supports_temperature_shortcut(self):
        axes = build_sweep_axes(set_specs=[], temperature_values="0,1")
        self.assertEqual(len(axes), 1)
        self.assertEqual(axes[0].name, "temperature")
        self.assertEqual(axes[0].values, (0.0, 1.0))

    def test_build_sweep_axes_rejects_duplicate_axis(self):
        with self.assertRaises(ValueError):
            build_sweep_axes(
                set_specs=["temperature=0,1"],
                temperature_values="0.2,0.8",
            )


class TestParamSweepGrid(unittest.TestCase):
    def test_build_condition_grid_cartesian_product(self):
        axes = [
            SweepAxis(name="temperature", values=(0.0, 1.0)),
            SweepAxis(name="top_p", values=(0.8, 0.95)),
        ]
        grid = build_condition_grid(axes)
        self.assertEqual(
            grid,
            [
                {"temperature": 0.0, "top_p": 0.8},
                {"temperature": 0.0, "top_p": 0.95},
                {"temperature": 1.0, "top_p": 0.8},
                {"temperature": 1.0, "top_p": 0.95},
            ],
        )


if __name__ == "__main__":
    unittest.main()
