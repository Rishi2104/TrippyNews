"""Tests for the evaluation harness."""

from __future__ import annotations

from newscheck.eval.harness import IntegrityEvaluator


class TestIntegrityEvaluator:
    def test_run_benchmark(self) -> None:
        evaluator = IntegrityEvaluator()
        report = evaluator.run_benchmark()

        assert report.wire_dedup_accuracy == 1.0
        assert report.claim_anchoring_accuracy == 1.0
        assert report.nli_accuracy == 1.0
        assert report.safety_gate_compliance == 1.0
        assert report.all_passed is True
