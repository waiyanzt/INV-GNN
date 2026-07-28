#!/usr/bin/env python3
"""Regression tests for shared-channel masking without requiring torch_sparse."""
from __future__ import annotations

import sys
import types
import unittest

import torch


class _DummySparseTensor:
    pass


if "torch_sparse" not in sys.modules:
    module = types.ModuleType("torch_sparse")
    module.SparseTensor = _DummySparseTensor
    sys.modules["torch_sparse"] = module

from model.sehgnn import SeHGNN  # noqa: E402


class SeHGNNChannelMaskTest(unittest.TestCase):
    def make_model(self) -> SeHGNN:
        torch.manual_seed(19)
        model = SeHGNN(
            dataset="IMDB",
            nfeat=8,
            hidden=8,
            nclass=3,
            feat_keys=["M", "MA", "MD"],
            label_feat_keys=["Y", "MY"],
            tgt_type="M",
            dropout=0.0,
            input_drop=0.0,
            att_drop=0.0,
            n_fp_layers=2,
            n_task_layers=4,
            act="none",
            residual=False,
            data_size={"M": 5, "MA": 6, "MD": 4},
        )
        return model.eval()

    def test_all_one_mask_matches_original_path(self) -> None:
        model = self.make_model()
        features = {
            "M": torch.randn(7, 5),
            "MA": torch.randn(7, 6),
            "MD": torch.randn(7, 4),
        }
        label_features = {"Y": torch.randn(7, 3), "MY": torch.randn(7, 3)}
        with torch.no_grad():
            original = model(None, features, label_features, None)
            masked = model(None, features, label_features, torch.ones(5))
        self.assertTrue(torch.allclose(original, masked, atol=2e-6, rtol=2e-6))


    def test_all_one_mask_preserves_attention_dropout(self) -> None:
        model = self.make_model()
        model.semantic_fusion.att_drop.p = 0.35
        model.train()
        features = {
            "M": torch.randn(7, 5),
            "MA": torch.randn(7, 6),
            "MD": torch.randn(7, 4),
        }
        label_features = {"Y": torch.randn(7, 3), "MY": torch.randn(7, 3)}
        with torch.no_grad():
            torch.manual_seed(991)
            original = model(None, features, label_features, None)
            torch.manual_seed(991)
            masked = model(None, features, label_features, torch.ones(5))
        self.assertTrue(torch.allclose(original, masked, atol=2e-6, rtol=2e-6))

    def test_absent_channel_cannot_change_output(self) -> None:
        model = self.make_model()
        features = {
            "M": torch.randn(7, 5),
            "MA": torch.randn(7, 6),
            "MD": torch.randn(7, 4),
        }
        changed = {key: value.clone() for key, value in features.items()}
        changed["MD"] = torch.randn_like(changed["MD"]) * 1000.0
        label_features = {"Y": torch.randn(7, 3), "MY": torch.randn(7, 3)}
        mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])
        with torch.no_grad():
            first = model(None, features, label_features, mask)
            second = model(None, changed, label_features, mask)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
