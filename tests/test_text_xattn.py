"""Unit tests for gated cross-attention, the 4-parameter validity gate, and inference."""

import os
import tempfile
import unittest

import torch
import torch.nn.functional as F

from diff_models import ResidualBlock, diff_CSDI
from tests.synthetic_factory import make_batch, make_model
from utils.text_gate import TextValidityGate, benefit_label


class GateTests(unittest.TestCase):
    def test_gate_has_four_parameters(self):
        gate = TextValidityGate(n_sources=2)
        named = dict(gate.named_parameters())
        self.assertEqual(set(named), {"a", "b", "beta"})
        self.assertEqual(named["a"].numel(), 1)
        self.assertEqual(named["b"].numel(), 1)
        self.assertEqual(tuple(named["beta"].shape), (2,))
        self.assertEqual(sum(param.numel() for param in gate.parameters()), 4)
        model = make_model(decomp=False, layers=1, num_steps=2)
        self.assertEqual(sum(param.numel() for param in model.text_gate.parameters()), 4)

    def test_benefit_label_is_open_unit_interval_and_detached(self):
        loss_null = torch.tensor([1.0, 3.0, 0.2], requires_grad=True)
        loss_cond = torch.tensor([1.0, 1.0, 0.8], requires_grad=True)
        label = benefit_label(loss_null, loss_cond)
        self.assertFalse(label.requires_grad)
        self.assertTrue(torch.all(label > 0))
        self.assertTrue(torch.all(label < 1))
        self.assertTrue(torch.allclose(label[0], torch.tensor(0.5)))
        loss_null.sum().backward()
        self.assertIsNotNone(loss_null.grad)

    def test_bce_does_not_train_detached_similarity(self):
        gate = TextValidityGate(2)
        with torch.no_grad():
            gate.a.fill_(1.0)
            gate.beta.copy_(torch.tensor([0.4, -0.2]))
        similarity = torch.tensor([0.2, -0.5, 0.7, 0.1], requires_grad=True)
        source_mask = torch.tensor(
            [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
        )
        prediction = gate(similarity.detach(), source_mask)
        self.assertEqual(float(prediction[-1]), 0.0)
        target = torch.tensor([0.2, 0.8, 0.1, 0.0])
        valid = source_mask.sum(dim=-1) > 0
        loss = F.binary_cross_entropy(prediction[valid].clamp(1e-6, 1 - 1e-6), target[valid])
        loss.backward()
        self.assertIsNone(similarity.grad)
        self.assertIsNotNone(gate.a.grad)
        self.assertGreater(float(gate.a.grad.abs()), 0.0)
        self.assertIsNotNone(gate.beta.grad)


class ReductionTests(unittest.TestCase):
    def _block(self):
        block = ResidualBlock(
            side_dim=4,
            channels=32,
            diffusion_embedding_dim=16,
            nheads=4,
            is_linear=False,
            with_text=True,
            context_dim=32,
            dropout=0.0,
            attn_drop=0.0,
            pre_norm=False,
            pred_len=4,
            use_tv=False,
            text_mode="gated",
        )
        block.eval()
        return block

    def _inputs(self, block):
        generator = torch.Generator().manual_seed(3)
        x = torch.randn(2, 32, 2, 8, generator=generator)
        cond = torch.randn(2, 4, 2, 8, generator=generator)
        diffusion = torch.randn(2, 16, generator=generator)
        memory = torch.randn(2, 5, 32, generator=generator)
        mask = torch.tensor(
            [[1, 1, 1, 0, 0], [1, 0, 0, 0, 0]],
            dtype=torch.bool,
        )
        return x, cond, diffusion, memory, mask

    def test_alpha_zero_and_inject_zero_match_numeric_path(self):
        block = self._block()
        x, cond, diffusion, memory, mask = self._inputs(block)
        with torch.no_grad():
            numeric = block(x, cond, diffusion, text_memory=None)[0]
            block.text_alpha.zero_()
            at_zero = block(
                x, cond, diffusion,
                text_memory=memory,
                text_token_mask=mask,
                text_inject=torch.ones(2),
            )[0]
            block.text_alpha.fill_(0.7)
            dropped = block(
                x, cond, diffusion,
                text_memory=memory,
                text_token_mask=mask,
                text_inject=torch.zeros(2),
            )[0]
            invalid = block(
                x, cond, diffusion,
                text_memory=memory,
                text_token_mask=torch.zeros_like(mask),
                text_inject=torch.ones(2),
            )[0]
            active = block(
                x, cond, diffusion,
                text_memory=memory,
                text_token_mask=mask,
                text_inject=torch.ones(2),
            )[0]
        self.assertTrue(torch.equal(at_zero, numeric))
        self.assertTrue(torch.equal(dropped, numeric))
        self.assertTrue(torch.equal(invalid, numeric))
        self.assertFalse(torch.equal(active, numeric))

    def test_padding_tokens_do_not_change_output(self):
        block = self._block()
        x, cond, diffusion, memory, mask = self._inputs(block)
        with torch.no_grad():
            block.text_alpha.fill_(0.8)
            first = block(
                x, cond, diffusion,
                text_memory=memory,
                text_token_mask=mask,
                text_inject=torch.ones(2),
            )[0]
            altered = memory.clone()
            altered = altered.masked_fill(~mask.unsqueeze(-1), 3.5)
            second = block(
                x, cond, diffusion,
                text_memory=altered,
                text_token_mask=mask,
                text_inject=torch.ones(2),
            )[0]
        self.assertTrue(torch.equal(first, second))

    def test_alpha_receives_gradient(self):
        block = self._block()
        block.train()
        for module in block.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        x, cond, diffusion, memory, mask = self._inputs(block)
        output = block(
            x, cond, diffusion,
            text_memory=memory,
            text_token_mask=mask,
            text_inject=torch.ones(2),
        )[0]
        output.square().mean().backward()
        self.assertIsNotNone(block.text_alpha.grad)
        self.assertGreater(float(block.text_alpha.grad.abs()), 0.0)

    def test_denoiser_reduction_is_not_vacuous(self):
        config = {
            "channels": 32,
            "num_steps": 4,
            "diffusion_embedding_dim": 16,
            "lookback_len": 6,
            "pred_len": 2,
            "with_timestep": False,
            "dropout": 0.0,
            "attn_drop": 0.0,
            "pre_norm": False,
            "time_weight": 0.0,
            "save_attn": False,
            "text_mode": "gated",
            "n_text_sources": 2,
            "with_texts": True,
            "context_dim": 16,
            "side_dim": 5,
            "is_linear": False,
            "nheads": 4,
            "layers": 1,
        }
        denoiser = diff_CSDI(config, inputdim=2)
        denoiser.eval()
        with torch.no_grad():
            denoiser.output_projection2.weight.normal_(0.0, 0.05)
            denoiser.output_projection2.bias.fill_(0.01)
        generator = torch.Generator().manual_seed(11)
        noisy = torch.randn(2, 2, 2, 8, generator=generator)
        side = torch.randn(2, 5, 2, 8, generator=generator)
        hidden = torch.randn(2, 6, 16, generator=generator)
        token_mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.float32)
        source_ids = torch.tensor([0, 0, 0, 1, 1, 1])
        step = torch.tensor([1])
        with torch.no_grad():
            for layer in denoiser.residual_layers:
                layer.text_alpha.zero_()
            numeric = denoiser(noisy, side, step, cfg_mask=None, text_hidden=None)
            zero_alpha = denoiser(
                noisy, side, step,
                cfg_mask=torch.ones(2),
                text_hidden=hidden,
                text_token_mask=token_mask,
                text_source_ids=source_ids,
            )
            for layer in denoiser.residual_layers:
                layer.text_alpha.fill_(0.5)
            dropped = denoiser(
                noisy, side, step,
                cfg_mask=torch.zeros(2),
                text_hidden=hidden,
                text_token_mask=token_mask,
                text_source_ids=source_ids,
            )
            changed = hidden.clone()
            changed[:, 3:] = torch.randn_like(changed[:, 3:])
            masked = denoiser(
                noisy, side, step,
                cfg_mask=torch.ones(2),
                text_hidden=changed,
                text_token_mask=token_mask,
                text_source_ids=source_ids,
            )
            active = denoiser(
                noisy, side, step,
                cfg_mask=torch.ones(2),
                text_hidden=hidden,
                text_token_mask=token_mask,
                text_source_ids=source_ids,
            )
        self.assertTrue(torch.equal(zero_alpha, numeric))
        self.assertTrue(torch.equal(dropped, numeric))
        self.assertTrue(torch.equal(masked, active))
        self.assertFalse(torch.equal(active, numeric))


class ForecastBehaviorTests(unittest.TestCase):
    def test_mask_noisy_target_zeros_history_channel(self):
        model = make_model(decomp=False, layers=1, num_steps=2)
        observed = torch.randn(2, 2, 16)
        cond_mask = torch.zeros(2, 2, 16)
        cond_mask[:, :, :12] = 1.0
        noisy = torch.randn_like(observed)
        model.mask_noisy_target = True
        masked = model.set_input_to_diffmodel(noisy, observed, cond_mask)
        history = masked[:, 1, :, :12]
        future = masked[:, 1, :, 12:]
        self.assertTrue(torch.equal(history, torch.zeros_like(history)))
        self.assertTrue(torch.equal(future, noisy[:, :, 12:]))
        model.mask_noisy_target = False
        legacy = model.set_input_to_diffmodel(noisy, observed, cond_mask)
        self.assertTrue(torch.equal(legacy[:, 1], noisy))

    def test_fpp4_router_ignores_text_and_uses_seasonal_period(self):
        model = make_model(decomp=False, layers=1, num_steps=2, lookback=12, pred=4)
        hist = torch.arange(12, dtype=torch.float32).view(1, 1, 12).repeat(1, 2, 1)
        hist_mask = torch.ones_like(hist)
        experts = model._build_fpp4_expert_futures(hist, hist_mask, 4)
        self.assertEqual(tuple(experts.shape), (1, 4, 2, 4))
        # h=1 reads index L-P+(0 mod P) = 8. The series values are 0..11.
        self.assertTrue(torch.equal(experts[0, 2, 0], hist[0, 0, 8:12]))
        text_a = torch.randn(1, 32)
        text_b = torch.randn(1, 32)
        pi_a = model._linear_history_router(hist, hist_mask, text_pooled=text_a, text_mask=torch.ones(1))
        pi_b = model._linear_history_router(hist, hist_mask, text_pooled=text_b, text_mask=torch.zeros(1))
        self.assertTrue(torch.equal(pi_a, pi_b))
        self.assertIsNone(model.pattern_text_head)
        self.assertEqual(model.pattern_num_experts, 4)

    def test_hash_encoder_keeps_recent_tokens(self):
        model = make_model(decomp=False, layers=1, num_steps=2)
        words = ["old%d" % index for index in range(20)] + ["recent", "tail"]
        hidden, _, packed = model._hash_encode_texts([" ".join(words)], max_length=6, truncation_side="left")
        ids = packed["input_ids"][0]
        self.assertEqual(int(ids[0]), 101)
        self.assertEqual(int(ids[5]), 102)
        kept = [model._stable_token_id(token) for token in ("recent", "tail")]
        self.assertEqual(int(ids[3]), 1 + (kept[0] % 30000))
        self.assertEqual(int(ids[4]), 1 + (kept[1] % 30000))
        right_hidden, _, right_packed = model._hash_encode_texts([" ".join(words)], max_length=6, truncation_side="right")
        self.assertFalse(torch.equal(hidden, right_hidden))
        self.assertNotEqual(int(right_packed["input_ids"][0, 1]), int(ids[1]))

    def test_bert_tokenizer_left_truncation_keeps_tail(self):
        from transformers import BertTokenizer

        tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "old", "recent", "tail"]
        with tempfile.TemporaryDirectory() as directory:
            vocab_path = os.path.join(directory, "vocab.txt")
            with open(vocab_path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(tokens) + "\n")
            tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=True)
            tokenizer.truncation_side = "left"
            encoded = tokenizer(
                ["old old old old old recent tail"],
                padding="max_length",
                truncation=True,
                max_length=4,
                return_tensors="pt",
            )
        ids = encoded["input_ids"][0].tolist()
        self.assertEqual(ids[0], tokenizer.cls_token_id)
        self.assertEqual(ids[1], tokenizer.convert_tokens_to_ids("recent"))
        self.assertEqual(ids[2], tokenizer.convert_tokens_to_ids("tail"))
        self.assertEqual(ids[3], tokenizer.sep_token_id)
        self.assertNotIn(tokenizer.convert_tokens_to_ids("old"), ids)

    def test_inference_does_not_use_future_targets(self):
        for decomp in (False, True):
            model = make_model(decomp=decomp, layers=1, num_steps=2, channels=32)
            model.eval()
            batch = make_batch(model, batch_size=2, target_dim=2)
            scrambled = {key: (value.clone() if torch.is_tensor(value) else list(value) if isinstance(value, list) else value) for key, value in batch.items()}
            future = scrambled["observed_data"][:, model.lookback_len:, :].clone()
            scrambled["observed_data"] = scrambled["observed_data"].clone()
            scrambled["observed_data"][:, model.lookback_len:, :] = torch.randn_like(future) * 50 + 9
            torch.manual_seed(19)
            first = model.evaluate(batch, n_samples=1, guide_w=1.0)[0]
            torch.manual_seed(19)
            second = model.evaluate(scrambled, n_samples=1, guide_w=1.0)[0]
            self.assertTrue(torch.allclose(first, second), "decomp=%s" % decomp)
            self.assertFalse(torch.allclose(batch["observed_data"], scrambled["observed_data"]))


if __name__ == "__main__":
    unittest.main()
