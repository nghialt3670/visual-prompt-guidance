"""CPU-only: exact gradients through checkpointed attention guidance + denoising."""
from contextlib import nullcontext

import pytest
import torch
from torch import nn

from visual_prompt_guidance.contracts import AttentionRequest
from visual_prompt_guidance.backends.flux2 import Flux2Klein, _KleinState
from visual_prompt_guidance.config import profile_settings
from visual_prompt_guidance.runtime import build_engine


class SmallAttention(nn.Module):
    fused_projections = False
    head_dim = 2

    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(4, 4)
        self.to_k = nn.Linear(4, 4)
        self.add_q_proj = nn.Linear(4, 4)
        self.add_k_proj = nn.Linear(4, 4)
        self.norm_q = nn.Identity()
        self.norm_k = nn.Identity()
        self.norm_added_q = nn.Identity()
        self.norm_added_k = nn.Identity()

    def forward(self, *, hidden_states, encoder_hidden_states, image_rotary_emb=None):
        return self.to_q(hidden_states).tanh() + .15 * self.to_k(hidden_states)


class SmallBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = SmallAttention()


class SmallTransformer(nn.Module):
    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([SmallBlock(), SmallBlock()])
        self.single_transformer_blocks = nn.ModuleList([])

    def cache_context(self, _):
        return nullcontext()

    def forward(self, *, hidden_states, timestep, guidance, encoder_hidden_states,
                txt_ids, img_ids, joint_attention_kwargs, return_dict):
        h = hidden_states
        for block in self.transformer_blocks:
            h = h + .15 * block.attn(hidden_states=h, encoder_hidden_states=encoder_hidden_states)
        return (h + .0001 * timestep.view(-1, 1, 1),)


class SmallPipeline:
    def __init__(self, transformer):
        self.transformer = transformer


def simulate_training(checkpointed: bool):
    torch.manual_seed(17)
    adapter = Flux2Klein()
    adapter._pipe = SmallPipeline(SmallTransformer().eval().requires_grad_(False))
    adapter.train_checkpoint_transformer = checkpointed
    adapter.train_checkpoint_attention = checkpointed
    adapter.train_query_chunk = 1
    z = torch.randn(1, 2, 4, requires_grad=True)
    state = _KleinState(
        latent_data=z, latent_ids=torch.zeros(1, 2, 3),
        reference=torch.randn(1, 1, 4), reference_ids=torch.zeros(1, 1, 3),
        prompt_embeds=torch.randn(1, 2, 4), text_ids=torch.zeros(1, 2, 3),
        negative_embeds=None, negative_ids=None, schedule=torch.tensor([2., 1.]),
        token_indices=[0], grid=(1, 2), native_size=(32, 16), original_size=(32, 16),
        cfg=1.0, mode='base', edit_mask=torch.ones(1, 2),
        source_latents=None, noise=None,
    )
    logits = torch.nn.Parameter(torch.tensor([.20, -.15, .05, -.1]))
    attention = AttentionRequest(layers=('D0', 'D1'), affinity_layers=None, need_affinity=False)
    predicted = adapter.forward(state, torch.tensor(2.), attention=attention)
    assert predicted.attention.labels == ('D0:H00', 'D0:H01', 'D1:H00', 'D1:H01')
    assert predicted.attention.maps.shape == (1, 4, 1, 2)
    weights = logits.softmax(dim=0)
    inner = (predicted.attention.maps * weights.view(1, -1, 1, 1)).sum()
    inner_grad = torch.autograd.grad(inner, z, create_graph=True)[0]
    state.latent_data = z - .03 * inner_grad
    # Retain the full denoising path: do NOT detach the first post-guidance
    # state, even when it is the input of a later denoising step.
    after = adapter.forward(state, torch.tensor(2.), attention=None)
    state.latent_data = state.latent_data + .07 * after.output
    final = adapter.forward(state, torch.tensor(1.), attention=None)
    loss = final.output.square().mean()
    outer_grad = torch.autograd.grad(loss, logits)[0]
    assert torch.isfinite(outer_grad).all()
    return loss.detach(), outer_grad.detach()


def test_flux_memory_checkpoints_preserve_second_order_head_learning():
    reference_loss, reference_gradient = simulate_training(False)
    optimized_loss, optimized_gradient = simulate_training(True)
    torch.testing.assert_close(optimized_loss, reference_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(optimized_gradient, reference_gradient, rtol=3e-4, atol=3e-6)
    assert reference_gradient.abs().sum() > 0


def test_memory_options_are_training_only_and_configurable():
    settings = profile_settings('train')
    assert settings['train_checkpoint_transformer']
    assert settings['train_checkpoint_attention']
    assert settings['train_checkpoint_vae']
    assert settings['train_query_chunk'] >= 1


def test_runtime_configures_memory_only_for_flux_training():
    class FakeBackend:
        pass
    settings = profile_settings('train')
    settings.update(model='FLUX.2 Klein', selected_layers=['D0'],
                    selected_heads=['D0:H00', 'D0:H01'], train_query_chunk=8)
    stub = FakeBackend()
    engine = build_engine(settings, training=True, backend=stub)
    assert engine.backend is stub
    assert stub.train_checkpoint_transformer is True
    assert stub.train_checkpoint_attention is True
    assert stub.train_checkpoint_vae is True
    assert stub.train_query_chunk == 8


def test_invalid_attention_chunk_rejected():
    class FakeBackend:
        pass
    settings = profile_settings('train')
    settings.update(model='FLUX.2 Klein', train_query_chunk=0)
    with pytest.raises(ValueError, match='train_query_chunk'):
        build_engine(settings, training=True, backend=FakeBackend())


def test_recomputed_attention_context_supports_higher_order_backward():
    """Regression: checkpoint(context_fn=sdpa_kernel) is one-shot and crashes.

    The FLUX backend must instead create a *new* math-SDPA context on every
    model_forward call, including multiple higher-order recomputations.
    """
    from torch.utils.checkpoint import checkpoint
    from torch.nn import functional as F
    from visual_prompt_guidance.backends.flux2 import _fresh_sdpa_context

    def run(checkpointed):
        x = torch.randn(1, 2, 4, 4, requires_grad=True)

        def model_forward(v):
            with _fresh_sdpa_context(True):
                return F.scaled_dot_product_attention(v, v, v)

        y = checkpoint(model_forward, x, use_reentrant=False) if checkpointed else model_forward(x)
        inner_grad = torch.autograd.grad(y.square().sum(), x, create_graph=True)[0]
        inner_grad.square().sum().backward()
        assert torch.isfinite(x.grad).all()
        return x, y, inner_grad

    torch.manual_seed(42)
    x0, out0, grad0 = run(False)
    torch.manual_seed(42)
    x1, out1, grad1 = run(True)
    torch.testing.assert_close(out0, out1)
    torch.testing.assert_close(grad0, grad1)
    torch.testing.assert_close(x0.grad, x1.grad, rtol=1e-4, atol=1e-5)
