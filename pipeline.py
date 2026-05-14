"""
AsymFLUX Pipeline Wrapper

Wraps the LakonLab PixelFlux2KleinPipeline for use in ComfyUI nodes.
This is a pixel-space model (not latent-space), so the output is directly
an image — no separate VAE decode step is needed.

Loads transformer weights from ComfyUI's diffusion_models folder (safetensors)
and uses ComfyUI's CLIP system for text encoding, following the piFlow pattern.
"""

import math
import torch
from safetensors.torch import load_file
from lakonlab.models.architectures import OklabColorEncoder
from lakonlab.models.diffusions.schedulers import FlowAdapterScheduler
from lakonlab.pipelines.pipeline_pixelflux2_klein import PixelFlux2KleinPipeline
from lakonlab.models.architectures.asymflow.asymflux2 import _AsymFlux2Transformer2DModel


# Default scheduler parameters from the official example
DEFAULT_SCHEDULER_CONFIG = dict(
    shift=17.0,
    use_dynamic_shifting=True,
    base_seq_len=1024 ** 2,
    max_seq_len=2048 ** 2,
    base_logshift=math.log(17.0),
    max_logshift=math.log(34.0),
    dynamic_shifting_type='sqrt',
    base_scheduler='UniPCMultistep',
)

# Default Oklab color encoder parameters from the official example
DEFAULT_VAE_CONFIG = dict(
    use_affine_norm=True,
    mean=(0.56, 0.0, 0.01),
    std=0.16,
)

# FLUX.2-klein transformer config for _AsymFlux2Transformer2DModel
# (AsymFLUX requires the custom transformer with x_t/condition_latents support)
TRANSFORMER_CONFIG = {
    "patch_size": 16,
    "in_channels": 3,
    "base_rank": 128,
    "num_layers": 8,
    "num_single_layers": 48,
    "attention_head_dim": 128,
    "num_attention_heads": 48,
    "joint_attention_dim": 15360,
    "timestep_guidance_channels": 256,
    "mlp_ratio": 3.0,
    "axes_dims_rope": (32, 32, 32, 32),
    "rope_theta": 2000,
    "eps": 1e-6,
    "sigma_min": 1e-4,
    "num_timesteps": 1,
    "guidance_embeds": True,
}


def _load_transformer_from_safetensors(model_path, dtype):
    """
    Load an _AsymFlux2Transformer2DModel from a raw safetensors file.
    Uses LakonLab's custom AsymFLUX transformer (supports x_t, condition_latents args).
    Follows the piFlow pattern: load state dict -> detect prefix -> build model -> load weights.
    Uses an embedded config dict so no HuggingFace network call is needed.
    """
    print(f"[AsymFLUX] Loading state dict from: {model_path}")
    state_dict = load_file(model_path)

    # Strip any prefix (e.g. "transformer.") if present
    prefix = ""
    for key in list(state_dict.keys()):
        if key.startswith("transformer."):
            prefix = "transformer."
            break

    if prefix:
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}

    # Build transformer from local config using LakonLab's _AsymFlux2Transformer2DModel
    # (no HuggingFace download needed)
    print("[AsymFLUX] Building _AsymFlux2Transformer2DModel from local config...")
    transformer = _AsymFlux2Transformer2DModel(**TRANSFORMER_CONFIG)
    transformer.to(dtype=dtype)

    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[AsymFLUX] Warning: missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"[AsymFLUX] Warning: unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    return transformer


class AsymFluxPipeWrapper:
    """
    Manages loading and caching of the PixelFlux2KleinPipeline.
    Loads transformer from local safetensors (ComfyUI diffusion_models folder).
    Uses ComfyUI's CLIP system for text encoding (no HuggingFace downloads).
    """

    def __init__(
        self,
        base_model_path: str,
        adapter_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enable_cpu_offload: bool = False,
    ):
        self.device = device
        self.dtype = dtype
        self.enable_cpu_offload = enable_cpu_offload

        print(f"[AsymFLUX] Loading base model from: {base_model_path}")
        print(f"[AsymFLUX] Loading adapter from: {adapter_path}")
        print(f"[AsymFLUX] dtype={dtype}, device={device}")

        # --- Load transformer from local safetensors (piFlow pattern) ---
        transformer = _load_transformer_from_safetensors(base_model_path, dtype)

        # --- Build VAE and scheduler (no text encoder needed - using ComfyUI CLIP) ---
        vae = OklabColorEncoder(**DEFAULT_VAE_CONFIG)
        scheduler = FlowAdapterScheduler(**DEFAULT_SCHEDULER_CONFIG)

        # --- Construct pipeline with dummy text_encoder/tokenizer ---
        # We pass None for text_encoder/tokenizer since we'll provide pre-computed
        # prompt_embeds from ComfyUI's CLIP system instead (piFlow pattern)
        self.pipe = PixelFlux2KleinPipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=None,
            tokenizer=None,
            transformer=transformer,
        )

        # --- Load adapter weights directly from safetensors (piFlow pattern) ---
        # piFlow merges adapter weights into the base state dict BEFORE building
        # the model, then uses load_model_weights which iterates key-by-key and
        # silently skips mismatches. We replicate this pattern here: load the
        # adapter state dict, then manually apply only keys that match both name
        # AND shape — no load_state_dict call that can crash on size mismatch.
        print(f"[AsymFLUX] Loading adapter from: {adapter_path}")
        adapter_state_dict = load_file(adapter_path)

        # Strip any prefix if present (piFlow does the same with unet_prefix_from_state_dict)
        prefix = ""
        for key in list(adapter_state_dict.keys()):
            if key.startswith("transformer."):
                prefix = "transformer."
                break

        if prefix:
            adapter_state_dict = {k[len(prefix):]: v for k, v in adapter_state_dict.items()}

        # Build a name→shape map from the already-loaded transformer, then only
        # apply adapter keys whose shape matches (piFlow's load_model_weights does
        # this implicitly by iterating over the merged dict key-by-key).
        param_shapes = {n: p.shape for n, p in transformer.named_parameters()}
        applied = 0
        skipped_shape = 0
        skipped_missing = 0
        for key, tensor in adapter_state_dict.items():
            if key in param_shapes:
                if param_shapes[key] == tensor.shape:
                    # Direct in-place copy via the module's state dict (piFlow pattern).
                    transformer.state_dict()[key].copy_(tensor.to(device=transformer.state_dict()[key].device))
                    applied += 1
                else:
                    skipped_shape += 1
            else:
                skipped_missing += 1

        if skipped_shape:
            print(f"[AsymFLUX] Adapter: skipped {skipped_shape} key(s) with shape mismatch.")
        if skipped_missing:
            print(f"[AsymFLUX] Adapter: skipped {skipped_missing} key(s) not in model.")
        print(f"[AsymFLUX] Adapter applied: {applied} keys.")

        self.adapter_name = adapter_path  # track which adapter is loaded
        print(f"[AsymFLUX] Adapter loaded: {self.adapter_name}")

        # --- Device placement ---
        if self.enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()
            print("[AsymFLUX] CPU offloading enabled.")
        else:
            self.pipe = self.pipe.to(self.device)

        print("[AsymFLUX] Pipeline ready.")

    def generate(
        self,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 38,
        guidance_scale: float = 4.0,
        orthogonal_guidance: float = 1.0,
        clamp_denoised: bool = True,
        seed: int = 0,
    ):
        """
        Run text-to-image generation using pre-computed prompt embeddings.
        Returns a PIL Image.
        
        Args:
            prompt_embeds: Pre-computed text embeddings from ComfyUI CLIP (B, seq_len, hidden)
            negative_prompt_embeds: Negative prompt embeddings from ComfyUI CLIP
        """
        generator = torch.Generator(device="cpu").manual_seed(seed)

        result = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            orthogonal_guidance=orthogonal_guidance,
            clamp_denoised=clamp_denoised,
            generator=generator,
            output_type="pil",
        )

        return result.images
