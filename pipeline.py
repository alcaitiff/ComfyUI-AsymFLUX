"""
AsymFLUX Pipeline Wrapper

Wraps the LakonLab PixelFlux2KleinPipeline for use in ComfyUI nodes.
This is a pixel-space model (not latent-space), so the output is directly
an image — no separate VAE decode step is needed.

Loads transformer weights from ComfyUI's diffusion_models folder (safetensors)
and text encoders/tokenizers from HuggingFace, following the piFlow pattern.
"""

import math
import torch
from safetensors.torch import load_file
from diffusers import Flux2Transformer2DModel
from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast
from lakonlab.models.architectures import OklabColorEncoder
from lakonlab.models.diffusions.schedulers import FlowAdapterScheduler
from lakonlab.pipelines.pipeline_pixelflux2_klein import PixelFlux2KleinPipeline


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

# HuggingFace repo for text encoders / tokenizers (shared with FLUX.2-klein)
TEXT_ENCODER_REPO = "black-forest-labs/FLUX.2-klein-base-9B"


def _load_transformer_from_safetensors(model_path, dtype):
    """
    Load a Flux2Transformer2DModel from a raw safetensors file.
    Follows the piFlow pattern: load state dict -> detect prefix -> build model -> load weights.
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

    # Build transformer from HF config, then load weights
    print("[AsymFLUX] Building Flux2Transformer2DModel from config...")
    transformer = Flux2Transformer2DModel.from_config(
        TEXT_ENCODER_REPO,
        subfolder="transformer",
    )
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
    Loads transformer from local safetensors (ComfyUI diffusion_models folder)
    and text encoders/tokenizers from HuggingFace.
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

        # --- Load text encoder and tokenizer from HuggingFace ---
        print("[AsymFLUX] Loading text encoder (Qwen3) from HuggingFace...")
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            TEXT_ENCODER_REPO,
            subfolder="text_encoder",
            torch_dtype=dtype,
        )
        print("[AsymFLUX] Loading tokenizer (Qwen2) from HuggingFace...")
        tokenizer = Qwen2TokenizerFast.from_pretrained(
            TEXT_ENCODER_REPO,
            subfolder="tokenizer",
        )

        # --- Build VAE and scheduler ---
        vae = OklabColorEncoder(**DEFAULT_VAE_CONFIG)
        scheduler = FlowAdapterScheduler(**DEFAULT_SCHEDULER_CONFIG)

        # --- Construct pipeline manually (bypass from_pretrained) ---
        self.pipe = PixelFlux2KleinPipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )

        # --- Load adapter using lakonlab's built-in PEFT adapter system ---
        print(f"[AsymFLUX] Loading adapter from: {adapter_path}")
        self.adapter_name = self.pipe.load_lakonlab_adapter(
            adapter_path,
            target_module_name='transformer',
        )
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
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 38,
        guidance_scale: float = 4.0,
        orthogonal_guidance: float = 1.0,
        clamp_denoised: bool = True,
        seed: int = 0,
    ):
        """
        Run text-to-image generation. Returns a PIL Image.
        """
        generator = torch.Generator(device="cpu").manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
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
