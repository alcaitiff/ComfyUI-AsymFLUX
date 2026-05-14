"""
ComfyUI Custom Nodes for AsymFLUX.2-klein-9B

Provides:
  - AsymFluxLoader:  Loads the base FLUX.2-klein model + AsymFLUX adapter
  - AsymFluxSampler: Runs pixel-space text-to-image generation (outputs IMAGE directly)
"""

import numpy as np
import torch
from PIL import Image
from .pipeline import AsymFluxPipeWrapper


# ---------------------------------------------------------------------------
# Global pipeline cache — avoid reloading 9B params on every queue
# ---------------------------------------------------------------------------
_PIPE_CACHE = {}


def _get_or_load_pipe(base_model_path, adapter_path, device, dtype, enable_cpu_offload):
    """Return a cached pipeline, or load a new one if config changed."""
    cache_key = (base_model_path, adapter_path, device, str(dtype), enable_cpu_offload)
    if cache_key not in _PIPE_CACHE:
        _PIPE_CACHE.clear()  # only cache one config at a time
        _PIPE_CACHE[cache_key] = AsymFluxPipeWrapper(
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            device=device,
            dtype=dtype,
            enable_cpu_offload=enable_cpu_offload,
        )
    return _PIPE_CACHE[cache_key]


# ---------------------------------------------------------------------------
# Loader Node
# ---------------------------------------------------------------------------
class AsymFluxLoader:
    """
    Loads the FLUX.2-klein base model and attaches the AsymFLUX adapter.
    Outputs a pipeline wrapper object used by the AsymFLUX Sampler.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_model_path": ("STRING", {
                    "default": "black-forest-labs/FLUX.2-klein-base-9B",
                    "tooltip": "HuggingFace repo ID or local path to the FLUX.2-klein base model.",
                }),
                "adapter_path": ("STRING", {
                    "default": "Lakonik/AsymFLUX.2-klein-9B",
                    "tooltip": "HuggingFace repo ID or local path to the AsymFLUX adapter weights.",
                }),
                "device": (["cuda", "cpu"], {
                    "default": "cuda",
                }),
                "dtype": (["bf16", "fp16", "fp32"], {
                    "default": "bf16",
                    "tooltip": "Model precision. bf16 recommended for best quality/speed tradeoff.",
                }),
                "enable_cpu_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable sequential CPU offloading for low-VRAM GPUs.",
                }),
            }
        }

    RETURN_TYPES = ("ASYM_PIPE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "AsymFLUX"
    DESCRIPTION = "Load the FLUX.2-klein base model and attach the AsymFLUX pixel-space adapter."

    def load(self, base_model_path, adapter_path, device, dtype, enable_cpu_offload):
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map[dtype]

        pipe = _get_or_load_pipe(
            base_model_path, adapter_path, device, torch_dtype, enable_cpu_offload
        )
        return (pipe,)


# ---------------------------------------------------------------------------
# Sampler Node
# ---------------------------------------------------------------------------
class AsymFluxSampler:
    """
    Runs AsymFLUX pixel-space text-to-image generation.
    Outputs IMAGE directly (no separate VAE decode needed — this is a pixel-space model).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("ASYM_PIPE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "Text description of the image to generate.",
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Low quality, worst quality, blurry, deformed, bad anatomy",
                    "tooltip": "Text description of attributes to avoid.",
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Random seed for reproducible generation.",
                }),
                "steps": ("INT", {
                    "default": 38,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of denoising steps. 38 recommended by the authors.",
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance scale. 4.0 recommended.",
                }),
                "orthogonal_guidance": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Orthogonal guidance strength for CFG.",
                }),
                "clamp_denoised": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clamp denoised output at each step (recommended for pixel-space models).",
                }),
                "width": ("INT", {
                    "default": 960,
                    "min": 256,
                    "max": 4096,
                    "step": 16,
                    "tooltip": "Output image width in pixels (must be multiple of 16).",
                }),
                "height": ("INT", {
                    "default": 1280,
                    "min": 256,
                    "max": 4096,
                    "step": 16,
                    "tooltip": "Output image height in pixels (must be multiple of 16).",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    CATEGORY = "AsymFLUX"
    OUTPUT_TOOLTIPS = ("Generated image in pixel space.",)
    DESCRIPTION = "Generate an image using the AsymFLUX pixel-space model. Outputs IMAGE directly."

    def sample(
        self, pipeline, prompt, negative_prompt, seed, steps,
        guidance_scale, orthogonal_guidance, clamp_denoised, width, height
    ):
        images = pipeline.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            orthogonal_guidance=orthogonal_guidance,
            clamp_denoised=clamp_denoised,
            seed=seed,
        )

        # Convert PIL images to ComfyUI IMAGE tensor format: (B, H, W, C) float32 [0, 1]
        output_images = []
        for img in images:
            arr = np.array(img).astype(np.float32) / 255.0
            output_images.append(torch.from_numpy(arr))

        batch = torch.stack(output_images, dim=0)
        return (batch,)


# ---------------------------------------------------------------------------
# Node Registration
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "AsymFluxLoader": AsymFluxLoader,
    "AsymFluxSampler": AsymFluxSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AsymFluxLoader": "AsymFLUX Loader",
    "AsymFluxSampler": "AsymFLUX Sampler",
}
