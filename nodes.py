"""
ComfyUI Custom Nodes for AsymFLUX.2-klein-9B

Provides:
  - AsymFluxLoader:  Loads the base FLUX.2-klein model + AsymFLUX adapter from ComfyUI model folders
  - AsymFluxSampler: Runs pixel-space text-to-image generation (outputs IMAGE directly)

Follows the piFlow pattern:
  - Loader returns a pipeline object (like piFlow returns MODEL)
  - Sampler accepts CONDITIONING (like piFlow accepts CONDITIONING)
  - Text encoding is handled by ComfyUI's CLIP system via CLIP Text Encode nodes
"""

import numpy as np
import torch
import folder_paths
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
# Helper: extract context from ComfyUI conditioning (piFlow pattern)
# ---------------------------------------------------------------------------
def _extract_context_from_conditioning(conditioning):
    """
    Extract text context and pooled output from ComfyUI's CONDITIONING.

    ComfyUI CONDITIONING raw structure (as returned by CLIPTextEncode):
      conditioning = [
          branch_0: [                          # positive conditions
              [text_tensor_0, metadata_dict_0],
              [text_tensor_1, metadata_dict_1],
              ...
          ]
      ]

    Each item in a branch is a 2-element list: [tensor, dict].
    - Index [0] = the text embedding tensor (cross-attention context)
    - Index [1] = dict with metadata like "pooled_output", "guidance", etc.

    Note: During sampling ComfyUI's convert_cond() moves the tensor into
    the dict under "cross_attn", but nodes receive the raw format above.

    Returns (context_tensor, pooled_output) or (None, None) if empty.
    """
    if conditioning is None or len(conditioning) == 0:
        return None, None

    # CONDITIONING is list[list[...]] — first level is condition branch
    cond_list = conditioning[0] if isinstance(conditioning[0], list) else conditioning

    context_tensors = []
    pooled_tensors = []

    for cond_item in cond_list:
        # ComfyUI pattern: each item is [tensor, dict]
        if isinstance(cond_item, (list, tuple)) and len(cond_item) >= 2:
            cond_tensor = cond_item[0]
            cond_dict = cond_item[1]
        elif isinstance(cond_item, torch.Tensor):
            # Edge case: bare tensor in the list
            cond_tensor = cond_item
            cond_dict = {}
        else:
            continue

        # The text embedding tensor lives at index [0]
        if isinstance(cond_tensor, torch.Tensor):
            context_tensors.append(cond_tensor)
        elif isinstance(cond_dict, dict):
            # Fallback: after convert_cond() the tensor may be under "cross_attn"
            for key in ("cross_attn", "cond", "c_crossattn"):
                if key in cond_dict and isinstance(cond_dict[key], torch.Tensor):
                    context_tensors.append(cond_dict[key])
                    break

        # pooled_output lives in the metadata dict at index [1]
        if isinstance(cond_dict, dict) and "pooled_output" in cond_dict:
            pooled_tensors.append(cond_dict["pooled_output"])

    if not context_tensors:
        return None, None

    context = torch.cat(context_tensors, dim=0)
    pooled = torch.cat(pooled_tensors, dim=0) if pooled_tensors else None
    return context, pooled


# ---------------------------------------------------------------------------
# Loader Node (piFlow pattern: returns pipeline, no CLIP input)
# ---------------------------------------------------------------------------
class AsymFluxLoader:
    """
    Loads the FLUX.2-klein base model from ComfyUI's diffusion_models folder
    and attaches an AsymFLUX adapter.

    Follows the piFlow pattern: returns a pipeline object (like piFlow returns MODEL).
    Text encoding is handled separately by ComfyUI CLIP Text Encode nodes.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_model_name": (folder_paths.get_filename_list("diffusion_models"), {
                    "tooltip": "FLUX.2-klein base model from ComfyUI's diffusion_models folder.",
                }),
                "adapter_name": (folder_paths.get_filename_list("asymflux_adapters"), {
                    "tooltip": "AsymFLUX adapter weights from the asymflux_adapters folder.",
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

    def load(self, base_model_name, adapter_name, dtype, enable_cpu_offload):
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map[dtype]

        base_model_path = folder_paths.get_full_path_or_raise("diffusion_models", base_model_name)
        adapter_path = folder_paths.get_full_path_or_raise("asymflux_adapters", adapter_name)

        pipe = _get_or_load_pipe(
            base_model_path, adapter_path, "cuda", torch_dtype, enable_cpu_offload
        )
        return (pipe,)


# ---------------------------------------------------------------------------
# Sampler Node (piFlow pattern: accepts CONDITIONING as required inputs)
# ---------------------------------------------------------------------------
class AsymFluxSampler:
    """
    Runs AsymFLUX pixel-space text-to-image generation.

    Follows the piFlow pattern:
      - Accepts CONDITIONING (positive/negative) from CLIP Text Encode nodes
      - Extracts c_crossattn context tensors from conditioning
      - Passes them as prompt_embeds to the diffusers pipeline

    Outputs IMAGE directly (no separate VAE decode needed — pixel-space model).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("ASYM_PIPE",),
                "positive": ("CONDITIONING", {
                    "tooltip": "Positive conditioning from CLIP Text Encode.",
                }),
                "negative": ("CONDITIONING", {
                    "tooltip": "Negative conditioning from CLIP Text Encode.",
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
        self, pipeline, positive, negative, seed, steps,
        guidance_scale, orthogonal_guidance, clamp_denoised, width, height
    ):
        # Extract c_crossattn from conditioning (piFlow pattern)
        prompt_context, pooled = _extract_context_from_conditioning(positive)
        neg_context, neg_pooled = _extract_context_from_conditioning(negative)

        if prompt_context is None:
            raise RuntimeError("[AsymFLUX] No valid conditioning found in 'positive'. Connect a CLIP Text Encode node.")
        if neg_context is None:
            raise RuntimeError("[AsymFLUX] No valid conditioning found in 'negative'. Connect a CLIP Text Encode node.")

        # Debug: print embedding shapes and dtypes to diagnose issues
        print(f"[AsymFLUX] prompt_context shape={prompt_context.shape}, dtype={prompt_context.dtype}")
        print(f"[AsymFLUX] neg_context shape={neg_context.shape}, dtype={neg_context.dtype}")
        if pooled is not None:
            print(f"[AsymFLUX] pooled shape={pooled.shape}, dtype={pooled.dtype}")

        images = pipeline.generate(
            prompt_embeds=prompt_context,
            negative_prompt_embeds=neg_context,
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
