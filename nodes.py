"""
ComfyUI Custom Nodes for AsymFLUX.2-klein-9B

Provides:
  - AsymFluxLoader:  Loads the base FLUX.2-klein model + AsymFLUX adapter from ComfyUI model folders
  - AsymFluxSampler: Runs pixel-space text-to-image generation (outputs IMAGE directly)

Follows the piFlow pattern: uses ComfyUI's CLIP system for text encoding instead of
loading text encoders from HuggingFace.
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
# Helper: extract prompt embeddings from ComfyUI conditioning
# ---------------------------------------------------------------------------
def _extract_embeds_from_conditioning(conditioning):
    """
    Extract prompt embeddings from ComfyUI's CONDITIONING output.
    
    CONDITIONING is a list of lists of dicts. Each dict contains:
    - 'pooled_output': pooled text features
    - 'txt_ids': text position IDs (for FLUX models)
    - 'output_hidden_states' or similar: the actual context/embeddings
    
    For FLUX-style models, we need the encoder_hidden_states (context) tensor.
    """
    if conditioning is None or len(conditioning) == 0:
        return None, None
    
    # conditioning is typically [[{...}]] - list of conditions, each with a list of cond items
    cond_list = conditioning[0] if isinstance(conditioning[0], list) else conditioning
    
    # Collect all context tensors from conditioning items
    context_tensors = []
    txt_ids_tensors = []
    
    for cond_item in cond_list:
        # The 'context' or 'output_hidden_states' key holds the text embeddings
        # ComfyUI stores this under various keys depending on the CLIP model
        context = None
        
        # Try common keys used by ComfyUI for text embeddings
        if 'output_hidden_states' in cond_item:
            context = cond_item['output_hidden_states']
        elif 'context' in cond_item:
            context = cond_item['context']
        elif 'encoder_hidden_states' in cond_item:
            context = cond_item['encoder_hidden_states']
        
        if context is not None:
            context_tensors.append(context)
        
        # Get txt_ids if available
        if 'txt_ids' in cond_item:
            txt_ids_tensors.append(cond_item['txt_ids'])
    
    if not context_tensors:
        return None, None
    
    # Concatenate all context tensors along batch dimension
    context = torch.cat(context_tensors, dim=0)
    txt_ids = torch.cat(txt_ids_tensors, dim=0) if txt_ids_tensors else None
    
    return context, txt_ids


# ---------------------------------------------------------------------------
# Loader Node
# ---------------------------------------------------------------------------
class AsymFluxLoader:
    """
    Loads the FLUX.2-klein base model from ComfyUI's diffusion_models folder
    and attaches an AsymFLUX adapter from the asymflux_adapters folder.
    
    Follows the piFlow pattern: text encoding is handled by ComfyUI's CLIP system,
    not loaded separately from HuggingFace.
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
            },
            "optional": {
                "clip": ("CLIP", {
                    "tooltip": "ComfyUI CLIP text encoder (loaded via CLIP Loader). Used for text encoding instead of HuggingFace downloads.",
                }),
            }
        }

    RETURN_TYPES = ("ASYM_PIPE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "AsymFLUX"
    DESCRIPTION = "Load the FLUX.2-klein base model and attach the AsymFLUX pixel-space adapter."

    def load(self, base_model_name, adapter_name, dtype, enable_cpu_offload, clip=None):
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map[dtype]

        base_model_path = folder_paths.get_full_path_or_raise("diffusion_models", base_model_name)
        adapter_path = folder_paths.get_full_path_or_raise("asymflux_adapters", adapter_name)

        if clip is not None:
            print(f"[AsymFLUX] Using ComfyUI CLIP for text encoding (like piFlow).")
        else:
            print("[AsymFLUX] Warning: No CLIP provided. Text encoding will use default empty embeddings.")

        pipe = _get_or_load_pipe(
            base_model_path, adapter_path, "cuda", torch_dtype, enable_cpu_offload
        )
        # Store the CLIP reference for use in sampling
        pipe.clip = clip
        return (pipe,)


# ---------------------------------------------------------------------------
# Sampler Node
# ---------------------------------------------------------------------------
class AsymFluxSampler:
    """
    Runs AsymFLUX pixel-space text-to-image generation.
    Uses ComfyUI's CONDITIONING system for text encoding (piFlow pattern).
    Outputs IMAGE directly (no separate VAE decode needed — this is a pixel-space model).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("ASYM_PIPE",),
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
            },
            "optional": {
                "positive": ("CONDITIONING", {
                    "tooltip": "Positive conditioning from CLIP text encode.",
                }),
                "negative": ("CONDITIONING", {
                    "tooltip": "Negative conditioning from CLIP text encode.",
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
        self, pipeline, seed, steps, guidance_scale, orthogonal_guidance, 
        clamp_denoised, width, height, positive=None, negative=None
    ):
        # Extract embeddings from conditioning (piFlow pattern)
        prompt_embeds, _ = _extract_embeds_from_conditioning(positive)
        negative_prompt_embeds, _ = _extract_embeds_from_conditioning(negative)
        
        # Fallback: if no conditioning provided, create empty embeddings
        if prompt_embeds is None:
            print("[AsymFLUX] Warning: No positive conditioning provided. Using empty prompt.")
            # Create minimal empty embeddings (batch=1, seq_len=1, hidden=4096*3 for 3 layers)
            prompt_embeds = torch.zeros(1, 1, 12288, dtype=pipeline.dtype, device=pipeline.device)
        
        if negative_prompt_embeds is None:
            print("[AsymFLUX] Warning: No negative conditioning provided. Using empty negative prompt.")
            negative_prompt_embeds = torch.zeros(1, 1, 12288, dtype=pipeline.dtype, device=pipeline.device)

        images = pipeline.generate(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
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
