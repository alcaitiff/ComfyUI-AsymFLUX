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


# ---------------------------------------------------------------------------
# Default scheduler parameters from the official example
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Default Oklab color encoder parameters from the official example
# ---------------------------------------------------------------------------
DEFAULT_VAE_CONFIG = dict(
    use_affine_norm=True,
    mean=(0.56, 0.0, 0.01),
    std=0.16,
)

# ---------------------------------------------------------------------------
# FLUX.2-klein transformer config for _AsymFlux2Transformer2DModel
# ---------------------------------------------------------------------------
TRANSFORMER_CONFIG = {
    "patch_size": 16,
    "in_channels": 3,
    "base_rank": 128,
    "num_layers": 8,
    "num_single_layers": 24,
    "attention_head_dim": 128,
    "num_attention_heads": 32,
    "joint_attention_dim": 12288,
    "timestep_guidance_channels": 256,
    "mlp_ratio": 3.0,
    "axes_dims_rope": (32, 32, 32, 32),
    "rope_theta": 2000,
    "eps": 1e-6,
    "sigma_min": 1e-4,
    "num_timesteps": 1,
    "guidance_embeds": False,  # FLUX.2-klein uses guidance_embeds=False
}


# ---------------------------------------------------------------------------
# Key mapping: ComfyUI double_blocks/single_blocks format → Diffusers format
# ---------------------------------------------------------------------------
def _convert_comfyui_to_diffusers_keys(state_dict):
    """
    Convert safetensors keys from ComfyUI/piFlow format (double_blocks, single_blocks)
    to Diffusers/LakonLab format (transformer_blocks, single_transformer_blocks).

    The FLUX.2-klein base model safetensors stored in ComfyUI's diffusion_models
    folder use the ComfyUI naming convention. The LakonLab _AsymFlux2Transformer2DModel
    inherits from Diffusers' Flux2Transformer2DModel which expects Diffusers keys.

    Returns a new state dict with converted keys and properly split/reshaped tensors.
    """
    converted = {}
    n_double = TRANSFORMER_CONFIG["num_layers"]       # 8
    n_single = TRANSFORMER_CONFIG["num_single_layers"] # 24
    num_heads = TRANSFORMER_CONFIG["num_attention_heads"]  # 32
    head_dim = TRANSFORMER_CONFIG["attention_head_dim"]      # 128
    hidden = num_heads * head_dim  # 4096

    # --- Double stream blocks: double_blocks.{i} → transformer_blocks.{i} ---
    for i in range(n_double):
        src = f"double_blocks.{i}"
        dst = f"transformer_blocks.{i}"

        # Image QKV: packed [Q|K|V] each of shape (hidden, hidden) → split into 3 tensors
        qkv_img = state_dict.get(f"{src}.img_attn.qkv.weight")
        if qkv_img is not None:
            converted[f"{dst}.attn.to_q.weight"] = qkv_img[:hidden, :]
            converted[f"{dst}.attn.to_k.weight"] = qkv_img[hidden:2*hidden, :]
            converted[f"{dst}.attn.to_v.weight"] = qkv_img[2*hidden:3*hidden, :]

        # Text (cross-attention) QKV: packed [Q|K|V] each of shape (hidden, joint_attention_dim)
        qkv_txt = state_dict.get(f"{src}.txt_attn.qkv.weight")
        if qkv_txt is not None:
            converted[f"{dst}.attn.add_q_proj.weight"] = qkv_txt[:hidden, :]
            converted[f"{dst}.attn.add_k_proj.weight"] = qkv_txt[hidden:2*hidden, :]
            converted[f"{dst}.attn.add_v_proj.weight"] = qkv_txt[2*hidden:3*hidden, :]

        # Image attention projection
        proj_img = state_dict.get(f"{src}.img_attn.proj.weight")
        if proj_img is not None:
            converted[f"{dst}.attn.to_out.0.weight"] = proj_img

        # Text attention projection
        proj_txt = state_dict.get(f"{src}.txt_attn.proj.weight")
        if proj_txt is not None:
            converted[f"{dst}.attn.to_add_out.weight"] = proj_txt

        # Image MLP
        mlp_0 = state_dict.get(f"{src}.img_mlp.0.weight")
        if mlp_0 is not None:
            converted[f"{dst}.ff.linear_in.weight"] = mlp_0
        mlp_2 = state_dict.get(f"{src}.img_mlp.2.weight")
        if mlp_2 is not None:
            converted[f"{dst}.ff.linear_out.weight"] = mlp_2

        # Text MLP
        txt_mlp_0 = state_dict.get(f"{src}.txt_mlp.0.weight")
        if txt_mlp_0 is not None:
            converted[f"{dst}.ff_context.linear_in.weight"] = txt_mlp_0
        txt_mlp_2 = state_dict.get(f"{src}.txt_mlp.2.weight")
        if txt_mlp_2 is not None:
            converted[f"{dst}.ff_context.linear_out.weight"] = txt_mlp_2

        # Image normalization scales
        q_norm = state_dict.get(f"{src}.img_attn.norm.query_norm.scale")
        if q_norm is not None:
            converted[f"{dst}.attn.norm_q.weight"] = q_norm
        k_norm = state_dict.get(f"{src}.img_attn.norm.key_norm.scale")
        if k_norm is not None:
            converted[f"{dst}.attn.norm_k.weight"] = k_norm

        # Text normalization scales
        tq_norm = state_dict.get(f"{src}.txt_attn.norm.query_norm.scale")
        if tq_norm is not None:
            converted[f"{dst}.attn.norm_added_q.weight"] = tq_norm
        tk_norm = state_dict.get(f"{src}.txt_attn.norm.key_norm.scale")
        if tk_norm is not None:
            converted[f"{dst}.attn.norm_added_k.weight"] = tk_norm

    # --- Single stream blocks: single_blocks.{i} → single_transformer_blocks.{i} ---
    for i in range(n_single):
        src = f"single_blocks.{i}"
        dst = f"single_transformer_blocks.{i}"

        # linear1 contains both attn (qkv+mlp_in) and the projection
        # In Diffusers Flux2: attn.to_qkv_mlp_proj.weight + attn.to_out.weight are separate
        # In ComfyUI: linear1 = same shape as to_qkv_mlp_proj, linear2 = to_out
        linear1 = state_dict.get(f"{src}.linear1.weight")
        if linear1 is not None:
            converted[f"{dst}.attn.to_qkv_mlp_proj.weight"] = linear1

        linear2 = state_dict.get(f"{src}.linear2.weight")
        if linear2 is not None:
            converted[f"{dst}.attn.to_out.weight"] = linear2

        # Norm scales (stored at single_blocks.{i}.norm.*)
        q_norm = state_dict.get(f"{src}.norm.query_norm.scale")
        if q_norm is not None:
            converted[f"{dst}.attn.norm_q.weight"] = q_norm
        k_norm = state_dict.get(f"{src}.norm.key_norm.scale")
        if k_norm is not None:
            converted[f"{dst}.attn.norm_k.weight"] = k_norm

    # --- Top-level modules ---
    # Input projections (1:1 mapping)
    x_embedder = state_dict.get("x_embedder.weight")
    if x_embedder is not None:
        converted["x_embedder.weight"] = x_embedder

    context_embedder = state_dict.get("context_embedder.weight")
    if context_embedder is not None:
        converted["context_embedder.weight"] = context_embedder

    # Time + guidance embedding
    time_in_linear1 = state_dict.get("time_in.in_layer.weight")
    if time_in_linear1 is not None:
        converted["time_guidage_embed.timestep_embedder.linear_1.weight"] = time_in_linear1
    time_in_linear2 = state_dict.get("time_in.out_layer.weight")
    if time_in_linear2 is not None:
        converted["time_guidance_embed.timestep_embedder.linear_2.weight"] = time_in_linear2

    # Note: FLUX.2-klein has guidance_embeds=False, so guidance_in keys may not exist

    # Stream modulation (ComfyUI: .linear → Diffusers: .lin)
    mod_img = state_dict.get("double_stream_modulation_img.linear.weight")
    if mod_img is not None:
        converted["double_stream_modulation_img.lin.weight"] = mod_img

    mod_txt = state_dict.get("double_stream_modulation_txt.linear.weight")
    if mod_txt is not None:
        converted["double_stream_modulation_txt.lin.weight"] = mod_txt

    mod_single = state_dict.get("single_stream_modulation.linear.weight")
    if mod_single is not None:
        converted["single_stream_modulation.lin.weight"] = mod_single

    # Output head
    proj_out = state_dict.get("final_layer.linear.weight")
    if proj_out is not None:
        converted["final_layer.linear.weight"] = proj_out

    norm_out = state_dict.get("final_layer.adaLN_modulation.1.weight")
    if norm_out is not None:
        # swap_scale_shift: FLUX normalizes with (scale, shift) swapped vs standard LayerNorm
        # Diffusers Flux2 uses AdaLayerNormZero which expects [shift, scale] order
        # The weight shape is (hidden, 2) — first half = shift, second half = scale
        # For Flux2's swap_scale_shift normalization, we need to swap the halves
        if norm_out.shape[1] == 2 * hidden:
            shift_part = norm_out[:, :hidden]
            scale_part = norm_out[:, hidden:]
            swapped = torch.cat([scale_part, shift_part], dim=1)
            converted["final_layer.adaLN_modulation.1.weight"] = swapped
        else:
            converted["final_layer.adaLN_modulation.1.weight"] = norm_out

    # Position embedding (pos_embed is a buffer, not a weight key — skip it)

    return converted


def _load_transformer_from_safetensors(model_path, dtype):
    """
    Load an _AsymFlux2Transformer2DModel from a ComfyUI-format safetensors file.
    
    Converts keys from ComfyUI format (double_blocks.*, single_blocks.*) to 
    Diffusers/LakonLab format (transformer_blocks.*, single_transformer_blocks.*).
    
    Follows the piFlow pattern: load state dict → convert keys → build model → load weights.
    """
    print(f"[AsymFLUX] Loading state dict from: {model_path}")
    state_dict = load_file(model_path)

    # Convert ComfyUI key format to Diffusers/LakonLab key format
    print("[AsymFLUX] Converting keys from ComfyUI format to Diffusers format...")
    converted_sd = _convert_comfyui_to_diffusers_keys(state_dict)
    
    # Clean up original state dict from memory
    del state_dict

    # Build transformer from local config using LakonLab's _AsymFlux2Transformer2DModel
    print("[AsymFLUX] Building _AsymFlux2Transformer2DModel from local config...")
    transformer = _AsymFlux2Transformer2DModel(**TRANSFORMER_CONFIG)
    transformer.to(dtype=dtype)

    missing, unexpected = transformer.load_state_dict(converted_sd, strict=False)
    if missing:
        print(f"[AsymFLUX] Warning: missing keys ({len(missing)}): {missing[:10]}")
    if unexpected:
        print(f"[AsymFLUX] Warning: unexpected keys ({len(unexpected)}): {unexpected[:10]}")

    # Clean up converted state dict from memory
    del converted_sd

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

        # --- Load transformer from local safetensors (converted from ComfyUI format) ---
        transformer = _load_transformer_from_safetensors(base_model_path, dtype)

        # --- Build VAE and scheduler (no text encoder needed - using ComfyUI CLIP) ---
        vae = OklabColorEncoder(**DEFAULT_VAE_CONFIG)
        vae.to(device=self.device, dtype=self.dtype)
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

        # --- Load adapter weights ---
        # The adapter from Lakonik/AsymFLUX.2-klein-9B uses Diffusers format keys
        # (transformer_blocks.*, single_transformer_blocks.*) since it's exported
        # from the HuggingFace repo. We load it directly into the transformer.
        print(f"[AsymFLUX] Loading adapter from: {adapter_path}")
        adapter_state_dict = load_file(adapter_path)

        # Check if adapter uses ComfyUI format (double_blocks.*) or Diffusers format (transformer_blocks.*)
        uses_comfyui_format = any(k.startswith("double_blocks.") for k in adapter_state_dict.keys())
        
        if uses_comfyui_format:
            print("[AsymFLUX] Adapter is in ComfyUI format, converting to Diffusers format...")
            adapter_state_dict = _convert_comfyui_to_diffusers_keys(adapter_state_dict)

        # Merge adapter weights into transformer using manual key-by-key copying.
        # Only update keys that exist in BOTH the base model and adapter,
        # preserving all other base model weights intact.
        merged_count = 0
        skipped_unexpected = 0
        transformer_state_dict = transformer.state_dict()
        
        for key, value in adapter_state_dict.items():
            if key in transformer_state_dict:
                if transformer_state_dict[key].shape == value.shape:
                    transformer_state_dict[key] = value.to(self.dtype)
                    merged_count += 1
                else:
                    print(f"[AsymFLUX] Adapter shape mismatch for {key}: "
                          f"adapter={value.shape}, model={transformer_state_dict[key].shape}")
                    skipped_unexpected += 1
            else:
                skipped_unexpected += 1
        
        transformer.load_state_dict(transformer_state_dict, strict=True)
        del transformer_state_dict
        
        print(f"[AsymFLUX] Adapter merged: {merged_count} keys updated, {skipped_unexpected} keys skipped")

        # Clean up adapter state dict from memory
        del adapter_state_dict
        
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
                          or (seq_len, hidden) - batch dimension will be added if missing
            negative_prompt_embeds: Negative prompt embeddings from ComfyUI CLIP
        """
        # Ensure prompt embeddings have a batch dimension.
        # ComfyUI's CLIP outputs tensors of shape (seq_len, hidden_dim), but the
        # PixelFlux2KleinPipeline expects (B, seq_len, hidden_dim). Without this
        # fix, prompt_embeds.shape[0] returns seq_len instead of batch_size=1,
        # causing the denoising loop to run with a wildly incorrect batch size.
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if negative_prompt_embeds.dim() == 2:
            negative_prompt_embeds = negative_prompt_embeds.unsqueeze(0)

        # Ensure prompt embeddings are on the same device and dtype as the model
        # ComfyUI's CLIP may output tensors on CPU in float32, but the transformer is on GPU in bfloat16
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(device=self.device, dtype=self.dtype)

        generator = torch.Generator(device="cpu").manual_seed(seed)

        # Pass pre-computed embeddings as plain tensors (not wrapped in lists).
        # The lakonlab PixelFlux2KleinPipeline.__call__ method determines batch_size
        # from prompt_embeds.shape[0] when prompt=None (line 147). Wrapping in a list
        # causes AttributeError: 'list' object has no attribute 'shape'.
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