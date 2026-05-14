import os
import folder_paths

ASYMFLUX_ADAPTERS_DIR = os.path.join(folder_paths.models_dir, "asymflux_adapters")
os.makedirs(ASYMFLUX_ADAPTERS_DIR, exist_ok=True)
folder_paths.add_model_folder_path("asymflux_adapters", ASYMFLUX_ADAPTERS_DIR)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
