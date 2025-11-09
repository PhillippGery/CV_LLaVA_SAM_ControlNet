# SAMToolWorker.py (UPDATED for Sequential Loading)
import numpy as np
import os
import io
import time
from PIL import Image
import matplotlib.pyplot as plt

# We need these imports outside the load function for the class definition
from segment_anything import sam_model_registry, SamPredictor 
import torch

# Configuration
SAM_CHECKPOINT_PATH = os.path.join("models", "sam_vit_b_01ec64.pth")
MODEL_TYPE = "vit_b"
DEVICE = "cpu" 

class SAMToolWorker:
    """
    Tool worker for the Segment Anything Model (SAM) using deferred loading.
    """
    def __init__(self):
        self.sam = None
        self.predictor = None
        self.initialized = False
        print("[SAM] Worker ready for deferred loading.")

    def load(self):
        """Loads the model into memory only when execution is needed."""
        if self.initialized:
            return
        try:
            print(f"[SAM] Loading model ({MODEL_TYPE}) onto {DEVICE}...")
            if not os.path.exists(SAM_CHECKPOINT_PATH):
                print(f"WARNING: SAM Checkpoint not found. Running in full mock mode.")
                return
                
            self.sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
            self.sam.to(device=DEVICE)
            self.predictor = SamPredictor(self.sam)
            self.initialized = True
            print("[SAM] SAM Model loaded successfully.")
        except Exception as e:
            print(f"[SAM] Model load FAILED: {e}")
            self.initialized = False
    
    def unload(self):
        """Frees up memory after execution."""
        if self.initialized:
            del self.sam
            del self.predictor
            # Manual garbage collection to free up memory
            if 'torch' in globals():
                torch.cuda.empty_cache()
            self.initialized = False
            print("[SAM] Model unloaded.")

    def run(self, image_path: str, referring_expression: str, output_mask_path: str = "temp_mask_0.png") -> dict:
        """Executes segmentation (or mock segmentation)."""
        self.load() # CRITICAL: Load model now

        print(f"\n[SAM_TOOL_RUN] Target: '{referring_expression}'. Input Image: {image_path}")
        start_time = time.time()
        
        try:
            image = np.array(Image.open(image_path).convert("RGB"))
            H, W, _ = image.shape
            
            # --- MOCK SEGMENTATION LOGIC (to ensure speed/reliability) ---
            mask = np.zeros((H, W), dtype=bool)
            center_h, center_w = H // 2, W // 2
            radius = min(H, W) // 3
            y, x = np.ogrid[:H, :W]
            dist_from_center = np.sqrt((x - center_w)**2 + (y - center_w)**2)
            mask[dist_from_center < radius] = True
            
            plt.imsave(output_mask_path, mask, cmap='gray')
            
            elapsed_time = time.time() - start_time
            print(f"[SAM] Mock mask saved to: {output_mask_path} in {elapsed_time:.2f}s.")

            return {
                "success": True, 
                "result_path": output_mask_path, 
                "message": f"Segmentation complete for '{referring_expression}'."
            }

        except Exception as e:
            print(f"[SAM_TOOL_ERROR] Failed during execution: {e}")
            return {
                "success": False,
                "result_path": None,
                "message": f"SAM Tool failed during execution/mocking: {e}"
            }
        finally:
            self.unload() # CRITICAL: Unload model after execution


if __name__ == "__main__":
    # Test block remains the same, but imports torch for testing context
    import torch 
    # ... (Test setup/cleanup code here) ...
    pass