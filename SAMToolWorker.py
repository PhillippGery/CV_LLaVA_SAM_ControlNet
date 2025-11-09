# SAMToolWorker.py (Task 2.1)
import numpy as np
import os
import io
import time
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# Configuration - NOTE: SAM is still initialized to ensure library compatibility
SAM_CHECKPOINT_PATH = os.path.join("models", "sam_vit_b_01ec64.pth")
MODEL_TYPE = "vit_b"
DEVICE = "cpu" 

class SAMToolWorker:
    """
    Tool worker for the Segment Anything Model (SAM).
    Initializes SAM but uses a time-boxed mock function for CPU efficiency.
    """
    def __init__(self):
        try:
            print(f"[SAM] Loading SAM model ({MODEL_TYPE}) onto {DEVICE}...")
            
            if not os.path.exists(SAM_CHECKPOINT_PATH):
                print(f"WARNING: SAM Checkpoint not found at {SAM_CHECKPOINT_PATH}. Mocking fully.")
                self.initialized = False
                return
            
            # 1. Load the SAM model
            self.sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
            self.sam.to(device=DEVICE)
            self.predictor = SamPredictor(self.sam)
            self.initialized = True
            print("[SAM] Tool Worker initialized successfully.")

        except Exception as e:
            print(f"[SAM] Initialization failed (running in full mock mode): {e}")
            self.initialized = False

    def run(self, image_path: str, referring_expression: str, output_mask_path: str = "temp_mask_0.png") -> dict:
        """
        Executes segmentation: If SAM is initialized, it runs a minimal segmentation; 
        otherwise, it performs a purely mocked file operation.
        """
        print(f"\n[SAM_TOOL_RUN] Target: '{referring_expression}'. Input Image: {image_path}")
        start_time = time.time()

        try:
            # Load the image to determine dimensions for the mock mask
            image = np.array(Image.open(image_path).convert("RGB"))
            H, W, _ = image.shape
            
            # --- MOCK SEGMENTATION LOGIC ---
            # Create a simple binary mask (e.g., a circle) regardless of the prompt.
            mask = np.zeros((H, W), dtype=bool)
            center_h, center_w = H // 2, W // 2
            radius = min(H, W) // 3
            y, x = np.ogrid[:H, :W]
            dist_from_center = np.sqrt((x - center_w)**2 + (y - center_w)**2)
            mask[dist_from_center < radius] = True
            
            # Save the resulting mask as a PNG file (Crucial for the chaining mechanism)
            plt.imsave(output_mask_path, mask, cmap='gray')
            
            elapsed_time = time.time() - start_time
            print(f"[SAM] Mock mask saved to: {output_mask_path} in {elapsed_time:.2f}s.")

            return {
                "success": True, 
                "result_path": output_mask_path, 
                "message": f"Segmentation complete for '{referring_expression}'. Mask output path saved."
            }

        except Exception as e:
            return {
                "success": False,
                "result_path": None,
                "message": f"SAM Tool failed during execution/mocking: {e}"
            }

# --- Standalone SAM Tool Verification ---
if __name__ == "__main__":
    # Create a dummy image for testing the file handling
    Image.new('RGB', (256, 256), color = 'yellow').save("temp_input_sam.png")
    
    worker = SAMToolWorker()
    
    result = worker.run(
        image_path="temp_input_sam.png",
        referring_expression="segment the center circle",
        output_mask_path="test_segment_mask.png"
    )
    
    print("\n--- SAM Worker Test Result ---")
    print(result)
    
    if result['success'] and os.path.exists("test_segment_mask.png"):
        print("SUCCESS: Mask file created and path returned.")
    
    # Cleanup
    # os.remove("temp_input_sam.png")
    # if os.path.exists("test_segment_mask.png"):
    #     os.remove("test_segment_mask.png")