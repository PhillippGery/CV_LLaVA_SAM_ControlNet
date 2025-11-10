# DiffusionToolWorker.py (UPDATED for Sequential Loading)
import os
import time
from typing import Dict, Any
from PIL import Image
import numpy as np

# We need these imports outside the load function for the class definition
import torch 
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler

# Configuration
MODEL_ID = "runwayml/stable-diffusion-v1-5" 
DEVICE = "cpu"

class DiffusionToolWorker:
    """
    Tool worker for the Conditional Diffusion Model using deferred loading.
    """
    def __init__(self):
        self.pipeline = None
        self.initialized = False
        print("[DIFFUSION] Worker ready for deferred loading.")

    def load(self):
        """Loads the model into memory only when execution is needed."""
        if self.initialized:
            return
        try:
            print("[DIFFUSION] Loading model...")
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                MODEL_ID, 
                safety_checker=None,
            )
            pipeline.to(DEVICE) 
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            self.pipeline = pipeline
            self.initialized = True
            print("[DIFFUSION] Model loaded successfully.")
        except Exception as e:
            print(f"[DIFFUSION] Model load FAILED: {e}")
            self.initialized = False
            
    def unload(self):
        """Frees up memory after execution."""
        if self.initialized:
            del self.pipeline
            # Manual garbage collection to free up memory
            if 'torch' in globals():
                torch.cuda.empty_cache() 
            self.initialized = False
            print("[DIFFUSION] Model unloaded.")

    def run(self, image_path: str, mask_path: str, prompt: str, output_image_path: str = "final_edited_image.png") -> dict:
        """Executes the conditional image editing."""
        self.load() # CRITICAL: Load model now
        
        print(f"\n[DIFFUSION_TOOL_RUN] Prompt: '{prompt}'. Mask Input: {mask_path}")
        start_time = time.time()

        try:
            image = Image.open(image_path).convert("RGB")
            mask_image = Image.open(mask_path).convert("L") 

            if not self.initialized:
                # Mock Mode: If initialization failed, just save a copy of the original image
                image.save(output_image_path)
                status_message = "Diffusion MOCK: Saved original image as final output (pipeline failed to load)."
            else:
                # 2. Run the Inpaint Pipeline
                edited_image = self.pipeline(
                    prompt=prompt,
                    image=image,
                    mask_image=mask_image,
                    num_inference_steps=200,  # Using your preferred 100 steps
                    guidance_scale=7.5
                ).images[0]
                
                # 3. Save the result
                edited_image.save(output_image_path)
                status_message = "Diffusion executed and edited image saved."

            elapsed_time = time.time() - start_time
            print(f"[DIFFUSION] Output saved to: {output_image_path} in {elapsed_time:.2f}s.")

            return {
                "success": True, 
                "result_path": output_image_path, 
                "message": status_message
            }

        except Exception as e:
            print(f"Diffusion Tool failed during execution: {e}")
            return {
                "success": False,
                "result_path": None,
                "message": f"Diffusion Tool failed during execution: {e}"
            }
        finally:
            self.unload() # CRITICAL: Unload model after execution

if __name__ == "__main__":
    import torch 
    
    # 1. Setup dummy files (Requires a dummy image and the mask from the SAM test)
    if not os.path.exists("test_segment_mask.png"):
        print("Pre-requisite 'test_segment_mask.png' missing. Please run SAMToolWorker.py first.")
    else:
        # Create a dummy image 
        Image.new('RGB', (512, 512), color = 'yellow').save("temp_input_diff.png")
        
        worker = DiffusionToolWorker()
        
        # 2. Test execution (This will be SLOW, even with low steps!)
        result = worker.run(
            image_path="temp_input_diff.png",
            mask_path="test_segment_mask.png",
            prompt="A blue wooden car",
        )
        
        print("\n--- Diffusion Worker Test Result ---")
        print(result)
        
        if result['success'] and os.path.exists("test_diffusion_output.png"):
            print("SUCCESS: Output image file created and path returned.")