# AgentScheduler.py (FINAL, CORRECTED, AND MEMORY-OPTIMIZED VERSION)
import json
import time
from typing import Dict, Any, Optional
from PIL import Image
import os 
# Add import for image creation helper
# Note: Ensure you have AgentCore, ParserModule, SAMToolWorker, DiffusionToolWorker files

# --- Import ACTUAL Components (The Core of Phase 2 Integration) ---
from agent_core import LMMCore
from ParserModule import parse_tool_call
from SAMToolWorker import SAMToolWorker
from DiffusionToolWorker import DiffusionToolWorker


class AgentScheduler:
    def __init__(self):
        # 1. Initialize LMM Core (Ollama API) - The lightest component
        self.lmm_core = LMMCore() 
        
        # 2. Initialize Tool Workers
        print("\n[AGENT] Initializing Tool Workers...")
        
        # Instantiate workers. They will load large models, potentially slowly.
        # This approach ensures the necessary attributes exist for the run_tool method.
        self.sam_worker = SAMToolWorker()           
        self.diffusion_worker = DiffusionToolWorker() 

        # Map tool names (as output by LLaVA) to their worker methods
        self.tool_workers = {
            "SAM": self.sam_worker.run,
            "DIFFUSION": self.diffusion_worker.run
        }
        
        # Check initial state (Crucial for the warning message below)
        self.current_image_path: str = "initial_image.png"
        self.conversation_history: list = []
        self.max_tool_steps = 3 

    def run_tool(self, tool_call: Dict[str, Any], current_image_path: str, mask_path: Optional[str] = None) -> Dict[str, Any]:
        """Routes the parsed tool call and manages file path passing for chaining."""
        tool_name = tool_call['tool_name'].upper()
        arguments = tool_call['arguments']
        
        if tool_name in self.tool_workers:
            worker_func = self.tool_workers[tool_name]
            
            # --- ARGUMENT MAPPING (Core Chaining Logic) ---
            if tool_name == "SAM":
                # SAM needs the current image path and a new path for the mask output
                return worker_func(
                    image_path=current_image_path,
                    referring_expression=arguments.get('referring_expression', 'object'),
                    output_mask_path=arguments.get('output_path', 'temp_mask_0.png')
                )
            
            elif tool_name == "DIFFUSION":
                # DIFFUSION needs the current image, the MASK, the prompt, and an output path
                if not mask_path:
                     return {"success": False, "result_path": None, "message": "DIFFUSION called, but no mask path available from SAM."}
                     
                return worker_func(
                    image_path=current_image_path,
                    mask_path=mask_path, 
                    prompt=arguments.get('prompt', 'edit the image'),
                    output_image_path=arguments.get('output_path', 'final_edited_image.png')
                )
        else:
            return {"success": False, "result_path": None, "message": f"Unknown tool: {tool_name}"}

    def execute_task(self, user_prompt: str, initial_image_path: str):
        """The main orchestration loop for the Tri-Modal Agent (TMA)."""
        print(f"\n--- TMA Starting End-to-End Task ---")
        print(f"User Prompt: '{user_prompt}'")
        
        self.current_image_path = initial_image_path
        mask_path = None # State variable to hold the mask output from SAM
        lmm_input_prompt = user_prompt
        
        for step in range(self.max_tool_steps):
            print(f"\n--- AGENT STEP {step + 1} (Current Image: {self.current_image_path}) ---")
            
            # 1. LMM Call: Get the next step/tool decision
            raw_llm_output = self.lmm_core.generate(self.current_image_path, lmm_input_prompt)
            print(f"LMM Response (Start): {raw_llm_output[:80]}...")
            
            # 2. Parse LMM Output: Check for a structured tool call
            tool_call = parse_tool_call(raw_llm_output)
            
            if tool_call:
                print(f"SUCCESS: Parsed Tool Call: {json.dumps(tool_call)}")
                
                # 3. Execute Tool: Run the worker function
                tool_result = self.run_tool(tool_call, self.current_image_path, mask_path)
                
                if tool_result['success']:
                    # CRITICAL: Tool Chaining Logic
                    if tool_call['tool_name'] == "SAM":
                        mask_path = tool_result['result_path']
                        # Pass the original user prompt again, reminding the LLM of the next step
                        # Tell the LMM: SAM is done, now finish the original goal (the diffusion part)
                        feedback = (
                            f"TOOL_SUCCESS: SAM completed. Mask is saved at {mask_path}. "
                            "THE NEXT AND FINAL STEP IS TO CALL THE DIFFUSION TOOL. "
                            "Generate a structured tool call for DIFFUSION with the mask, the original image, "
                            "and a prompt to change the segmented object's color to blue."
                        )
                        #feedback = f"TOOL_SUCCESS: Tool {tool_call['tool_name']} completed. Mask saved at {mask_path}. What is the next step?"
                        
                    elif tool_call['tool_name'] == "DIFFUSION":
                        self.current_image_path = tool_result['result_path']
                        feedback = f"TOOL_SUCCESS: DIFFUSION completed. Final image saved at {self.current_image_path}. The task is fully complete. Final Answer."
                        #feedback = f"TOOL_SUCCESS: Tool {tool_call['tool_name']} completed. Final image saved at {self.current_image_path}. Answer the original query."
                    
                    print(f"Tool Execution Success: {tool_result['message']}")
                    lmm_input_prompt = feedback
                    
                else:
                    print(f"Tool Execution FAILED: {tool_result['message']}")
                    lmm_input_prompt = f"TOOL_FAILURE: Tool {tool_call['tool_name']} failed. Re-think the approach and provide a new tool call or final answer."
                
            else:
                # No tool call found, assume this is the final answer or a failure to reason.
                print("\nAGENT TERMINATION: Final Answer Received.")
                print(f"Final Output: {raw_llm_output}")
                return raw_llm_output 
        
        print("\nAGENT TERMINATION: Max steps reached.")
        return "Task stopped due to reaching maximum step limit."


# --- End-to-End Orchestration Test (Task 2.3) ---
if __name__ == "__main__":
    
    # Ensure a starting image exists for the test
    start_image_path = "initial_image.png"
    Image.open("Icon_Bird_512x512.png").save(start_image_path)

    start_image_path = "Icon_Bird_512x512.png" 
    #Image.new('RGB', (512, 512), color = 'yellow').save(start_image_path) 

    #test_instruction = "Segment the object in the middle and change its color to blue."
    test_instruction = "Segment the bird in the image and change its color to blue."
    
    # Initialize and run the full TMA
    print("\n[SCHEDULER] Initializing Agent Scheduler and Tools...")
    
    # IMPORTANT: The model loading is slow and happens here!
    scheduler = AgentScheduler() 
    
    # Optional check to confirm Diffusion load status after initialization
    if not hasattr(scheduler.diffusion_worker, 'initialized') or not scheduler.diffusion_worker.initialized:
        print("\n!!! WARNING: Diffusion Tool is running in MOCK/FAIL mode. Execution will use mock steps.")
    
    final_response = scheduler.execute_task(
        user_prompt=test_instruction, 
        initial_image_path=start_image_path
    )
    
    print("\n--- END-TO-END DEMO COMPLETE ---")
    print(f"Final Agent Response: {final_response}")
    print(f"Final output image: {scheduler.current_image_path}")

    # You can now look for final_edited_image.png in your directory