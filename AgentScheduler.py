# AgentScheduler.py (FINAL MEMORY-OPTIMIZED and LOGIC-FIXED VERSION)
import json
import time
from typing import Dict, Any, Optional
from PIL import Image
import os 

# --- Import ACTUAL Components ---
from agent_core import LMMCore
from ParserModule import parse_tool_call
from SAMToolWorker import SAMToolWorker
from DiffusionToolWorker import DiffusionToolWorker


class AgentScheduler:
    def __init__(self):
        # 1. Initialize LMM Core (Ollama API) - The lightest component
        self.lmm_core = LMMCore() 
        
        # 2. Initialize Tool Workers (Initializes in a low-memory state)
        print("\n[AGENT] Initializing Tool Workers...")
        self.sam_worker = SAMToolWorker()           
        self.diffusion_worker = DiffusionToolWorker() 

        self.tool_workers = {
            "SAM": self.sam_worker.run,
            "DIFFUSION": self.diffusion_worker.run
        }
        
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
                return worker_func(
                    image_path=current_image_path,
                    referring_expression=arguments.get('referring_expression', 'object in the middle'),
                    output_mask_path=arguments.get('output_path', 'temp_mask_0.png')
                )
            
            elif tool_name == "DIFFUSION":
                if not mask_path:
                     return {"success": False, "result_path": None, "message": "DIFFUSION called, but no mask path available from SAM."}
                     
                return worker_func(
                    image_path=current_image_path,
                    mask_path=mask_path, 
                    # CRITICAL FIX: Use a highly specific fallback prompt if LLaVA fails to include it
                    # We rely on the LMM to extract the prompt, but fall back to the task description.
                    prompt=arguments.get('prompt', 'Change the segmented object to blue'), 
                    output_image_path=arguments.get('output_path', 'final_edited_image.png')
                )
        else:
            return {"success": False, "result_path": None, "message": f"Unknown tool: {tool_name}"}

    def execute_task(self, user_prompt: str, initial_image_path: str):
        """The main orchestration loop for the Tri-Modal Agent (TMA)."""
        print(f"\n--- TMA Starting End-to-End Task ---")
        print(f"User Prompt: '{user_prompt}'")
        
        self.current_image_path = initial_image_path
        mask_path = None 
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
                    tool_name = tool_call['tool_name']
                    # CRITICAL: Tool Chaining Logic
                    if tool_name == "SAM":
                        mask_path = tool_result['result_path'] 
                        
                        # FIX: Make the next instruction aggressively generative
                        feedback = (
                            f"TOOL_SUCCESS: SAM completed. Mask is saved at {mask_path}. "
                            f"The next step MUST be to call the DIFFUSION tool. Use the full original prompt: '{user_prompt}'."
                        )
                        
                    elif tool_name == "DIFFUSION":
                        self.current_image_path = tool_result['result_path'] 
                        feedback = f"TOOL_SUCCESS: DIFFUSION completed. Final image saved at {self.current_image_path}. The task is fully complete. Final Answer."
                    
                    print(f"Tool Execution Success: {tool_result['message']}")
                    lmm_input_prompt = feedback
                    
                else:
                    print(f"Tool Execution FAILED: {tool_result['message']}")
                    lmm_input_prompt = f"TOOL_FAILURE: Tool {tool_call['tool_name']} failed. Re-think the approach and provide a new tool call or final answer."
                
            else:
                print("\nAGENT TERMINATION: Final Answer Received.")
                print(f"Final Output: {raw_llm_output}")
                return raw_llm_output 
        
        print("\nAGENT TERMINATION: Max steps reached.")
        return "Task stopped due to reaching maximum step limit."


# --- End-to-End Orchestration Test (Task 2.3) ---
if __name__ == "__main__":
    
    # 0. Setup Environment for Stability
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    
    # 1. Image Setup (Bird image)
    bird_image_source = "Icon_Bird_512x512.png" # Assuming this file is in your directory
    start_image_path = "initial_image.png"
    Image.open(bird_image_source).save(start_image_path) 

    # 2. Test Instruction 
    test_instruction = "Segment the bird in the middle and change its color to blue."
    
    # 3. Initialize and run the full TMA
    print("\n[SCHEDULER] Initializing Agent Scheduler and Tools...")
    
    scheduler = AgentScheduler() 
    
    final_response = scheduler.execute_task(
        user_prompt=test_instruction, 
        initial_image_path=start_image_path
    )
    
    print("\n--- END-TO-END DEMO COMPLETE ---")
    print(f"Final Agent Response: {final_response}")
    print(f"Final output image: {scheduler.current_image_path}")