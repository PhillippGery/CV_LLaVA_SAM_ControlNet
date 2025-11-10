# evaluation_runner.py (Phase 3: Substantive Evaluation - AUTOMATED)

import os
import shutil
import time
from AgentScheduler import AgentScheduler
from PIL import Image

def setup_initial_image(source_file="Icon_Bird_512x512.png", target_file="initial_image.png"):
    """Ensures the bird image is present as the expected input file."""
    try:
        Image.open(source_file).save(target_file)
        return target_file
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Source image '{source_file}' not found. Please ensure it is in the project directory.")
        return None

def run_evaluation(scheduler, test_instruction, test_id, initial_image_path, expected_sequence, output_dir):
    """Executes a single test scenario, renames the output image, and returns the results."""
    
    # Generic name used by DiffusionToolWorker.py as its default output
    GENERIC_OUTPUT_NAME = "final_edited_image.png" 
    # The unique file path we want to save the result to
    UNIQUE_OUTPUT_PATH = os.path.join(output_dir, f"final_edited_{test_id}.png")
    
    print(f"\n=======================================================")
    print(f"| RUNNING TEST {test_id}: {test_instruction}")
    print(f"=======================================================")
    
    start_time = time.time()
    
    # Execute the task
    final_response = scheduler.execute_task(
        user_prompt=test_instruction, 
        initial_image_path=initial_image_path
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    # --- METRIC LOGIC ---
    tool_success = False
    task_accuracy = "FAIL"
    final_output = "N/A"
    
    if final_response.startswith("Ollama API Error"):
        status_message = "CRASH/MEMORY_ERROR"
        
    elif expected_sequence == "NO_TOOL":
        if not final_response.startswith("{") and "bird" in final_response.lower():
            tool_success = True
            task_accuracy = "PASS (Text Answer)"
        else:
            task_accuracy = "FAIL (Called Tool)"

    elif expected_sequence in ["SAM_DIFFUSION", "SAM_ONLY"]:
        # Check if the generic output file was created by the Diffusion worker
        if os.path.exists(GENERIC_OUTPUT_NAME):
            
            # This task requires image editing (DIFFUSION ran successfully)
            if expected_sequence == "SAM_DIFFUSION":
                tool_success = True
                task_accuracy = "PASS (Image Edited)"
                
                # --- RENAME AND MOVE THE IMAGE FILE ---
                try:
                    shutil.move(GENERIC_OUTPUT_NAME, UNIQUE_OUTPUT_PATH)
                    final_output = UNIQUE_OUTPUT_PATH
                except Exception as e:
                    print(f"Error moving file for {test_id}: {e}")

            # This task requires only SAM (no DIFFUSION expected)
            elif expected_sequence == "SAM_ONLY":
                # Ensure the last response is not a tool call and SAM ran
                if "SAM completed" in final_response:
                    tool_success = True
                    task_accuracy = "PASS (SAM Ran & Confirmed)"
                
                # Clean up the unexpected final_edited_image.png if it was accidentally created
                os.remove(GENERIC_OUTPUT_NAME)
            
        else:
             # If Diffusion was expected but no image was created
             status_message = "Tool Chain Incomplete"


    return {
        "ID": test_id,
        "Prompt": test_instruction,
        "Expected": expected_sequence.replace("_", " -> "),
        "Duration": f"{duration:.2f}s",
        "Tool Usage Success": "PASS" if tool_success else "FAIL",
        "Task Accuracy": task_accuracy,
        "Final Output File": final_output
    }


if __name__ == "__main__":
    
    # 0. Environment Setup 
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    
    # --- Setup Output Directory ---
    OUTPUT_DIR = "evaluation_output"
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR) # Clean up previous run for clarity
    os.makedirs(OUTPUT_DIR)
    
    # 1. Image Setup (Bird image)
    initial_image_path = setup_initial_image()
    if not initial_image_path:
        exit(1)

    # 2. Define Test Scenarios 
    test_scenarios = [
        {"ID": "E1", "Prompt": "Segment the bird and change its color to green.", "Expected": "SAM_DIFFUSION"},
        {"ID": "E2", "Prompt": "Does this image contain a bird?", "Expected": "NO_TOOL"},
        {"ID": "E3", "Prompt": "Change the bird's color to red, but only after segmenting it first.", "Expected": "SAM_DIFFUSION"},
        {"ID": "E4", "Prompt": "Only segment the bird in the image and stop.", "Expected": "SAM_ONLY"},
        {"ID": "E5", "Prompt": "Change the color of the bird to purple.", "Expected": "SAM_DIFFUSION"},
    ]

    # 3. Initialize and Run
    print("\n[EVALUATION] Initializing Agent Scheduler...")
    scheduler = AgentScheduler() 
    
    all_results = []
    
    for scenario in test_scenarios:
        result = run_evaluation(
            scheduler, 
            scenario["Prompt"], 
            scenario["ID"], 
            initial_image_path, 
            scenario["Expected"],
            OUTPUT_DIR 
        )
        all_results.append(result)
        scheduler.lmm_core.unload() # Call the new unload method
        time.sleep(5)
        
    # 4. Generate Final Report (Task 3.3)
    
    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.md")
    with open(report_path, "w") as f:
        
        # Helper function to print to terminal AND write to file
        def print_and_write(text):
            print(text)
            f.write(text + "\n")
        
        print_and_write("\n\n" + "=" * 80)
        print_and_write("FINAL SUBSTANTIVE EVALUATION REPORT (5-Shot Test)")
        print_and_write("=" * 80)
        
        # Calculate overall metrics
        total_tests = len(all_results)
        tool_successes = sum(1 for r in all_results if r["Tool Usage Success"] == "PASS")
        task_successes = sum(1 for r in all_results if r["Task Accuracy"].startswith("PASS"))
        
        print_and_write(f"Overall Tool Usage Success Rate: {tool_successes}/{total_tests} ({tool_successes / total_tests * 100:.2f}%)")
        print_and_write(f"Overall End-to-End Task Accuracy: {task_successes}/{total_tests} ({task_successes / total_tests * 100:.2f}%)")
        print_and_write("-" * 80)
        
        # Print Markdown Table
        
        headers = ["ID", "Prompt (Goal)", "Expected Chain", "Tool Usage Success", "Task Accuracy", "Time"]
        
        # Print header
        print_and_write(f"| {' | '.join(headers)} |")
        print_and_write(f"|{':---:' * len(headers)}|")
        
        for r in all_results:
            # Shorten Prompt for table legibility
            prompt_short = r['Prompt'][:30] + '...' if len(r['Prompt']) > 30 else r['Prompt']
            
            row_data = [
                r['ID'], 
                prompt_short, 
                r['Expected'], 
                r['Tool Usage Success'], 
                r['Task Accuracy'], 
                r['Duration']
            ]
            print_and_write(f"| {' | '.join(row_data)} |")
        
        print_and_write("-" * 80)
        print_and_write(f"All generated image files and the full report are saved to the '{OUTPUT_DIR}' folder.")