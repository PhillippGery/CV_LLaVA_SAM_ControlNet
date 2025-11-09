# agent_core.py (UPDATED for Ollama API)
import requests
import base64
import os
from PIL import Image

OLLAMA_API_URL = "http://localhost:11434/api/generate"

class LMMCore:
    """
    Handles communication with a running Ollama server for LLaVA inference.
    """
    def __init__(self):
        # The model name pulled via 'ollama pull llava'
        self.model_name = "llava" 
        print(f"LLaVA Core setup using Ollama API for model: {self.model_name}.")
        self._test_connection()

    def _test_connection(self):
        # Test if the Ollama server is running
        try:
            response = requests.post(OLLAMA_API_URL, json={"model": self.model_name, "prompt": "Test"})
            if response.status_code != 200:
                 print(f"WARNING: Ollama server is running but returned status {response.status_code}.")
            else:
                 print("Ollama API connection verified.")
        except requests.exceptions.ConnectionError:
            print("\n!!! CRITICAL ERROR: Ollama API is not running or accessible at http://localhost:11434.")
            print("Please start the Ollama application on your Mac before running the agent.")
            raise

    def generate(self, image_path: str, prompt: str) -> str:
        """
        Performs the multimodal inference by sending image (Base64) + text prompt to Ollama.
        """
        
        # 1. Encode Image to Base64 (Required by Ollama for multimodal input)
        try:
            with Image.open(image_path) as img:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG") # Use PNG for better general compatibility
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        except FileNotFoundError:
            return "ERROR: Input image not found for LMMCore."
        except Exception as e:
            return f"ERROR: Image encoding failed: {e}"
            
        # 2. Construct the prompt structure required by LLaVA-Plus logic
        # We inject the LLaVA-Plus tool instruction template here.
        tool_instruction = (
            "You are an expert agent. Respond ONLY with a structured JSON tool call "
            "like {\"tool_name\": \"SAM\", \"arguments\": {...}} for tool use, "
            "or your final answer. The available tools are SAM and DIFFUSION."
        )
        
        full_prompt = f"{tool_instruction}\nUSER: {prompt}\nASSISTANT:"
        
        # 3. Send request to Ollama
        data = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,  # Wait for the full response
            
            # --- CRITICAL FIX: Use the 'format: json' flag to encourage structured output ---
            "options": {
                "temperature": 0.1 # Encourage deterministic (JSON) output
            },
            "format": "json" # Request JSON output format
        }
        
        # Add the image payload for multimodal models
        data["images"] = [img_str]

        try:
            print(f"-> Sending multimodal request to Ollama...")
            response = requests.post(OLLAMA_API_URL, json=data, timeout=120) # Increased timeout for CPU
            
            if response.status_code == 200:
                llm_output = response.json().get('response', '')
                return llm_output.strip()
            else:
                return f"Ollama API Error: Status {response.status_code} - {response.text}"

        except requests.exceptions.RequestException as e:
            return f"Connection Error to Ollama: {e}"

# Required for image handling in the final version
import io 

# Example Usage (for testing)
if __name__ == "__main__":
    # Create a dummy image for testing (You must have one for real testing)
    dummy_image = Image.new('RGB', (100, 100), color = 'red')
    dummy_image.save("test_image.png")
    
    try:
        core = LMMCore()
        test_prompt = "Segment the object in the image and describe it."
        raw_output = core.generate("test_image.png", test_prompt)
        print("\n--- LMM Raw Output ---")
        print(raw_output)
        
        # Clean up dummy image
        os.remove("test_image.png")
        
    except Exception as e:
        print(f"\nLMM Core Test Failed: {e}")