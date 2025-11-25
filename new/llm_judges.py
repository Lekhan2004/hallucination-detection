import os
import json
from typing import Dict, Any, Optional
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import requests
# ==========================
# 1. The Wrapper Class
# ==========================

class GeminiWrapper:
    def __init__(self, api_key: str = None, model_name: str = "gemini-1.5-pro"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API Key not found. Set GEMINI_API_KEY env var.")
            
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Default config - we will override this for the judge
        self.generation_config = genai.GenerationConfig(
            temperature=0.0, # Low temperature for factual judging
            max_output_tokens=8192,
            response_mime_type="text/plain" 
        )

        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    def generate(self, prompt: str, json_mode: bool = False) -> str:
        # Adjust config based on mode
        if json_mode:
            self.generation_config.response_mime_type = "application/json"
        else:
            self.generation_config.response_mime_type = "text/plain"

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API Error: {str(e)}")

# ==========================
# 2. Judge Logic
# ==========================

def _build_judge_prompt(rec: Dict[str, Any]) -> str:
    """Constructs the prompt text."""
    problem = rec.get("problem", "")
    gold = rec.get("gold_answer_raw", "")
    model_ans = rec.get("our_final_answer_raw", "")
    full_resp = rec.get("full_response", "")

    return (
        "You are a strict hallucination detector for math word problems.\n\n"
        "You are given:\n"
        "1) The problem statement.\n"
        "2) The reference correct final numeric answer.\n"
        "3) A model's full chain-of-thought (explanation) and its final answer.\n\n"
        "You must evaluate whether the model's final answer is correct, and whether the "
        "explanation contains hallucinations: unsupported numbers, steps, or claims not "
        "grounded in the problem.\n\n"
        "Return a SINGLE JSON object with the following keys:\n"
        "  correct: boolean (true if final answer matches the reference answer)\n"
        "  hallucination: boolean (true if you see any hallucination in the reasoning)\n"
        "  hallucination_score: number between 0 and 1 (0 = no hallucination, 1 = very hallucinated)\n"
        "  reason: short natural language explanation\n\n"
        f"Problem:\n{problem}\n\n"
        f"Reference final answer: {gold}\n\n"
        f"Model final answer: {model_ans}\n\n"
        f"Model chain-of-thought:\n{full_resp}\n"
    )

def _parse_json_from_content(content: str) -> Optional[Dict[str, Any]]:
    """ robust parsing in case the model wraps JSON in markdown blocks """
    try:
        return json.loads(content)
    except Exception:
        pass
    
    # Fallback: extract substring between first { and last }
    try:
        start = content.index("{")
        end = content.rindex("}") + 1
        return json.loads(content[start:end])
    except Exception:
        return None

def deepseek_judge(rec: Dict[str, Any], api_key="AIzaSyBMg6v31a6eCYgALJ0W3cVww6xdMZY-vzE") -> Dict[str, Any]:
    """
    LLM-as-judge using the GeminiWrapper.
    """
    record_id = rec.get("id", "unknown")
    print(f"=== Gemini judge for id: {record_id} ===")

    try:
        # 1. Initialize Wrapper
        # We prefer passing the key via env var, but support argument passing
        llm = GeminiWrapper(api_key=api_key, model_name="gemini-2.5-pro")

        # 2. Build Prompt
        prompt = _build_judge_prompt(rec)

        # 3. Call API (Enable JSON Mode natively)
        # JSON mode guarantees the structure, so we don't need regex parsing usually
        raw_response = llm.generate(prompt, json_mode=True)
        
        print(f"Gemini raw response preview: {raw_response[:200]}...")

        # 4. Parse Response
        parsed_data = _parse_json_from_content(raw_response)
        
        if parsed_data:
            parsed_data["raw"] = raw_response
            return parsed_data
        else:
            return {"error": "json_parse_failed", "raw": raw_response}

    except Exception as e:
        print(f"Gemini Judge Failed: {e}")
        return {"error": "api_exception", "details": str(e)}


# ========================== # Mistral Self-Judge via Ollama # ========================== 

OLLAMA_MODEL_NAME = "mistral:7b-instruct" 
OLLAMA_URL = "http://127.0.0.1:11434/api/generate" 
def mistral_self_judge(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Self-evaluation judge using the same Mistral model in Ollama.

    Returns dict with keys:
      correct, hallucination, hallucination_score, reason
    or None on failure.
    """
    prompt = _build_judge_prompt(rec)

    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
        },
    }
    print(payload, "payload")

    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    print(data, "data")

    content = data.get("response", "")

    try:
        return json.loads(content)
    except Exception:
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            obj_str = content[start:end]
            return json.loads(obj_str)
        except Exception:
            return None
