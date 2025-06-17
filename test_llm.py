from src.utils.llm_integration import parse_model_analysis, extract_json_from_string
from src.prompts import MODEL_ANALYSIS_PROMPT
from src.bud_ai import call_bud_LLM
from src.utils.text_extraction import extract_text_from_huggingface
import logging


logger = logging.getLogger(__name__)


hf_url = f"https://huggingface.co/Qwen/Qwen3-Embedding-8B"
logger.info(f"Extracting model description from {hf_url}")
            
model_description = extract_text_from_huggingface(hf_url)
            
  
 # Prepare prompt for LLM
full_prompt = MODEL_ANALYSIS_PROMPT + model_description
            
# Call LLM for analysis
logger.info(f"Calling LLM for model analysis of Qwen3-14B")
llm_response = call_bud_LLM(full_prompt)
parsed = parse_model_analysis(llm_response)
print(parsed)










