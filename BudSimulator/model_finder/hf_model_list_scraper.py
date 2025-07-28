from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.options import Options
import json
import time
import csv
from datetime import datetime
from urllib.parse import urljoin
import os
import re

class HuggingFaceSeleniumScraper:
    """Advanced scraper using Selenium for JavaScript-rendered content"""
    
    TASKS_DEFINITION = {
      "tasks": {
        "natural_language_processing": [
          {"task": "text-generation", "description": "Generate text from a prompt", "aliases": ["causal-lm"]},
          {"task": "text-classification", "description": "Classify text into predefined categories", "aliases": ["sentiment-analysis", "sequence-classification"]},
          {"task": "token-classification", "description": "Classify tokens in a sequence", "aliases": ["ner", "named-entity-recognition", "pos-tagging"]},
          {"task": "question-answering", "description": "Answer questions based on a given context", "aliases": ["qa", "extractive-qa"]},
          {"task": "fill-mask", "description": "Fill in masked tokens in a sequence", "aliases": ["masked-lm"]},
          {"task": "summarization", "description": "Generate summaries of longer texts", "aliases": ["text-summarization"]},
          {"task": "translation", "description": "Translate text from one language to another", "aliases": ["text-translation"]},
          {"task": "conversational", "description": "Generate responses in a conversational context", "aliases": ["dialogue", "chatbot"]},
          {"task": "feature-extraction", "description": "Extract features/embeddings from text", "aliases": ["embeddings", "sentence-similarity"]},
          {"task": "zero-shot-classification", "description": "Classify text without training on specific labels", "aliases": ["zero-shot"]},
          {"task": "table-question-answering", "description": "Answer questions about tabular data", "aliases": ["table-qa"]},
          {"task": "text2text-generation", "description": "General text-to-text generation tasks", "aliases": ["seq2seq"]}
        ],
        "audio": [
          {"task": "automatic-speech-recognition", "description": "Convert speech to text", "aliases": ["asr", "speech-to-text"]},
          {"task": "audio-classification", "description": "Classify audio into categories", "aliases": ["audio-tagging"]},
          {"task": "text-to-speech", "description": "Convert text to speech", "aliases": ["tts"]},
          {"task": "text-to-audio", "description": "Generate audio from text descriptions", "aliases": ["audio-generation"]},
          {"task": "audio-to-audio", "description": "Transform audio (e.g., source separation, enhancement)", "aliases": ["audio-processing"]},
          {"task": "voice-activity-detection", "description": "Detect speech segments in audio", "aliases": ["vad"]}
        ],
        "computer_vision": [
          {"task": "image-classification", "description": "Classify images into categories", "aliases": ["image-recognition"]},
          {"task": "image-segmentation", "description": "Segment images into regions", "aliases": ["semantic-segmentation", "instance-segmentation"]},
          {"task": "object-detection", "description": "Detect and locate objects in images", "aliases": ["detection"]},
          {"task": "image-to-image", "description": "Transform images (style transfer, enhancement, etc.)", "aliases": ["image-processing"]},
          {"task": "depth-estimation", "description": "Estimate depth from images", "aliases": ["monocular-depth-estimation"]},
          {"task": "image-to-text", "description": "Generate text descriptions from images", "aliases": ["image-captioning"]},
          {"task": "zero-shot-image-classification", "description": "Classify images without specific training", "aliases": ["zero-shot-vision"]},
          {"task": "unconditional-image-generation", "description": "Generate images without specific conditions", "aliases": ["image-generation"]},
          {"task": "video-classification", "description": "Classify videos into categories", "aliases": ["video-recognition"]},
          {"task": "keypoint-detection", "description": "Detect keypoints (e.g., pose estimation)", "aliases": ["pose-estimation"]}
        ],
        "multimodal": [
          {"task": "text-to-image", "description": "Generate images from text descriptions", "aliases": ["text2image"]},
          {"task": "visual-question-answering", "description": "Answer questions about images", "aliases": ["vqa"]},
          {"task": "document-question-answering", "description": "Answer questions about documents", "aliases": ["doc-vqa", "document-qa"]},
          {"task": "image-text-to-text", "description": "Generate text from image and text inputs", "aliases": ["vision-language-modeling"]},
          {"task": "text-to-video", "description": "Generate videos from text descriptions", "aliases": ["text2video"]},
          {"task": "any-to-any", "description": "Convert between any modalities", "aliases": ["multimodal-generation"]},
          {"task": "image-to-3d", "description": "Generate 3D models from images", "aliases": ["3d-reconstruction"]},
          {"task": "text-to-3d", "description": "Generate 3D models from text", "aliases": ["3d-generation"]}
        ],
        "tabular": [
          {"task": "tabular-classification", "description": "Classify tabular/structured data", "aliases": ["structured-data-classification"]},
          {"task": "tabular-regression", "description": "Regression on tabular data", "aliases": ["structured-data-regression"]}
        ],
        "reinforcement_learning": [
          {"task": "reinforcement-learning", "description": "Train agents using reinforcement learning", "aliases": ["rl"]}
        ],
        "time_series": [
          {"task": "time-series-forecasting", "description": "Forecast future values in time series", "aliases": ["forecasting"]}
        ],
        "graph": [
          {"task": "graph-ml", "description": "Machine learning on graph-structured data", "aliases": ["graph-neural-networks"]}
        ],
        "other": [
          {"task": "multiple-choice", "description": "Answer multiple choice questions", "aliases": ["mcq"]},
          {"task": "robotics", "description": "Control and planning for robotics", "aliases": ["robot-control"]}
        ]
      }
    }

    def __init__(self, headless=True):
        self.base_url = "https://huggingface.co"
        self.models_url = "https://huggingface.co/models"
        self.models = []
        self.driver = None
        self.headless = headless
        self.task_lookup_map = self._setup_task_lookup_map()
        
        # Set up data directory and file paths
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Use a fixed filename for the output JSON
        self.output_json_file = os.path.join(self.data_dir, 'hf_models.json')
        self.progress_file = os.path.join(self.data_dir, 'scraping_progress.txt')
        
        # Initialize the JSON file with empty array
        with open(self.output_json_file, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=4, ensure_ascii=False)
        
        # Initialize progress file
        with open(self.progress_file, 'w') as f:
            f.write(f"Scraping session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Target: {self.models_url}\n")
            f.write("="*50 + "\n")

    def _setup_task_lookup_map(self):
        """Builds a lookup map from task/alias to (modality, canonical_task)."""
        lookup_map = {}
        for modality, tasks in self.TASKS_DEFINITION["tasks"].items():
            for task_info in tasks:
                canonical_task = task_info["task"]
                # Add the main task name (e.g., "text-generation")
                lookup_map[canonical_task.lower()] = (modality, canonical_task)
                # Add all aliases (e.g., "causal-lm")
                for alias in task_info.get("aliases", []):
                    lookup_map[alias.lower()] = (modality, canonical_task)
        return lookup_map
        
    @staticmethod
    def check_chromedriver():
        """Check if ChromeDriver is available"""
        try:
            # Try to create a temporary driver to test
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            temp_driver = webdriver.Chrome(options=chrome_options)
            temp_driver.quit()
            return True
        except Exception as e:
            print(f"ChromeDriver not available: {e}")
            print("Please install ChromeDriver:")
            print("1. Visit https://chromedriver.chromium.org/")
            print("2. Download the version matching your Chrome browser")
            print("3. Add chromedriver to your PATH")
            print("4. Or install via: brew install chromedriver (on macOS)")
            return False
        
    def setup_driver(self):
        """Setup Chrome driver with options"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Initialize driver
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
            self.driver.set_page_load_timeout(30)
        except Exception as e:
            print(f"Error setting up Chrome driver: {e}")
            raise
        
    def close_driver(self):
        """Close the driver"""
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                print(f"Error closing driver: {e}")
            finally:
                self.driver = None
            
    def wait_for_models_to_load(self):
        """Wait for model cards to load on the page"""
        try:
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'article'))
            )
            time.sleep(2) # Additional wait for dynamic content
        except TimeoutException:
            print("Timeout waiting for models to load")
            return False
        return True
    
    def extract_models_from_page(self):
        """Extract all model information from the current page"""
        page_models = []
        model_elements = self.driver.find_elements(By.CSS_SELECTOR, 'article')
        
        for element in model_elements:
            try:
                model_info = self.extract_model_info(element)
                if model_info and model_info.get('uri'):
                    page_models.append(model_info)
            except Exception as e:
                print(f"Error extracting model info from element: {e}")
                continue
        
        return page_models
    
    def extract_model_info(self, element):
        """Extract detailed information from a model element"""
        model_info = {}
        try:
            # Get model URI and full URL from the main link
            link_element = element.find_element(By.CSS_SELECTOR, 'a[href]')
            href = link_element.get_attribute('href')
            if not href:
                return None
            
            model_info['uri'] = href.replace(self.base_url, '')
            model_info['full_url'] = href

            # Get the full text of the card for parsing
            card_text = element.text
            card_text_lower = card_text.lower()
            
            # Extract Parameters (e.g., 7B, 180B)
            param_match = re.search(r'\b(\d+(?:[.,]\d+)?(?:x\d+)?[KMBT]?B)\b', card_text, re.IGNORECASE)
            if param_match:
                model_info['parameters'] = param_match.group(1)

            # Extract modality and task by searching for task keywords in the card text
            model_info['modality'] = 'other'  # Default modality
            model_info['task'] = 'unknown'   # Default task
            # Sort keys by length to match longer keys first (e.g., 'text-to-image' before 'image')
            sorted_tasks = sorted(self.task_lookup_map.keys(), key=len, reverse=True)

            for task_keyword in sorted_tasks:
                modality, canonical_task = self.task_lookup_map[task_keyword]
                
                # Prepare search patterns for the keyword (e.g., 'text-generation' and 'text generation')
                patterns = [re.escape(task_keyword)]
                if '-' in task_keyword:
                    patterns.append(re.escape(task_keyword.replace('-', ' ')))
                
                search_pattern = r'\b(' + '|'.join(patterns) + r')\b'
                if re.search(search_pattern, card_text_lower):
                    model_info['modality'] = modality
                    model_info['task'] = canonical_task
                    break  # Stop after first match

            # Extract other info like downloads, likes
            info_parts = [part.strip() for part in card_text.split('•')]
            
            # Downloads and Likes
            for part in info_parts:
                part_lower = part.lower()
                # Downloads
                dl_match = re.search(r'([\d\.,]+[KMB]?)', part_lower)
                if dl_match and ('k' in part_lower or 'm' in part_lower or 'b' in part_lower):
                    if 'downloads' not in model_info:
                        model_info['downloads'] = dl_match.group(1).upper()

                # Likes
                if 'likes' in part_lower:
                    like_match = re.search(r'([\d\.,]+[KMB]?)', part)
                    if like_match:
                        model_info['likes'] = like_match.group(1)

            return model_info
            
        except Exception:
            # Fallback to at least get the URI if parsing fails
            try:
                link_element = element.find_element(By.CSS_SELECTOR, 'a[href]')
                href = link_element.get_attribute('href')
                if href:
                    return {'uri': href.replace(self.base_url, ''), 'full_url': href}
            except:
                return None
            return None

    def extract_model_details(self, url, wait_time=10):
        """
        Extract detailed model information from a Hugging Face model page.
        This method reuses the existing driver.
        """
        wait = WebDriverWait(self.driver, wait_time)
        
        # Initialize result dictionary
        result = {
            "modalities": [],
            "languages": [],
            "inference_engines": [],
            "model_details": {},
            "model_size": None,
            "adapters_finetunes_quantizations": {},
            "images": [],
            "github_repos": [],
            "arxiv_links": [],
            "model_website": url,
            "license": None
        }
        
        try:
            # Navigate to the page
            self.driver.get(url)
            
            # Wait for the main content to load
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "container")))
            
            # Extract model name
            try:
                model_name_elem = self.driver.find_element(By.CSS_SELECTOR, "h1 a.font-mono")
                result["model_details"]["name"] = model_name_elem.text.strip()
            except NoSuchElementException:
                try:
                    # Alternative selector
                    title = self.driver.title
                    result["model_details"]["name"] = title.split('·')[0].strip()
                except:
                    pass
            
            # Extract pipeline tag (modality)
            try:
                pipeline_tags = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/models?pipeline_tag='] .tag")
                for tag in pipeline_tags:
                    modality = tag.find_element(By.TAG_NAME, "span").text.strip()
                    result["modalities"].append(modality)
            except NoSuchElementException:
                pass
            
            # Extract languages
            try:
                lang_button = self.driver.find_element(By.XPATH, "//button[contains(.,'languages')]")
                lang_text = lang_button.text
                # Extract number from "24 languages" format
                lang_count = re.search(r'(\d+)\s*languages?', lang_text)
                if lang_count:
                    result["model_details"]["language_count"] = int(lang_count.group(1))
            except NoSuchElementException:
                pass
            
            # Extract from SVELTE_HYDRATER data attributes
            hydrater_elements = self.driver.find_elements(By.CSS_SELECTOR, "div.SVELTE_HYDRATER[data-props]")
            for elem in hydrater_elements:
                try:
                    data_props = elem.get_attribute("data-props")
                    if data_props:
                        props = json.loads(data_props)
                        
                        if 'model' in props:
                            model_data = props['model']
                            
                            # Languages
                            if 'cardData' in model_data and 'language' in model_data['cardData']:
                                result["languages"] = model_data['cardData']['language']
                            
                            # License
                            if 'cardData' in model_data and 'license' in model_data['cardData']:
                                result["license"] = model_data['cardData']['license']
                            
                            # Library name (inference engine)
                            if 'library_name' in model_data:
                                result["inference_engines"].append(model_data['library_name'])
                            
                            # Model config
                            if 'config' in model_data:
                                if 'architectures' in model_data['config']:
                                    result["model_details"]["architecture"] = model_data['config']['architectures']
                                if 'model_type' in model_data['config']:
                                    result["model_details"]["model_type"] = model_data['config']['model_type']
                            
                            # Downloads
                            if 'downloads' in model_data:
                                result["model_details"]["downloads"] = model_data['downloads']
                            
                            # Safetensors (model size)
                            if 'safetensors' in model_data:
                                safetensors = model_data['safetensors']
                                if 'total' in safetensors:
                                    result["model_size"] = {
                                        "total_params": safetensors['total'],
                                        "parameters": safetensors.get('parameters', {})
                                    }
                except (json.JSONDecodeError, NoSuchElementException):
                    continue
            
            # Extract model size from visible elements
            try:
                size_elements = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'params')]")
                for elem in size_elements:
                    text = elem.text
                    size_match = re.search(r'(\d+(?:\.\d+)?[BMK]?)\s*params?', text, re.I)
                    if size_match:
                        if not result["model_size"]:
                            result["model_size"] = {}
                        result["model_size"]["size_string"] = size_match.group(1)
                        break
            except NoSuchElementException:
                pass
            
            # Extract inference engines from tags
            try:
                tag_elements = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/models?other='] .tag span")
                for tag in tag_elements:
                    tag_text = tag.text.lower()
                    if tag_text in ['vllm', 'transformers', 'tensorrt', 'onnx']:
                        if tag_text not in [e.lower() for e in result["inference_engines"]]:
                            result["inference_engines"].append(tag.text)
            except NoSuchElementException:
                pass
            
            # Extract adapters, finetunes, quantizations from model tree
            try:
                model_tree_links = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/models?other=base_model:']")
                for link in model_tree_links:
                    href = link.get_attribute("href")
                    text = link.text.strip()
                    
                    if 'adapter:' in href:
                        result["adapters_finetunes_quantizations"]["adapters"] = text
                    elif 'finetune:' in href:
                        result["adapters_finetunes_quantizations"]["finetunes"] = text
                    elif 'quantized:' in href:
                        result["adapters_finetunes_quantizations"]["quantizations"] = text
            except NoSuchElementException:
                pass
            
            # Extract license from tags
            try:
                license_elem = self.driver.find_element(By.CSS_SELECTOR, "a[href*='license:'] .tag span:last-child")
                result["license"] = license_elem.text.strip()
            except NoSuchElementException:
                pass
            
            # Scroll to load more content
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(1)
            
            # Extract images from model card and code examples
            try:
                # Wait for model card to load
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "model-card-content")))
                
                # Get all text content
                page_source = self.driver.page_source
                
                # Extract image URLs
                image_pattern = r'https?://[^\s"\'\]<>]+\.(?:jpg|jpeg|png|gif|webp)'
                images = re.findall(image_pattern, page_source)
                
                # Filter out UI/avatar images
                result["images"] = [img for img in images if 'avatar' not in img and 'logo' not in img]
                result["images"] = list(set(result["images"]))[:10]  # Limit to 10 unique images
                
            except TimeoutException:
                pass
            
            # Extract GitHub repositories
            try:
                github_links = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='github.com']")
                for link in github_links:
                    href = link.get_attribute("href")
                    if href and 'github.com' in href:
                        # Clean up the URL
                        github_url = href.split('?')[0].split('#')[0]
                        if github_url not in result["github_repos"]:
                            result["github_repos"].append(github_url)
            except NoSuchElementException:
                pass
            
            # Extract arXiv links
            try:
                arxiv_links = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='arxiv.org']")
                for link in arxiv_links:
                    href = link.get_attribute("href")
                    if href and href not in result["arxiv_links"]:
                        result["arxiv_links"].append(href)
            except NoSuchElementException:
                pass
            
            # Extract additional model details from the model card
            try:
                # Key features
                features_heading = self.driver.find_element(By.XPATH, "//h2[contains(text(), 'Key Features')]")
                features_list = features_heading.find_element(By.XPATH, "following-sibling::ul[1]")
                features = features_list.find_elements(By.TAG_NAME, "li")
                result["model_details"]["key_features"] = [f.text.strip() for f in features]
            except NoSuchElementException:
                pass
            
            # Extract base model
            try:
                base_model_elem = self.driver.find_element(By.XPATH, "//p[contains(text(), 'Base model')]/following-sibling::a")
                result["model_details"]["base_model"] = base_model_elem.text.strip()
            except NoSuchElementException:
                pass
            
            # Extract usage/inference information
            try:
                usage_section = self.driver.find_element(By.XPATH, "//h2[contains(text(), 'Usage')]")
                usage_parent = usage_section.find_element(By.XPATH, "..")
                usage_text = usage_parent.text.lower()
                
                # Check for inference engines mentioned in usage
                if 'vllm' in usage_text and 'vLLM' not in result["inference_engines"]:
                    result["inference_engines"].append('vLLM')
                if 'transformers' in usage_text and 'transformers' not in result["inference_engines"]:
                    result["inference_engines"].append('transformers')
                if 'tensorrt' in usage_text and 'TensorRT' not in result["inference_engines"]:
                    result["inference_engines"].append('TensorRT')
            except NoSuchElementException:
                pass
            
        except Exception as e:
            print(f"Error occurred while scraping details from {url}: {str(e)}")
            result["error"] = str(e)
        
        # Clean up empty values
        result = {k: v for k, v in result.items() if v}
        
        return result

    def scrape_models(self, max_pages):
        """Scrape models for a specified number of pages."""
        if not self.driver:
            self.setup_driver()
        
        consecutive_empty_pages = 0
        
        try:
            for page in range(max_pages):
                # Instruction 6: Print page number / total pages, page URL
                url = f"{self.models_url}?p={page}&sort=trending"
                print("-" * 60)
                print(f"Scraping Page {page + 1} / {max_pages}")
                print(f"URL: {url}")

                try:
                    self.driver.get(url)
                    
                    if "404" in self.driver.title or "not found" in self.driver.title.lower():
                        print(f"Page {page} returned 404. This might be the last page. Stopping.")
                        break
                    
                    if not self.wait_for_models_to_load():
                        print("Warning: Timed out waiting for models to load.")
                        consecutive_empty_pages += 1
                        if consecutive_empty_pages >= 3:
                            print("Stopping after 3 consecutive failed pages.")
                            break
                        continue

                    # Instruction 2.1: Extract all model cards
                    page_models = self.extract_models_from_page()
                    
                    if not page_models:
                        print("Warning: No models found on this page. This might be the end.")
                        consecutive_empty_pages += 1
                        if consecutive_empty_pages >= 3:
                            print("Stopping after 3 consecutive pages with no models.")
                            break
                        continue
                    
                    # Scrape details for each model
                    for i, model_info in enumerate(page_models):
                        full_url = model_info.get('full_url')
                        if full_url:
                            print(f"  -> Scraping details for model {i+1}/{len(page_models)}: {model_info.get('uri', 'N/A')}")
                            try:
                                details = self.extract_model_details(full_url)
                                model_info['details'] = details
                            except Exception as e:
                                print(f"    Error scraping details for {full_url}: {e}")
                                model_info['details'] = {"error": str(e)}
                            time.sleep(0.5) # Be respectful to the server

                    consecutive_empty_pages = 0  # Reset on success
                    self.models.extend(page_models)
                    
                    # Instruction 3 & 4: Save to JSON after each page
                    self._save_results_to_json()

                    # Instruction 5: Give an update to the user
                    print(f"\nPage {page + 1} finished. Found {len(page_models)} models.")
                    print(f"Total models fetched so far: {len(self.models)}")
                    print("Latest models found on this page:")
                    for model in page_models[-4:]:
                        print(f"  - {model.get('uri', 'N/A')}")
                    
                    time.sleep(1)  # Be respectful to the server

                except Exception as e:
                    print(f"An error occurred on page {page + 1}: {e}")
                    time.sleep(5)  # Wait longer after an error
                    continue
                    
        finally:
            self.close_driver()
        
        print("-" * 60)
        print(f"\nScraping finished. Total models scraped: {len(self.models)}")
        return self.models
    
    def _save_results_to_json(self):
        """Saves the current list of models to the session's JSON file."""
        try:
            with open(self.output_json_file, 'w', encoding='utf-8') as f:
                json.dump(self.models, f, indent=4, ensure_ascii=False)
            
            with open(self.progress_file, 'a') as f:
                timestamp = datetime.now().strftime('%H:%M:%S')
                f.write(f"[{timestamp}] Page finished. Total models: {len(self.models)}\n")

        except Exception as e:
            print(f"Error saving results to JSON: {e}")
    
    def get_output_file(self):
        """Get the file path for the output JSON file."""
        return self.output_json_file


def main():
    """Main function to run the Selenium scraper."""
    print("Hugging Face Models Scraper (Selenium)")
    print("=====================================\n")
    
    # Check for ChromeDriver before asking for input
    if not HuggingFaceSeleniumScraper.check_chromedriver():
        print("❌ Cannot proceed without ChromeDriver. Please install it first.")
        return
    print("✅ ChromeDriver is available!")

    # Instruction 1: Take the input from the user on the number of pages
    while True:
        try:
            pages_to_scrape = input("Enter the number of pages to scrape: ").strip()
            if not pages_to_scrape:
                print("Please enter a number.")
                continue
            max_pages = int(pages_to_scrape)
            if max_pages > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a whole number.")

    headless = input("Run in headless mode? (y/n, default=y): ").strip().lower() != 'n'
    
    scraper = HuggingFaceSeleniumScraper(headless=headless)
    
    output_file = scraper.get_output_file()
    print(f"\n📁 Scraping session started. Results will be saved to:")
    print(f"   -> {output_file}")
    
    try:
        # Instructions 2-7 are handled inside scrape_models
        models = scraper.scrape_models(max_pages=max_pages)
        
        if models:
            print(f"\n🎉 Scraping completed successfully!")
            print(f"📊 Total models scraped: {len(models)}")
            print(f"💾 Final results are saved in: {output_file}")
        else:
            print("\nScraping finished, but no models were found.")

    except KeyboardInterrupt:
        print("\nScraping interrupted by user.")
        output_file = scraper.get_output_file()
        if scraper.models:
            print(f"Partial results ({len(scraper.models)} models) are saved in: {output_file}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during scraping: {e}")
        output_file = scraper.get_output_file()
        if scraper.models:
            print(f"Partial results ({len(scraper.models)} models) are saved in: {output_file}")
    finally:
        scraper.close_driver()


if __name__ == "__main__":
    main()