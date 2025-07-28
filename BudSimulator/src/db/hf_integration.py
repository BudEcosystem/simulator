"""
HuggingFace integration for importing models into the database.
"""

import sys
import os
import time
import json
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path

import requests
from lxml import html
from huggingface_hub import HfApi, hf_hub_download

# Add parent directory to path to import bud_models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bud_models import (
    HuggingFaceConfigLoader, 
    ModelMemoryCalculator,
    analyze_hf_model
)
from .model_manager import ModelManager
from .model_converter import ModelConverter

# Import new utilities
from ..utils.text_extraction import extract_text_from_huggingface
from ..utils.llm_integration import parse_model_analysis, validate_analysis
from ..prompts import MODEL_ANALYSIS_PROMPT


logger = logging.getLogger(__name__)


class HuggingFaceModelImporter:
    """Import models from HuggingFace into the database."""
    
    def __init__(self, model_manager: Optional[ModelManager] = None,
                 hf_token: Optional[str] = None):
        """Initialize the importer.
        
        Args:
            model_manager: Model manager instance
            hf_token: HuggingFace API token for private models
        """
        self.model_manager = model_manager or ModelManager()
        self.hf_loader = HuggingFaceConfigLoader(token=hf_token)
        self.memory_calculator = ModelMemoryCalculator()
        self.converter = ModelConverter()

        # Directory to store downloaded logos
        self.logos_dir = Path(__file__).resolve().parent.parent.parent / "logos"
        self.logos_dir.mkdir(exist_ok=True)
    
    # ------------------------------------------------------------------
    # Logo scraping utilities
    # ------------------------------------------------------------------

    def scrap_hf_logo(self, hf_url: str) -> Optional[str]:
        """Scrape the HuggingFace organization page to extract the logo URL.

        Args:
            hf_url: The model id/URI on HuggingFace (e.g. "mistralai/Mistral-7B").

        Returns:
            Absolute URL to the logo image if found, else None.
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }

        # Extract organization/user name (first part before "/")
        org_name = hf_url.split("/")[0] if "/" in hf_url else hf_url
        full_url = f"https://huggingface.co/{org_name}"

        try:
            time.sleep(1)  # gentle delay to avoid rate-limiting
            response = requests.get(full_url, headers=headers, timeout=10)
            response.raise_for_status()

            tree = html.fromstring(response.content)

            # Primary XPath selector (avatar in header)
            xpath_sel = "//main/header//img[@alt and (contains(@class,'rounded') or contains(@class,'avatar'))]"
            img_elems = tree.xpath(xpath_sel)

            # Fallback CSS selector
            if not img_elems:
                img_elems = tree.cssselect("header img")

            if img_elems:
                img_url = img_elems[0].get("src")
                if img_url and img_url.startswith("/"):
                    img_url = f"https://huggingface.co{img_url}"
                return img_url
            return None
        except Exception as e:
            logger.warning(f"Failed to scrape logo for {org_name}: {e}")
            return None

    def _save_logo(self, model_id: str, logo_url: Optional[str]) -> Optional[str]:
        """Download logo and save under logos/; return relative path for DB."""
        if not logo_url:
            return None

        try:
            resp = requests.get(logo_url, stream=True, timeout=10)
            resp.raise_for_status()

            ext = Path(logo_url).suffix or ".png"
            safe_name = model_id.replace("/", "_") + ext
            local_path = self.logos_dir / safe_name

            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Return relative path from project root so FastAPI static can serve
            return f"logos/{safe_name}"
        except Exception as e:
            logger.warning(f"Unable to download logo for {model_id}: {e}")
            return None

    def _parse_model_card(self, model_id: str) -> Dict[str, Any]:
        """Parses the model card to extract analysis, logo, etc."""
        try:
            model_info = self.hf_loader.get_model_info(model_id)
            card_data = getattr(model_info, 'cardData', {})
            
            analysis = {}
            if card_data.get('model-summary'):
                analysis['description'] = card_data['model-summary'].get('description')
                analysis['advantages'] = card_data['model-summary'].get('advantages')
                analysis['disadvantages'] = card_data['model-summary'].get('disadvantages')
                analysis['usecases'] = card_data['model-summary'].get('use-cases')
            
            eval_results = []
            if card_data.get('model-card-results'):
                for res in card_data['model-card-results']:
                    eval_results.append({
                        "name": res.get('metric'),
                        "score": res.get('value')
                    })
            analysis['evals'] = eval_results

            logo_url = getattr(model_info, 'author_avatar_url', None)

            return {
                "analysis": analysis,
                "logo": logo_url
            }
        except Exception as e:
            logger.error(f"Could not parse model card for {model_id}: {e}")
            return {"analysis": None, "logo": None}

    def _analyze_model_with_llm(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Analyze model using LLM by extracting and processing HuggingFace model card.
        
        Args:
            model_id: HuggingFace model identifier
            
        Returns:
            Model analysis dictionary or None if analysis fails
        """
        try:
            # Import here to avoid circular import
            from ..bud_ai import call_bud_LLM
            
            # Extract text from HuggingFace model page
            hf_url = f"https://huggingface.co/{model_id}"
            logger.info(f"Extracting model description from {hf_url}")
            
            model_description = extract_text_from_huggingface(hf_url)
            
            if not model_description or "Error" in model_description:
                logger.warning(f"Failed to extract model description: {model_description}")
                return None
            
            # Prepare prompt for LLM
            full_prompt = MODEL_ANALYSIS_PROMPT + model_description
            
            # Call LLM for analysis
            logger.info(f"Calling LLM for model analysis of {model_id}")
            llm_response = call_bud_LLM(full_prompt)
            
            # Parse and validate the analysis
            analysis = parse_model_analysis(llm_response)
            validated_analysis = validate_analysis(analysis)
            
            logger.info(f"Successfully analyzed model {model_id} with LLM")
            return validated_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze model {model_id} with LLM: {e}")
            return None

    def import_model(self, model_id: str, force_update: bool = False) -> bool:
        """Import a model from HuggingFace.
        
        Args:
            model_id: HuggingFace model identifier (e.g., "mistralai/Mistral-7B")
            force_update: If True, update existing model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if model already exists
            existing = self.model_manager.get_model(model_id)
            if existing and not force_update:
                logger.info(f"Model {model_id} already exists in database")
                return True
            
            # Fetch model configuration from HuggingFace
            logger.info(f"Fetching model {model_id} from HuggingFace")
            hf_config = self.hf_loader.get_model_config(model_id)
            
            # Detect model type and attention type
            model_type = self.memory_calculator.detect_model_type(hf_config)
            attention_type = self.memory_calculator.detect_attention_type(hf_config)
            
            # Get parameter count
            parameter_count = hf_config.get('num_parameters')
            
            # Scrape and save logo locally
            scraped_url = self.scrap_hf_logo(model_id)
            local_logo_path = self._save_logo(model_id, scraped_url)

            # Always use LLM analysis for model analysis
            logger.info(f"Using LLM analysis for {model_id}")
            model_analysis = self._analyze_model_with_llm(model_id)
            
            # Use the scraped logo
            logo = local_logo_path

            # Analyze model for additional metadata
            analysis = analyze_hf_model(model_id, token=self.hf_loader.token)
            
            if existing:
                # Update existing model
                success = self.model_manager.update_model(
                    model_id=model_id,
                    config=hf_config,
                    model_type=model_type,
                    attention_type=attention_type,
                    parameter_count=parameter_count,
                    logo=logo,
                    model_analysis=model_analysis,
                    change_description=f"Updated from HuggingFace"
                )
            else:
                # Add new model
                model_db_id = self.model_manager.add_model(
                    model_id=model_id,
                    config=hf_config,
                    source="huggingface",
                    model_type=model_type,
                    attention_type=attention_type,
                    parameter_count=parameter_count,
                    logo=logo,
                    model_analysis=model_analysis
                )
                success = model_db_id is not None
            
            if success:
                # Add quality metrics if available
                self._import_quality_metrics(model_id, analysis)
                
                # Cache memory analysis
                memory_analysis = analysis.get('memory_analysis', {})
                if memory_analysis:
                    cache_key = f"memory_analysis_2048_fp16"
                    self.model_manager.cache_value(
                        model_id, cache_key, memory_analysis, 
                        ttl_seconds=86400  # 24 hours
                    )
                
                logger.info(f"Successfully imported model {model_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to import model {model_id}: {e}")
            return False
    
    def import_models(self, model_ids: List[str], 
                     force_update: bool = False) -> Dict[str, bool]:
        """Import multiple models from HuggingFace.
        
        Args:
            model_ids: List of model identifiers
            force_update: If True, update existing models
            
        Returns:
            Dictionary mapping model IDs to success status
        """
        results = {}
        
        for model_id in model_ids:
            results[model_id] = self.import_model(model_id, force_update)
        
        return results
    
    def search_and_import(self, search_query: str, limit: int = 10,
                         filter_params: Optional[Dict[str, Any]] = None) -> List[str]:
        """Search HuggingFace for models and import them.
        
        Args:
            search_query: Search query
            limit: Maximum number of models to import
            filter_params: Additional filter parameters
            
        Returns:
            List of imported model IDs
        """
        # This would require HuggingFace Hub search API
        # For now, this is a placeholder
        logger.warning("Search and import functionality not yet implemented")
        return []
    
    def sync_existing_models(self) -> Dict[str, bool]:
        """Sync all existing models in the database with HuggingFace.
        
        Returns:
            Dictionary mapping model IDs to sync status
        """
        models = self.model_manager.list_models(source="huggingface")
        results = {}
        
        for model in models:
            model_id = model['model_id']
            try:
                # Re-import to update
                results[model_id] = self.import_model(model_id, force_update=True)
            except Exception as e:
                logger.error(f"Failed to sync model {model_id}: {e}")
                results[model_id] = False
        
        return results
    
    def _import_quality_metrics(self, model_id: str, analysis: Dict[str, Any]) -> None:
        """Import quality metrics from analysis results."""
        # Check if analysis contains quality metrics
        config = analysis.get('config', {})
        
        # Look for quality metrics in various places
        quality_sources = [
            config.get('model_quality'),
            analysis.get('quality_metrics'),
            analysis.get('benchmarks')
        ]
        
        for source in quality_sources:
            if source and isinstance(source, dict):
                for metric_name, metric_data in source.items():
                    if isinstance(metric_data, dict):
                        value = metric_data.get('value') or metric_data.get('accuracy')
                        if value is not None:
                            self.model_manager.add_quality_metric(
                                model_id=model_id,
                                metric_name=metric_name,
                                metric_value=float(value),
                                shots=metric_data.get('shots'),
                                metadata=metric_data
                            )
                    elif isinstance(metric_data, (int, float)):
                        self.model_manager.add_quality_metric(
                            model_id=model_id,
                            metric_name=metric_name,
                            metric_value=float(metric_data)
                        )
    
    def validate_import(self, model_id: str) -> Dict[str, Any]:
        """Validate that a model was imported correctly.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Validation results
        """
        validation = {
            'exists': False,
            'has_config': False,
            'has_metrics': False,
            'convertible': False,
            'errors': []
        }
        
        # Check if model exists
        model = self.model_manager.get_model(model_id)
        if not model:
            validation['errors'].append("Model not found in database")
            return validation
        
        validation['exists'] = True
        
        # Check config
        config = self.model_manager.get_model_config(model_id)
        if config:
            validation['has_config'] = True
        else:
            validation['errors'].append("Model configuration is missing")
        
        # Check metrics
        metrics = self.model_manager.get_quality_metrics(model_id)
        validation['has_metrics'] = len(metrics) > 0
        
        # Check if convertible to GenZ format
        try:
            genz_config = self.converter.hf_to_genz(config)
            validation['convertible'] = genz_config is not None
        except Exception as e:
            validation['errors'].append(f"Conversion error: {e}")
        
        return validation 