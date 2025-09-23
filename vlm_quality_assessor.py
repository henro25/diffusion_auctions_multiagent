"""
VLM Quality Assessor Module

This module provides Vision Language Model (VLM) based image quality assessment
for the diffusion auctions project. It supports multiple VLMs with fallback
capabilities and batch processing for efficient analysis.

Authors: Lillian Sun, Warren Zhu, Henry Huang
Academic Context: 4th Year research project on multi-winner auctions for generative AI
"""

import os
import json
import logging
import hashlib
import traceback
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import time

import torch
from PIL import Image
from tqdm import tqdm


class VLMQualityAssessor:
    """Vision Language Model Quality Assessor with multi-model support and fallback."""

    def __init__(self, config_path: str = "config/vlm_config.json"):
        """Initialize the VLM Quality Assessor.

        Args:
            config_path: Path to the VLM configuration file
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.processors = {}
        self.loaded_models = []

        # Setup logging
        self._setup_logging()

        # Setup cache
        self.cache_dir = Path(self.config["processing"]["cache_dir"])
        self.cache_dir.mkdir(exist_ok=True)

        self.logger.info("VLM Quality Assessor initialized")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {config_path}: {e}")

    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config["logging"]

        # Create logs directory
        os.makedirs(os.path.dirname(log_config["log_file"]), exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger("VLMAssessor")
        self.logger.setLevel(getattr(logging, log_config["level"]))

        # File handler
        handler = logging.FileHandler(log_config["log_file"])
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def load_models(self) -> None:
        """Load VLM models according to configuration."""
        self.logger.info("Loading VLM models...")

        # Try to load primary model
        try:
            self._load_primary_model()
        except Exception as e:
            self.logger.error(f"Failed to load primary model: {e}")
            self.logger.info("Attempting to load fallback models...")

        # Load fallback models
        for model_config in self.config["fallback_models"]:
            if model_config["enabled"]:
                try:
                    self._load_fallback_model(model_config)
                except Exception as e:
                    self.logger.error(f"Failed to load {model_config['name']}: {e}")

        if not self.loaded_models:
            raise RuntimeError("No VLM models could be loaded. Check your configuration and dependencies.")

        self.logger.info(f"Successfully loaded models: {self.loaded_models}")

    def _load_primary_model(self) -> None:
        """Load the primary Qwen2.5-VL model."""
        model_config = self.config["primary_model"]
        model_name = model_config["name"]

        if not model_config["enabled"]:
            raise ValueError("Primary model is disabled in configuration")

        self.logger.info(f"Loading primary model: {model_name}")

        try:
            # Import required libraries for Qwen2.5-VL
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info

            # Load model
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_config["model_id"],
                torch_dtype=getattr(torch, model_config["config"]["torch_dtype"]),
                attn_implementation=model_config["config"]["attn_implementation"],
                device_map=model_config["config"]["device_map"],
                trust_remote_code=model_config["config"]["trust_remote_code"]
            )

            # Load processor with pixel settings
            min_pixels = model_config["config"]["min_pixels"] * 28 * 28
            max_pixels = model_config["config"]["max_pixels"] * 28 * 28

            processor = AutoProcessor.from_pretrained(
                model_config["model_id"],
                min_pixels=min_pixels,
                max_pixels=max_pixels
            )

            self.models[model_name] = model
            self.processors[model_name] = processor
            self.loaded_models.append(model_name)

            # Store the process_vision_info function
            self.process_vision_info = process_vision_info

            self.logger.info(f"Successfully loaded {model_name}")

        except ImportError as e:
            raise RuntimeError(f"Missing dependencies for {model_name}. Please install: pip install qwen-vl-utils transformers accelerate")
        except Exception as e:
            raise RuntimeError(f"Failed to load {model_name}: {str(e)}")

    def _load_fallback_model(self, model_config: Dict) -> None:
        """Load a fallback model."""
        model_name = model_config["name"]

        self.logger.info(f"Loading fallback model: {model_name}")

        try:
            if "internvl" in model_name:
                self._load_internvl_model(model_config)
            elif "llava" in model_name:
                self._load_llava_model(model_config)
            else:
                raise ValueError(f"Unsupported fallback model: {model_name}")

        except Exception as e:
            self.logger.error(f"Failed to load fallback model {model_name}: {e}")
            raise

    def _load_internvl_model(self, model_config: Dict) -> None:
        """Load InternVL model."""
        from transformers import AutoModel, AutoTokenizer

        model_name = model_config["name"]

        model = AutoModel.from_pretrained(
            model_config["model_id"],
            torch_dtype=getattr(torch, model_config["config"]["torch_dtype"]),
            low_cpu_mem_usage=model_config["config"]["low_cpu_mem_usage"],
            trust_remote_code=model_config["config"]["trust_remote_code"],
            device_map=model_config["config"]["device_map"]
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_config["model_id"],
            trust_remote_code=True
        )

        self.models[model_name] = model
        self.processors[model_name] = tokenizer
        self.loaded_models.append(model_name)

        self.logger.info(f"Successfully loaded {model_name}")

    def _load_llava_model(self, model_config: Dict) -> None:
        """Load LLaVA model."""
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

        model_name = model_config["name"]

        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_config["model_id"],
            torch_dtype=getattr(torch, model_config["config"]["torch_dtype"]),
            device_map=model_config["config"]["device_map"],
            trust_remote_code=model_config["config"]["trust_remote_code"]
        )

        processor = LlavaNextProcessor.from_pretrained(model_config["model_id"])

        self.models[model_name] = model
        self.processors[model_name] = processor
        self.loaded_models.append(model_name)

        self.logger.info(f"Successfully loaded {model_name}")

    def _get_cache_key(self, image_path: str, model_name: str) -> str:
        """Generate cache key for image and model combination."""
        # Create hash from image path and model name
        content = f"{image_path}_{model_name}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Retrieve cached result if available."""
        if not self.config["processing"]["enable_caching"]:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache {cache_key}: {e}")
        return None

    def _save_cached_result(self, cache_key: str, result: Dict) -> None:
        """Save result to cache."""
        if not self.config["processing"]["enable_caching"]:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save cache {cache_key}: {e}")

    def assess_single_image(self, image_path: str) -> Dict:
        """Assess quality of a single image using available VLM models.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing quality assessment results
        """
        if not self.loaded_models:
            raise RuntimeError("No VLM models are loaded. Call load_models() first.")

        # Try each model in order
        for model_name in self.loaded_models:
            cache_key = self._get_cache_key(image_path, model_name)

            # Check cache first
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.logger.debug(f"Using cached result for {image_path} with {model_name}")
                return cached_result

            # Attempt assessment
            try:
                result = self._assess_with_model(image_path, model_name)
                result["model_used"] = model_name
                result["cache_key"] = cache_key

                # Save to cache
                self._save_cached_result(cache_key, result)

                return result

            except Exception as e:
                self.logger.warning(f"Assessment failed with {model_name} for {image_path}: {e}")
                if model_name == self.loaded_models[-1]:  # Last model
                    self.logger.error(f"All models failed for {image_path}")
                    return self._get_fallback_result(image_path, str(e))
                continue

        return self._get_fallback_result(image_path, "All models failed")

    def _assess_with_model(self, image_path: str, model_name: str) -> Dict:
        """Assess image quality with a specific model."""
        if model_name == "qwen2.5-vl-7b":
            return self._assess_with_qwen(image_path)
        elif model_name == "internvl-1.5":
            return self._assess_with_internvl(image_path)
        elif model_name == "llava-next-13b":
            return self._assess_with_llava(image_path)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _assess_with_qwen(self, image_path: str) -> Dict:
        """Assess image quality with Qwen2.5-VL model."""
        model = self.models["qwen2.5-vl-7b"]
        processor = self.processors["qwen2.5-vl-7b"]

        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": self.config["quality_assessment"]["prompt_template"]}
                ]
            }
        ]

        # Process input
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # Generate response
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.1
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return self._parse_quality_response(response)

    def _assess_with_internvl(self, image_path: str) -> Dict:
        """Assess image quality with InternVL model."""
        model = self.models["internvl-1.5"]
        tokenizer = self.processors["internvl-1.5"]

        # Load and process image
        image = Image.open(image_path).convert('RGB')

        # Prepare input
        prompt = f"<image>\n{self.config['quality_assessment']['prompt_template']}"

        # Generate response
        with torch.inference_mode():
            response = model.chat(
                tokenizer,
                image,
                prompt,
                generation_config={
                    'max_new_tokens': 256,
                    'do_sample': False,
                    'temperature': 0.1
                }
            )

        return self._parse_quality_response(response)

    def _assess_with_llava(self, image_path: str) -> Dict:
        """Assess image quality with LLaVA model."""
        model = self.models["llava-next-13b"]
        processor = self.processors["llava-next-13b"]

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Prepare conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.config["quality_assessment"]["prompt_template"]},
                    {"type": "image"}
                ]
            }
        ]

        # Process inputs
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(prompt, [image], return_tensors="pt")
        inputs = inputs.to(model.device)

        # Generate response
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.1
            )

        response = processor.decode(output[0], skip_special_tokens=True)

        return self._parse_quality_response(response)

    def _parse_quality_response(self, response: str) -> Dict:
        """Parse the VLM response to extract quality scores."""
        try:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)

            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)

                # Validate required fields
                required_fields = self.config["quality_assessment"]["dimensions"] + ["overall_quality"]

                for field in required_fields:
                    if field not in result:
                        result[field] = 0.5  # Default neutral score
                    else:
                        # Ensure score is within valid range
                        score = float(result[field])
                        result[field] = max(0.0, min(1.0, score))

                # Ensure reasoning is present
                if "reasoning" not in result:
                    result["reasoning"] = "No reasoning provided by model"

                return result

        except Exception as e:
            self.logger.warning(f"Failed to parse VLM response: {e}")

        # Fallback parsing or default values
        return self._get_default_quality_scores(f"Failed to parse response: {response[:100]}...")

    def _get_default_quality_scores(self, reason: str = "Assessment failed") -> Dict:
        """Get default quality scores when assessment fails."""
        result = {}
        for dimension in self.config["quality_assessment"]["dimensions"]:
            result[dimension] = 0.5  # Neutral score
        result["overall_quality"] = 0.5
        result["reasoning"] = reason
        return result

    def _get_fallback_result(self, image_path: str, error_message: str) -> Dict:
        """Get fallback result when all models fail."""
        self.logger.error(f"VLM assessment failed for {image_path}: {error_message}")
        result = self._get_default_quality_scores(f"VLM assessment failed: {error_message}")
        result["model_used"] = "fallback"
        result["error"] = error_message
        return result

    def assess_batch(self, image_paths: List[str]) -> List[Dict]:
        """Assess quality of multiple images in batch.

        Args:
            image_paths: List of image file paths

        Returns:
            List of quality assessment results
        """
        if not self.loaded_models:
            raise RuntimeError("No VLM models are loaded. Call load_models() first.")

        results = []
        batch_size = self.config["processing"]["batch_size"]

        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]

            if self.config["logging"]["enable_progress_bar"]:
                batch_iter = tqdm(batch_paths, desc=f"VLM Assessment Batch {i//batch_size + 1}")
            else:
                batch_iter = batch_paths

            for image_path in batch_iter:
                try:
                    result = self.assess_single_image(image_path)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to assess {image_path}: {e}")
                    results.append(self._get_fallback_result(image_path, str(e)))

            # Clear GPU cache between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def cleanup(self):
        """Cleanup models and free GPU memory."""
        self.logger.info("Cleaning up VLM models...")

        for model_name in list(self.models.keys()):
            del self.models[model_name]

        for processor_name in list(self.processors.keys()):
            del self.processors[processor_name]

        self.models.clear()
        self.processors.clear()
        self.loaded_models.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("VLM cleanup complete")


def test_vlm_assessor():
    """Test function for VLM Quality Assessor."""
    try:
        # Initialize assessor
        assessor = VLMQualityAssessor()

        # Load models
        assessor.load_models()

        # Test with a sample image (you'll need to provide a valid image path)
        test_image_path = "path/to/test/image.jpg"  # Replace with actual image

        if os.path.exists(test_image_path):
            result = assessor.assess_single_image(test_image_path)
            print("Assessment result:", json.dumps(result, indent=2))
        else:
            print("Test image not found. Please provide a valid image path for testing.")

        # Cleanup
        assessor.cleanup()

    except Exception as e:
        print(f"Test failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test_vlm_assessor()