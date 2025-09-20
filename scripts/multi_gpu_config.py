"""
Multi-GPU Configuration for Diffusion Auctions
Handles distribution of image generation tasks across multiple GPUs
"""

import torch
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from typing import List, Tuple, Dict, Any
from tqdm import tqdm


class GenerationTask:
    """Represents a single image generation task"""

    def __init__(
        self,
        prompt_data: Dict,
        bid_combination: Tuple[float, float, float],
        prompt_index: int,
        sample_index: int,
        output_dir: str,
    ):
        self.prompt_data = prompt_data
        self.bid_combination = bid_combination
        self.prompt_index = prompt_index
        self.sample_index = sample_index
        self.output_dir = output_dir


class MultiGPUManager:
    """Manages multi-GPU image generation for diffusion auctions"""

    def __init__(
        self,
        num_gpus: int = None,
        gpu_indices: List[int] = None,
        torch_dtype=torch.bfloat16,
    ):
        self.torch_dtype = torch_dtype
        self.pipelines = {}

        # Determine which GPUs to use
        total_gpus = torch.cuda.device_count()
        if total_gpus == 0:
            raise RuntimeError("No CUDA GPUs available")

        if gpu_indices is not None:
            # Validate specified GPU indices
            invalid_indices = [
                idx for idx in gpu_indices if idx >= total_gpus or idx < 0
            ]
            if invalid_indices:
                raise ValueError(
                    f"Invalid GPU indices {invalid_indices}. Available GPUs: 0-{total_gpus - 1}"
                )
            self.gpu_indices = gpu_indices
            self.num_gpus = len(gpu_indices)
        elif num_gpus is not None:
            # Use first N GPUs
            self.num_gpus = min(num_gpus, total_gpus)
            self.gpu_indices = list(range(self.num_gpus))
        else:
            # Use all available GPUs
            self.num_gpus = total_gpus
            self.gpu_indices = list(range(total_gpus))

        print(
            f"Initializing MultiGPUManager with {self.num_gpus} GPUs: {self.gpu_indices}"
        )

    def initialize_pipelines(
        self, model_name: str = "black-forest-labs/FLUX.1-schnell"
    ):
        """Initialize pipeline on each GPU"""
        import sys
        import os

        sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
        from pipelines import FluxPipelineAuction

        for gpu_id in self.gpu_indices:
            device = torch.device(f"cuda:{gpu_id}")
            print(f"Loading pipeline on GPU {gpu_id}...")

            pipeline = FluxPipelineAuction.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
            ).to(device)

            self.pipelines[gpu_id] = pipeline

            # Clear cache after loading
            torch.cuda.empty_cache()

    def generate_single_image(
        self, task: GenerationTask, gpu_id: int
    ) -> Dict[str, Any]:
        """Generate a single image on specified GPU"""
        device = torch.device(f"cuda:{gpu_id}")
        pipeline = self.pipelines[gpu_id]

        # Set device for this process
        torch.cuda.set_device(device)

        agent1_prompt = task.prompt_data.get("agent1_prompt", "")
        agent2_prompt = task.prompt_data.get("agent2_prompt", "")
        agent3_prompt = task.prompt_data.get("agent3_prompt", "")
        base_prompt = task.prompt_data.get("base_prompt", "")

        bid1, bid2, bid3 = task.bid_combination

        filename_base = f"idx{task.prompt_index:03d}_b1_{bid1:.2f}_b2_{bid2:.2f}_b3_{bid3:.2f}_s{task.sample_index:02d}"

        prompt_specific_output_dir = os.path.join(
            task.output_dir, f"prompt_{task.prompt_index:03d}"
        )
        os.makedirs(prompt_specific_output_dir, exist_ok=True)
        output_path = os.path.join(prompt_specific_output_dir, f"{filename_base}.png")

        try:
            # Import generation parameters from main script
            from generate_images_3_agent import GUIDANCE_SCALE, NUM_INFERENCE_STEPS

            with torch.cuda.device(device):
                images = pipeline(
                    agent1_prompt=agent1_prompt,
                    agent1_bid=bid1,
                    agent2_prompt=agent2_prompt,
                    agent2_bid=bid2,
                    agent3_prompt=agent3_prompt,
                    agent3_bid=bid3,
                    base_prompt=base_prompt,
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                )

                if images and hasattr(images, "images") and len(images.images) > 0:
                    images.images[0].save(output_path)

                    return {
                        "success": True,
                        "item_index": task.prompt_index,
                        "bids": task.bid_combination,
                        "sample_index": task.sample_index,
                        "agent1_prompt": agent1_prompt,
                        "agent2_prompt": agent2_prompt,
                        "agent3_prompt": agent3_prompt,
                        "base_prompt": base_prompt,
                        "image_path": output_path,
                        "gpu_id": gpu_id,
                    }
                else:
                    return {
                        "success": False,
                        "error": "No image generated",
                        "gpu_id": gpu_id,
                    }

        except Exception as e:
            return {"success": False, "error": str(e), "gpu_id": gpu_id}
        finally:
            # Clear GPU cache after generation
            torch.cuda.empty_cache()

    def generate_images_parallel(
        self,
        prompts: List[Dict],
        bidding_combinations: List[Tuple],
        num_samples_per_combination: int,
        output_dir: str,
        num_prompts_to_process: int = None,
    ) -> List[Dict]:
        """Generate images in parallel across multiple GPUs"""

        # Create all tasks
        tasks = []
        num_items_to_process = num_prompts_to_process or len(prompts)

        for i, prompt_data in enumerate(prompts[:num_items_to_process]):
            for bid_combination in bidding_combinations:
                for sample_idx in range(num_samples_per_combination):
                    task = GenerationTask(
                        prompt_data=prompt_data,
                        bid_combination=bid_combination,
                        prompt_index=i,
                        sample_index=sample_idx,
                        output_dir=output_dir,
                    )
                    tasks.append(task)

        print(f"Created {len(tasks)} generation tasks")
        print(f"Using {self.num_gpus} GPUs for parallel processing")

        # Process tasks in parallel using ThreadPoolExecutor
        results = []
        successful_results = []

        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            # Submit tasks to GPUs in round-robin fashion
            future_to_task = {}

            for i, task in enumerate(tasks):
                gpu_id = self.gpu_indices[i % self.num_gpus]
                future = executor.submit(self.generate_single_image, task, gpu_id)
                future_to_task[future] = (task, gpu_id)

            # Collect results with progress bar
            with tqdm(total=len(tasks), desc="Generating images") as pbar:
                for future in as_completed(future_to_task):
                    task, gpu_id = future_to_task[future]
                    try:
                        result = future.result()
                        results.append(result)

                        if result["success"]:
                            successful_results.append(result)
                            pbar.set_postfix(
                                {
                                    "GPU": gpu_id,
                                    "Success": len(successful_results),
                                    "Failed": len(results) - len(successful_results),
                                }
                            )
                        else:
                            pbar.set_postfix(
                                {
                                    "GPU": gpu_id,
                                    "Success": len(successful_results),
                                    "Failed": len(results) - len(successful_results),
                                    "Last Error": result.get("error", "Unknown")[:30],
                                }
                            )

                    except Exception as exc:
                        print(f"Task generated an exception: {exc}")
                        results.append(
                            {"success": False, "error": str(exc), "gpu_id": gpu_id}
                        )

                    pbar.update(1)

        print(
            f"\nCompleted: {len(successful_results)} successful, {len(results) - len(successful_results)} failed"
        )
        return successful_results


def setup_multi_gpu_environment():
    """Setup environment variables for multi-GPU usage"""
    # Set multiprocessing start method
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    # Set CUDA environment variables for better multi-GPU performance
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    print("Multi-GPU environment configured")


def get_gpu_memory_info():
    """Get memory information for all available GPUs"""
    if not torch.cuda.is_available():
        return "No CUDA GPUs available"

    info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_total = props.total_memory / (1024**3)  # GB
        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
        memory_free = memory_total - memory_allocated

        info.append(
            {
                "gpu_id": i,
                "name": props.name,
                "total_memory_gb": round(memory_total, 2),
                "allocated_memory_gb": round(memory_allocated, 2),
                "free_memory_gb": round(memory_free, 2),
            }
        )

    return info
