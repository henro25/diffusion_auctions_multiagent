"""
FluxPipelineAuction - Multi-Agent Auction Pipeline for FLUX Diffusion Models

This module implements a specialized diffusion pipeline that enables multiple agents
to bid for influence in image generation. The pipeline uses iterative score composition
algorithms to blend agent influences based on their bid amounts.

Core Algorithm:
- Agents submit prompts and bids as lists (dynamic number of agents)
- Agents are sorted by bid amount (ascending for iterative processing)
- Score composition uses iterative pairwise weighted interpolation
- Higher bids result in greater visual influence

Authors: Lillian Sun, Warren Zhu, Henry Huang
Academic Context: 4th Year research project on multi-winner auctions for generative AI
"""

import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Union
from copy import deepcopy

from diffusers import FluxPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps, calculate_shift
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils import is_torch_xla_available

# Compatibility workaround for PyTorch versions < 2.5.0 that don't support enable_gqa
# This allows the pipeline to work with older PyTorch versions while newer versions
# benefit from the GQA optimization.
try:
    # Check if enable_gqa is supported by calling with it
    import inspect
    if "enable_gqa" not in inspect.signature(F.scaled_dot_product_attention).parameters:
        # Monkey-patch to remove enable_gqa for older PyTorch versions
        _orig_sdpa = F.scaled_dot_product_attention

        def _sdpa_drop_enable_gqa(*args, **kwargs):
            kwargs.pop("enable_gqa", None)
            return _orig_sdpa(*args, **kwargs)

        F.scaled_dot_product_attention = _sdpa_drop_enable_gqa
except Exception:
    # If inspection fails, continue without patching
    pass

# XLA availability check
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

# Global flags for score composition
GLOBAL_WORK = False
DO_TRUE_CFG = False


class FluxPipelineAuction(FluxPipeline):
    """
    FLUX diffusion pipeline with multi-agent auction mechanism.

    This pipeline extends the standard FLUX pipeline to support multiple agents
    (dynamic count) who can bid to influence the final generated image. The core
    innovation is the iterative score composition algorithm that progressively
    combines noise predictions based on bid amounts.

    Key Features:
    - Dynamic multi-agent prompt combination
    - Bid-based influence weighting
    - Iterative score composition (handles any number of agents)
    - Base prompt support for scene context

    Usage:
        pipeline = FluxPipelineAuction.from_pretrained("black-forest-labs/FLUX.1-schnell")

        image = pipeline(
            agent_prompts=["Starbucks coffee mug", "Apple MacBook Air", "New York Times newspaper"],
            agent_bids=[0.6, 0.3, 0.1],
            base_prompt="Two friends chatting at a cafe"
        )
    """

    @torch.no_grad()
    def __call__(
        self,
        # Dynamic agent prompts and bids
        agent_prompts: Union[List[str], List[List[str]]] = None,
        agent_bids: List[float] = None,
        base_prompt: Optional[Union[str, List[str]]] = None,
        # Standard diffusion pipeline parameters
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        # IP Adapter support
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        """
        Generate image using multi-agent auction mechanism.

        Args:
            agent_prompts: List of text prompts for agents (dynamic count)
            agent_bids: List of bid amounts for agents (same length as agent_prompts)
            base_prompt: Base scene description
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps
            sigmas: Custom timestep sigmas
            guidance_scale: Guidance scale for generation
            negative_prompt: Negative prompt
            negative_prompt_embeds: Pre-computed negative prompt embeddings
            negative_pooled_prompt_embeds: Pre-computed negative pooled embeddings
            num_images_per_prompt: Number of images per prompt
            generator: Random number generator
            latents: Initial latents
            ip_adapter_image: IP adapter image input
            ip_adapter_image_embeds: IP adapter image embeddings
            output_type: Output type ("pil", "latent", etc.)
            return_dict: Whether to return a dict or tuple
            joint_attention_kwargs: Joint attention kwargs
            callback_on_step_end: Callback function
            callback_on_step_end_tensor_inputs: Tensor names for callback
            max_sequence_length: Maximum sequence length

        Returns:
            FluxPipelineOutput: Generated images
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Input validation
        self._validate_inputs(
            agent_prompts,
            agent_bids,
            base_prompt,
            height,
            width,
            negative_prompt,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = (
            joint_attention_kwargs if joint_attention_kwargs is not None else {}
        )
        self._current_timestep = None
        self._interrupt = False

        # 2. Determine batch size from agent prompts
        batch_size = self._determine_batch_size(agent_prompts)

        device = self._execution_device
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None)
            if self.joint_attention_kwargs is not None
            else None
        )

        # 3. Organize and sort agent data by bid amount (ascending for iterative processing)
        agents_data_for_composition = self._organize_and_sort_agents(
            agent_prompts, agent_bids, base_prompt
        )

        # Set global flags for score composition
        global GLOBAL_WORK, DO_TRUE_CFG
        GLOBAL_WORK = True
        DO_TRUE_CFG = True

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        # Get dtype from the transformer's first parameter
        transformer_dtype = next(self.transformer.parameters()).dtype
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            transformer_dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare timesteps
        image_seq_len = latents.shape[2] * self.vae_scale_factor
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 7. Guidance preparation
        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 8. IP Adapter setup
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            processed_ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
            self._joint_attention_kwargs["ip_adapter_image_embeds"] = (
                processed_ip_adapter_image_embeds
            )

        # 9. Main denoising loop with iterative score composition
        generation_results = self._denoising_loop_with_score_composition(
            latents,
            latent_image_ids,
            timesteps,
            num_inference_steps,
            num_warmup_steps,
            agents_data_for_composition,
            device,
            guidance,
            callback_on_step_end,
            callback_on_step_end_tensor_inputs,
        )

        # 10. Post-processing
        self._current_timestep = None
        if output_type == "latent":
            image = generation_results
        else:
            latents = self._unpack_latents(
                generation_results, height, width, self.vae_scale_factor
            )
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return FluxPipelineOutput(images=image)

    def _validate_inputs(
        self,
        agent_prompts,
        agent_bids,
        base_prompt,
        height,
        width,
        negative_prompt,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs,
        max_sequence_length,
    ):
        """Validate input parameters for dynamic agent count."""
        # Check agent prompts and bids
        if agent_prompts is None:
            raise ValueError("agent_prompts must be provided")
        if agent_bids is None:
            raise ValueError("agent_bids must be provided")
        if not isinstance(agent_prompts, (list, tuple)):
            raise ValueError(
                f"agent_prompts must be a list or tuple, got {type(agent_prompts)}"
            )
        if not isinstance(agent_bids, (list, tuple)):
            raise ValueError(
                f"agent_bids must be a list or tuple, got {type(agent_bids)}"
            )
        if len(agent_prompts) != len(agent_bids):
            raise ValueError(
                f"agent_prompts and agent_bids must have the same length. "
                f"Got {len(agent_prompts)} and {len(agent_bids)}"
            )
        if len(agent_prompts) == 0:
            raise ValueError("At least one agent prompt must be provided")

        # Use first agent prompt for standard validation
        first_agent_prompt = agent_prompts[0]

        # Call parent class validation with first agent's prompt
        self.check_inputs(
            first_agent_prompt,
            None,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=None,
            prompt_embeds=None,  # We'll handle embeddings ourselves
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

    def _determine_batch_size(self, agent_prompts):
        """Determine batch size from agent prompts."""
        if agent_prompts is None or len(agent_prompts) == 0:
            raise ValueError("No agent prompts provided")

        first_prompt = agent_prompts[0]
        if isinstance(first_prompt, str):
            return 1
        elif isinstance(first_prompt, (list, tuple)):
            return len(first_prompt)
        else:
            raise ValueError(
                f"Agent prompt must be string or list of strings, got {type(first_prompt)}"
            )

    def _organize_and_sort_agents(self, agent_prompts, agent_bids, base_prompt):
        """
        Organize agent data and sort by bid amount (ascending order).

        Ascending order is used for iterative processing where we progressively
        combine agents from lowest to highest bid. Returns minimal data needed for
        iterative composition (no pre-computed embeddings).

        Args:
            agent_prompts: List of agent prompts
            agent_bids: List of agent bids
            base_prompt: Base prompt to be applied consistently

        Returns:
            List of agent data dicts sorted by bid (ascending), with:
            - prompt: Agent's text prompt
            - bid: Agent's bid amount
            - base_prompt: Base prompt for composition
        """
        if not agent_prompts or not agent_bids:
            raise ValueError(
                "Both agent_prompts and agent_bids must be provided and non-empty"
            )

        # Create agent data list
        agents_data = []
        for prompt, bid in zip(agent_prompts, agent_bids):
            agent_data = {
                "prompt": prompt,
                "bid": float(bid),
                "base_prompt": base_prompt,  # Keep base_prompt for composition
            }
            agents_data.append(agent_data)

        # Sort by bid amount (ascending)
        sorted_agents = sorted(agents_data, key=lambda x: x["bid"])

        return sorted_agents

    def _denoising_loop_with_score_composition(
        self,
        latents,
        latent_image_ids,
        timesteps,
        num_inference_steps,
        num_warmup_steps,
        agents_data_for_composition,
        device,
        guidance,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
    ):
        """
        Main denoising loop with iterative multi-agent score composition.

        For each timestep, applies the iterative score composition algorithm:
        1. Start with lists of agents sorted by bid (ascending)
        2. Pop the top two agents (highest bids)
        3. Compute noise predictions for each agent and their combined prompt
        4. Apply weighted composition based on normalized bids
        5. Combine agents into a single composite agent
        6. Repeat until only one agent remains
        """

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                timestep_expanded = t.expand(latents.shape[0]).to(latents.dtype)

                # Apply iterative score composition for this timestep
                noise_pred = self._apply_iterative_score_composition(
                    latents,
                    timestep_expanded,
                    guidance,
                    latent_image_ids,
                    agents_data_for_composition,
                    device,
                )

                # Scheduler step
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                # Callback handling
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k_cb in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k_cb] = locals()[k_cb]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                # Progress update
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    pass  # xm.mark_step() would go here if needed

        return latents

    def _apply_iterative_score_composition(
        self,
        latents,
        timestep_expanded,
        guidance,
        latent_image_ids,
        agents_data_for_composition,
        device,
    ):
        """
        Apply iterative score composition algorithm for dynamic number of agents.

        Embeddings are computed on-demand during iteration:
        - First iteration: Encode agent_n with base_prompt
        - All iterations: Encode combined agents with base_prompt
        - Special handling for base-prompt-only case (all bids = 0)

        Algorithm:
        1. Check if all bids are zero (base-prompt-only case)
           - If yes, return noise prediction for base prompt only
        2. Sort agents by bid (ascending) - already done in _organize_and_sort_agents
        3. Loop until only one agent remains:
           a. Pop the two agents with the highest bids
           b. Normalize their bids relative to each other
           c. Get noise predictions for each agent and their combined prompt
           d. Apply weighted composition: w_dom = 2*norm_bid_highest - 1
           e. Combined_noise = (1-w_dom)*noise_shared + w_dom*noise_dom
           f. Create new composite agent with combined prompt and summed bid
           g. Add composite agent back to the list
        4. Return final noise prediction

        Note: base_prompt is kept separate and applied consistently.
        Agent prompts are combined iteratively without base_prompt.
        Edge case: All zero bids -> return base_prompt-only prediction
        """
        # Make a copy of agent data to work with
        remaining_agents = deepcopy(agents_data_for_composition)

        # Get base_prompt from first agent (same for all)
        base_prompt = remaining_agents[0]["base_prompt"] if remaining_agents else ""

        if base_prompt is None:
            base_prompt = ""

        # Check if all agents have zero or near-zero bids (base-prompt-only case)
        total_bid = sum(agent["bid"] for agent in remaining_agents)
        if total_bid < 1e-9:
            # All bids are essentially zero: return base prompt only
            embeds, pooled_embeds, text_ids = self.encode_prompt(
                prompt=base_prompt,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=512,
                lora_scale=None,
            )
            return self._get_noise_prediction(
                latents,
                timestep_expanded,
                guidance,
                latent_image_ids,
                embeds,
                pooled_embeds,
                text_ids,
            )

        if len(remaining_agents) <= 1:
            raise ValueError("At least one agent with non-zero bid is required for composition")

        # Running noise prediction (will be updated iteratively)
        combined_noise_pred = None

        # Iteratively combine agents from lowest to highest bid
        while len(remaining_agents) > 1:
            # Pop the two highest bidders
            agent_n = remaining_agents.pop()  # Highest bid
            agent_n_minus_1 = remaining_agents.pop()  # Second highest bid

            # Get the current bids (already sorted ascending)
            bid_n = agent_n["bid"]
            bid_n_minus_1 = agent_n_minus_1["bid"]

            # Normalize bids (relative to their sum)
            bid_sum = bid_n + bid_n_minus_1
            if bid_sum < 1e-9:
                # Both bids are essentially zero, skip composition
                remaining_agents.append(agent_n)
                continue

            norm_bid_n = bid_n / bid_sum

            # Get noise prediction for agent_n (the dominant agent in this pair)
            # Encode agent_n with base_prompt on-demand
            agent_n_prompt = agent_n["prompt"]
            if base_prompt:
                combined_prompt_agent_n = f"{base_prompt} with {agent_n_prompt}"
            else:
                combined_prompt_agent_n = agent_n_prompt

            embeds_n, pooled_embeds_n, text_ids_n = self.encode_prompt(
                prompt=combined_prompt_agent_n,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=512,
                lora_scale=None,
            )

            noise_pred_dom = self._get_noise_prediction(
                latents,
                timestep_expanded,
                guidance,
                latent_image_ids,
                embeds_n,
                pooled_embeds_n,
                text_ids_n,
            )

            # Get noise prediction for shared prompt (both agents combined with base)
            # Create combined prompt for this iteration
            combined_agents_prompt = f"{agent_n['prompt']} and {agent_n_minus_1['prompt']}"
            if base_prompt:
                combined_prompt_with_base = f"{base_prompt} with {combined_agents_prompt}"
            else:
                combined_prompt_with_base = combined_agents_prompt

            # Encode the combined prompt for this iteration
            embeds_shared, pooled_embeds_shared, text_ids_shared = self.encode_prompt(
                prompt=combined_prompt_with_base,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=512,
                lora_scale=None,
            )

            noise_pred_shared = self._get_noise_prediction(
                latents,
                timestep_expanded,
                guidance,
                latent_image_ids,
                embeds_shared,
                pooled_embeds_shared,
                text_ids_shared,
            )

            # Calculate weight for dominant agent: w_dom = 2*norm_bid_n - 1
            w_dom = 2 * norm_bid_n - 1
            w_dom = max(0.0, min(1.0, w_dom))  # Clamp to [0, 1]

            w_shared = 1.0 - w_dom

            # Apply weighted composition
            if combined_noise_pred is None:
                # First iteration: combine the two noise predictions
                combined_noise_pred = (
                    w_shared * noise_pred_shared + w_dom * noise_pred_dom
                )
            else:
                # Subsequent iterations: use running combined_noise_pred as noise_pred_dom
                combined_noise_pred = (
                    w_shared * noise_pred_shared + w_dom * combined_noise_pred
                )

            # Create composite agent
            # Store only agent prompts (without base_prompt)
            composite_agent = {
                "prompt": combined_agents_prompt,  # Only agent prompts, no base
                "bid": bid_n + bid_n_minus_1,  # Sum of bids
                "base_prompt": base_prompt,  # Keep base_prompt for next iteration
                # Embeddings will be computed on-demand in next iteration
            }

            # Add composite agent back to remaining agents (will be sorted by bid)
            remaining_agents.append(composite_agent)
            # Re-sort by bid to maintain ascending order
            remaining_agents.sort(key=lambda x: x["bid"])

        return combined_noise_pred

    def _get_noise_prediction(
        self,
        latents,
        timestep_expanded,
        guidance,
        latent_image_ids,
        embeds,
        pooled_embeds,
        text_ids,
    ):
        """Get noise prediction from transformer for given embeddings."""
        return self.transformer(
            hidden_states=latents,
            timestep=timestep_expanded / 1000,
            guidance=guidance,
            pooled_projections=pooled_embeds,
            encoder_hidden_states=embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]
