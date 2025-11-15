"""
FluxPipelineAuction - Multi-Agent Auction Pipeline for FLUX Diffusion Models

This module implements a specialized diffusion pipeline that enables multiple agents
to bid for influence in image generation. The pipeline uses score composition
algorithms to blend agent influences based on their bid amounts.

Core Algorithm:
- Agents submit prompts and bids (0.0-1.0)
- Agents are sorted by bid amount (descending)
- Score composition uses recursive weighted interpolation
- Higher bids result in greater visual influence

Authors: Lillian Sun, Warren Zhu, Henry Huang
Academic Context: 4th Year research project on multi-winner auctions for generative AI
"""

import torch
from typing import Any, Callable, Dict, List, Optional, Union

from diffusers import FluxPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps, calculate_shift
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils import is_torch_xla_available

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
    (up to 3) who can bid to influence the final generated image. The core
    innovation is the score composition algorithm that recursively combines
    noise predictions based on bid amounts.

    Key Features:
    - Multi-agent prompt combination
    - Bid-based influence weighting
    - Recursive score composition
    - Base prompt support for scene context

    Usage:
        pipeline = FluxPipelineAuction.from_pretrained("black-forest-labs/FLUX.1-schnell")

        image = pipeline(
            agent1_prompt="Starbucks coffee mug",
            agent1_bid=0.6,
            agent2_prompt="Apple MacBook Air",
            agent2_bid=0.3,
            agent3_prompt="New York Times newspaper",
            agent3_bid=0.1,
            base_prompt="Two friends chatting at a cafe"
        )
    """

    @torch.no_grad()
    def __call__(
        self,
        # Agent prompts and bids
        agent1_prompt: Union[str, List[str]] = None,
        agent1_prompt_2: Optional[Union[str, List[str]]] = None,
        agent1_bid: float = 0.0,
        agent1_prompt_embeds: Optional[torch.FloatTensor] = None,
        agent1_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        agent2_prompt: Union[str, List[str]] = None,
        agent2_prompt_2: Optional[Union[str, List[str]]] = None,
        agent2_bid: float = 0.0,
        agent2_prompt_embeds: Optional[torch.FloatTensor] = None,
        agent2_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        agent3_prompt: Union[str, List[str]] = None,
        agent3_prompt_2: Optional[Union[str, List[str]]] = None,
        agent3_bid: float = 0.0,
        agent3_prompt_embeds: Optional[torch.FloatTensor] = None,
        agent3_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        base_prompt: Optional[Union[str, List[str]]] = None,  # Optional base prompt
        base_prompt_2: Optional[Union[str, List[str]]] = None,
        # Standard diffusion pipeline parameters
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
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
            agent1_prompt: Text prompt for agent 1
            agent1_bid: Bid amount for agent 1 (0.0-1.0)
            agent2_prompt: Text prompt for agent 2
            agent2_bid: Bid amount for agent 2 (0.0-1.0)
            agent3_prompt: Text prompt for agent 3
            agent3_bid: Bid amount for agent 3 (0.0-1.0)
            base_prompt: Base scene description
            ... (other standard diffusion parameters)

        Returns:
            FluxPipelineOutput: Generated images
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Input validation
        self.check_inputs(
            agent1_prompt,
            agent1_prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=agent1_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=agent1_pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = (
            joint_attention_kwargs if joint_attention_kwargs is not None else {}
        )
        self._current_timestep = None
        self._interrupt = False

        # 2. Determine batch size from available prompts
        batch_size = self._determine_batch_size(
            agent1_prompt,
            agent1_prompt_embeds,
            agent2_prompt,
            agent2_prompt_embeds,
            agent3_prompt,
            agent3_prompt_embeds,
        )

        device = self._execution_device
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None)
            if self.joint_attention_kwargs is not None
            else None
        )

        # 3. Organize and sort agent data by bid amount
        sorted_agents_data = self._organize_and_sort_agents(
            agent1_prompt,
            agent1_prompt_2,
            agent1_bid,
            agent1_prompt_embeds,
            agent1_pooled_prompt_embeds,
            agent2_prompt,
            agent2_prompt_2,
            agent2_bid,
            agent2_prompt_embeds,
            agent2_pooled_prompt_embeds,
            agent3_prompt,
            agent3_prompt_2,
            agent3_bid,
            agent3_prompt_embeds,
            agent3_pooled_prompt_embeds,
        )

        # 4. Generate prompt embeddings for score composition
        (
            s1_embeds,
            s1_pooled_embeds,
            s1_text_ids,
            s1_s2_embeds,
            s1_s2_pooled_embeds,
            s1_s2_text_ids,
            s1_s2_s3_embeds,
            s1_s2_s3_pooled_embeds,
            s1_s2_s3_text_ids,
        ) = self._generate_prompt_embeddings(
            sorted_agents_data,
            base_prompt,
            base_prompt_2,
            device,
            num_images_per_prompt,
            max_sequence_length,
            lora_scale,
        )

        # Set global flags for score composition
        global GLOBAL_WORK, DO_TRUE_CFG
        GLOBAL_WORK = True
        DO_TRUE_CFG = True

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            s1_s2_s3_embeds.dtype,
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

        # 9. Main denoising loop with score composition
        generation_results = self._denoising_loop_with_score_composition(
            latents,
            latent_image_ids,
            timesteps,
            num_inference_steps,
            num_warmup_steps,
            sorted_agents_data,
            guidance,
            s1_embeds,
            s1_pooled_embeds,
            s1_text_ids,
            s1_s2_embeds,
            s1_s2_pooled_embeds,
            s1_s2_text_ids,
            s1_s2_s3_embeds,
            s1_s2_s3_pooled_embeds,
            s1_s2_s3_text_ids,
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

    def _determine_batch_size(
        self,
        agent1_prompt,
        agent1_prompt_embeds,
        agent2_prompt,
        agent2_prompt_embeds,
        agent3_prompt,
        agent3_prompt_embeds,
    ):
        """Determine batch size from available agent prompts."""
        for prompt, embeds in [
            (agent1_prompt, agent1_prompt_embeds),
            (agent2_prompt, agent2_prompt_embeds),
            (agent3_prompt, agent3_prompt_embeds),
        ]:
            if prompt is not None:
                return 1 if isinstance(prompt, str) else len(prompt)
            # Use explicit check to avoid tensor boolean ambiguity
            elif embeds is not None:
                return embeds.shape[0]
        raise ValueError(
            "At least one agent prompt or its embeddings must be provided to determine batch_size."
        )

    def _organize_and_sort_agents(self, *agent_params):
        """Organize agent data and sort by bid amount (descending)."""
        # Group parameters by agent (3 agents × 5 params each = 15 params)
        agents_initial_data = []
        for i in range(3):
            base_idx = i * 5
            agent_data = {
                "id": f"agent{i + 1}",
                "prompt": agent_params[base_idx],
                "prompt_2": agent_params[base_idx + 1],
                "bid": agent_params[base_idx + 2],
                "prompt_embeds": agent_params[base_idx + 3],
                "pooled_prompt_embeds": agent_params[base_idx + 4],
            }
            agents_initial_data.append(agent_data)

        # Filter active agents (those with prompts or embeddings)
        # Use safer boolean checks to avoid tensor boolean ambiguity
        agents_active_data = []
        for agent in agents_initial_data:
            has_prompt = agent["prompt"] is not None
            has_embeds = (
                agent.get("prompt_embeds") is not None
                and agent.get("pooled_prompt_embeds") is not None
            )
            if has_prompt or has_embeds:
                agents_active_data.append(agent)

        if not agents_active_data:
            raise ValueError("No active agents with prompts or embeddings provided.")

        # Sort by bid amount (descending order)
        return sorted(agents_active_data, key=lambda x: x["bid"], reverse=True)

    def _generate_prompt_embeddings(
        self,
        sorted_agents_data,
        base_prompt,
        base_prompt_2,
        device,
        num_images_per_prompt,
        max_sequence_length,
        lora_scale,
    ):
        """Generate embeddings for all agent combinations needed for score composition."""

        s_agent1_data = sorted_agents_data[0] if len(sorted_agents_data) > 0 else None
        s_agent2_data = sorted_agents_data[1] if len(sorted_agents_data) > 1 else None
        s_agent3_data = sorted_agents_data[2] if len(sorted_agents_data) > 2 else None

        def _get_effective_prompt(agent_data, base_p, base_p2):
            """Combine agent prompt with base prompt."""
            if not agent_data:
                return None, None

            main_p = agent_data["prompt"]
            sec_p = agent_data["prompt_2"]

            effective_p = ""
            if isinstance(base_p, str) and base_p:
                effective_p += base_p
            if isinstance(main_p, str) and main_p:
                effective_p = f"{effective_p} and {main_p}" if effective_p else main_p

            effective_p2 = ""
            if isinstance(base_p2, str) and base_p2:
                effective_p2 += base_p2
            if isinstance(sec_p, str) and sec_p:
                effective_p2 = f"{effective_p2} and {sec_p}" if effective_p2 else sec_p

            return (
                effective_p if effective_p else None,
                effective_p2 if effective_p2 else None,
            )

        # Generate embeddings for s1 (highest bidder only)
        s1_full_prompt, s1_full_prompt_2 = _get_effective_prompt(
            s_agent1_data, base_prompt, base_prompt_2
        )

        # Check if we should generate embeddings for s1 (avoid tensor boolean issues)
        should_generate_s1 = s_agent1_data is not None and (
            s1_full_prompt or (s_agent1_data.get("prompt_embeds") is not None)
        )

        (s1_embeds, s1_pooled_embeds, s1_text_ids) = (
            self.encode_prompt(
                prompt=s1_full_prompt,
                prompt_2=s1_full_prompt_2,
                prompt_embeds=s_agent1_data["prompt_embeds"] if s_agent1_data else None,
                pooled_prompt_embeds=s_agent1_data["pooled_prompt_embeds"]
                if s_agent1_data
                else None,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
            if should_generate_s1
            else (None, None, None)
        )

        # Generate embeddings for s1+s2 (top 2 bidders combined)
        s1_s2_combined_prompt = self._combine_agent_prompts(
            [s_agent1_data, s_agent2_data], base_prompt, ", "
        )
        s1_s2_combined_prompt_2 = self._combine_agent_prompts(
            [s_agent1_data, s_agent2_data], base_prompt_2, ", ", prompt_type="prompt_2"
        )

        # Check if we should generate embeddings for s1_s2 (avoid tensor boolean issues)
        should_generate_s1_s2 = s1_s2_combined_prompt is not None

        (s1_s2_embeds, s1_s2_pooled_embeds, s1_s2_text_ids) = (
            self.encode_prompt(
                prompt=s1_s2_combined_prompt,
                prompt_2=s1_s2_combined_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
            if should_generate_s1_s2
            else (s1_embeds, s1_pooled_embeds, s1_text_ids)
        )

        # Generate embeddings for s1+s2+s3 (all agents combined)
        s1_s2_s3_combined_prompt = self._combine_agent_prompts(
            [s_agent1_data, s_agent2_data, s_agent3_data], base_prompt, " and "
        )
        s1_s2_s3_combined_prompt_2 = self._combine_agent_prompts(
            [s_agent1_data, s_agent2_data, s_agent3_data],
            base_prompt_2,
            " and ",
            prompt_type="prompt_2",
        )

        # Check if we should generate embeddings for s1_s2_s3 (avoid tensor boolean issues)
        should_generate_s1_s2_s3 = s1_s2_s3_combined_prompt is not None

        (s1_s2_s3_embeds, s1_s2_s3_pooled_embeds, s1_s2_s3_text_ids) = (
            self.encode_prompt(
                prompt=s1_s2_s3_combined_prompt,
                prompt_2=s1_s2_s3_combined_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
            if should_generate_s1_s2_s3
            else (s1_s2_embeds, s1_s2_pooled_embeds, s1_s2_text_ids)
        )

        # Ensure at least one valid embedding set exists (avoid tensor boolean issues)
        if s1_s2_s3_embeds is None:
            # Use explicit checks to avoid tensor boolean ambiguity
            s1_s2_available = s1_s2_embeds is not None
            s1_available = s1_embeds is not None

            if s1_s2_available:
                s1_s2_s3_embeds, s1_s2_s3_pooled_embeds, s1_s2_s3_text_ids = (
                    s1_s2_embeds,
                    s1_s2_pooled_embeds,
                    s1_s2_text_ids,
                )
            elif s1_available:
                s1_s2_s3_embeds, s1_s2_s3_pooled_embeds, s1_s2_s3_text_ids = (
                    s1_embeds,
                    s1_pooled_embeds,
                    s1_text_ids,
                )
            else:
                raise ValueError(
                    "Failed to generate any valid prompt embeddings for the agents."
                )

        return (
            s1_embeds,
            s1_pooled_embeds,
            s1_text_ids,
            s1_s2_embeds,
            s1_s2_pooled_embeds,
            s1_s2_text_ids,
            s1_s2_s3_embeds,
            s1_s2_s3_pooled_embeds,
            s1_s2_s3_text_ids,
        )

    def _combine_agent_prompts(
        self, agent_data_list, base_prompt, separator, prompt_type="prompt"
    ):
        """Combine multiple agent prompts with a base prompt."""
        prompt_parts = []

        if base_prompt:
            prompt_parts.append(base_prompt)

        for agent_data in agent_data_list:
            if agent_data and agent_data[prompt_type]:
                prompt_parts.append(agent_data[prompt_type])

        return separator.join(p for p in prompt_parts if p) if prompt_parts else None

    def _denoising_loop_with_score_composition(
        self,
        latents,
        latent_image_ids,
        timesteps,
        num_inference_steps,
        num_warmup_steps,
        sorted_agents_data,
        guidance,
        s1_embeds,
        s1_pooled_embeds,
        s1_text_ids,
        s1_s2_embeds,
        s1_s2_pooled_embeds,
        s1_s2_text_ids,
        s1_s2_s3_embeds,
        s1_s2_s3_pooled_embeds,
        s1_s2_s3_text_ids,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
    ):
        """Main denoising loop with multi-agent score composition."""

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                timestep_expanded = t.expand(latents.shape[0]).to(latents.dtype)

                # Get noise predictions for each agent combination
                # Use explicit fallback logic to avoid tensor boolean ambiguity
                s1_embed_to_use = (
                    s1_embeds if s1_embeds is not None else s1_s2_s3_embeds
                )
                s1_pooled_to_use = (
                    s1_pooled_embeds
                    if s1_pooled_embeds is not None
                    else s1_s2_s3_pooled_embeds
                )
                s1_text_ids_to_use = (
                    s1_text_ids if s1_text_ids is not None else s1_s2_s3_text_ids
                )

                noise_pred_s1 = self._get_noise_prediction(
                    latents,
                    timestep_expanded,
                    guidance,
                    latent_image_ids,
                    s1_embed_to_use,
                    s1_pooled_to_use,
                    s1_text_ids_to_use,
                )

                s1_s2_embed_to_use = (
                    s1_s2_embeds if s1_s2_embeds is not None else s1_s2_s3_embeds
                )
                s1_s2_pooled_to_use = (
                    s1_s2_pooled_embeds
                    if s1_s2_pooled_embeds is not None
                    else s1_s2_s3_pooled_embeds
                )
                s1_s2_text_ids_to_use = (
                    s1_s2_text_ids if s1_s2_text_ids is not None else s1_s2_s3_text_ids
                )

                noise_pred_s1_s2 = self._get_noise_prediction(
                    latents,
                    timestep_expanded,
                    guidance,
                    latent_image_ids,
                    s1_s2_embed_to_use,
                    s1_s2_pooled_to_use,
                    s1_s2_text_ids_to_use,
                )

                noise_pred_s1_s2_s3 = self._get_noise_prediction(
                    latents,
                    timestep_expanded,
                    guidance,
                    latent_image_ids,
                    s1_s2_s3_embeds,
                    s1_s2_s3_pooled_embeds,
                    s1_s2_s3_text_ids,
                )

                # Apply score composition algorithm
                noise_pred = self._apply_score_composition(
                    sorted_agents_data,
                    noise_pred_s1,
                    noise_pred_s1_s2,
                    noise_pred_s1_s2_s3,
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

    def _apply_score_composition(
        self, sorted_agents_data, noise_pred_s1, noise_pred_s1_s2, noise_pred_s1_s2_s3
    ):
        """
        Apply the core score composition algorithm for multi-agent auction.

        This implements the recursive weighted interpolation based on bid ratios:
        1. Sort agents by bid amount: sb1 >= sb2 >= sb3
        2. Calculate dominance weights using bid ratios
        3. Apply recursive composition to combine noise predictions

        Algorithm:
        - P_dom_A = (sb1 + sb2) / S3 where S3 = sb1 + sb2 + sb3
        - w_A = 2 * P_dom_A - 1
        - P_dom_B = sb1 / S2 where S2 = sb1 + sb2
        - w_B = 2 * P_dom_B - 1
        - s_1_2_intermediate = (1-w_B) * noise_pred_s1_s2 + w_B * noise_pred_s1
        - final_noise_pred = (1-w_A) * noise_pred_s1_s2_s3 + w_A * s_1_2_intermediate
        """

        # Extract sorted bid amounts
        sb1 = sorted_agents_data[0]["bid"] if len(sorted_agents_data) > 0 else 0.0
        sb2 = sorted_agents_data[1]["bid"] if len(sorted_agents_data) > 1 else 0.0
        sb3 = sorted_agents_data[2]["bid"] if len(sorted_agents_data) > 2 else 0.0

        # Ensure non-negative bids
        sb1, sb2, sb3 = max(0, sb1), max(0, sb2), max(0, sb3)
        S3 = sb1 + sb2 + sb3

        # Handle edge case: all bids are zero
        if S3 < 1e-9:
            return noise_pred_s1_s2_s3

        # Calculate dominance probabilities and weights
        p_dom_A = (sb1 + sb2) / S3
        w_A = 2 * p_dom_A - 1

        S2 = sb1 + sb2
        if S2 < 1e-9:
            w_B = 1.0  # Only one agent has non-zero bid
        else:
            p_dom_B = sb1 / S2
            w_B = 2 * p_dom_B - 1

        # Clamp weights to valid range [0,1]
        w_A = max(0.0, min(1.0, w_A))
        w_B = max(0.0, min(1.0, w_B))

        # Apply recursive score composition
        # Step 1: Combine top 2 agents
        s_1_2_intermediate_noise_pred = (
            1 - w_B
        ) * noise_pred_s1_s2 + w_B * noise_pred_s1

        # Step 2: Combine result with all 3 agents
        final_noise_pred = (
            1 - w_A
        ) * noise_pred_s1_s2_s3 + w_A * s_1_2_intermediate_noise_pred

        return final_noise_pred
