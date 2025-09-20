import os
import json
import torch
from tqdm import tqdm

from diffusers import FluxPipeline
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps, calculate_shift
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils import is_torch_xla_available

# ===== CONFIGURATION SECTION =====
# Modify these parameters as needed

# Path configurations
PROMPTS_PATH = "../prompts/prompts_3_agent.json"  # Path to prompts file
OUTPUT_DIR = "images/images_3_agent"  # Output directory for generated images

# Sampling configuration
NUM_SAMPLES_PER_COMBINATION = 1  # Number of times to sample each prompt-bid combination
NUM_PROMPTS_TO_PROCESS = None  # Number of prompts to process (None = all prompts)

# Generation parameters
GUIDANCE_SCALE = 10.0  # Guidance scale for generation
NUM_INFERENCE_STEPS = 5  # Number of denoising steps
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float16

# Bidding combinations for 3 agents (b1, b2, b3)
BIDDING_COMBINATIONS_3_AGENT = [
    (1.0, 0.0, 0.0),  # Agent 1 dominant
    # (0.0, 1.0, 0.0),  # Agent 2 dominant
    # (0.0, 0.0, 1.0),  # Agent 3 dominant
    # (0.5, 0.5, 0.0),  # Agent 1 & 2 equal, Agent 3 none
    (0.33, 0.33, 0.33),  # All equal
    # (0.45, 0.45, 0.1),  # A1 & A2 strong and equal, A3 minor voice
    (0.4, 0.4, 0.2),  # A1 & A2 strong and equal, A3 minor voice
    # (0.45, 0.1, 0.45),  # A1 & A3 strong and equal, A2 minor voice
    # (0.1, 0.45, 0.45),  # A2 & A3 strong and equal, A1 minor voice
    # (0.4, 0.3, 0.3),  # A1 slightly more influential than A2 & A3 (who are equal)
    # (0.3, 0.4, 0.3),  # A2 slightly more influential
    # (0.3, 0.3, 0.4),  # A3 slightly more influential
    # (0.7, 0.2, 0.1),  # Strong A1, moderate A2, weak A3
    # (0.1, 0.7, 0.2),  # Strong A2, moderate A3, weak A1
    # (0.2, 0.1, 0.7),  # Strong A3, moderate A1, weak A2
    # (0.5, 0.3, 0.2),  # A1 > A2 > A3 with smaller gaps
    # (0.2, 0.5, 0.3),  # A2 > A3 > A1 with smaller gaps
    # (0.3, 0.2, 0.5),  # A3 > A1 > A2 with smaller gaps
    (0.6, 0.3, 0.1),  # Agent 1 > Agent 2 > Agent 3
    # (0.1, 0.6, 0.3),  # Agent 2 > Agent 3 > Agent 1
    # (0.1, 0.3, 0.6),  # Agent 3 > Agent 2 > Agent 1
    (0.6, 0.2, 0.2),  # Agent 1 > Agent 2 = Agent 3
]

# ===== END CONFIGURATION SECTION =====

# Load the prompts
with open(PROMPTS_PATH, "r") as f:
    prompts = json.load(f)

GLOBAL_WORK = False
DO_TRUE_CFG = False

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class FluxPipelineAuction(FluxPipeline):
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
        guidance_scale: float = 3.5,  # Kept for potential use with a general negative prompt
        negative_prompt: Optional[
            Union[str, List[str]]
        ] = None,  # For overall CFG if desired
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        # IP Adapter (largely kept from original, may need adjustment if used with 3 agents)
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        # Assuming a single set of IP adapter images, applied based on context or to all.
        # If per-agent IP adapter, this section would need more significant changes.
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[
            Dict[str, Any]
        ] = None,  # Kept, usage context might need review
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs.
        # For 3-agent, decide primary prompt for check_inputs or adapt check_inputs.
        # Here, we'll use agent1_prompt as the primary for the check, assuming similar structures.
        # A more robust check might iterate or check the longest/combined prompt.
        self.check_inputs(
            agent1_prompt,  # Or a representative prompt
            agent1_prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=agent1_prompt_embeds,  # if agent1 is primary for check
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=agent1_pooled_prompt_embeds,  # if agent1 is primary for check
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = (
            guidance_scale  # Used by transformer if it implements CFG
        )
        self._joint_attention_kwargs = (
            joint_attention_kwargs if joint_attention_kwargs is not None else {}
        )
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if agent1_prompt is not None and isinstance(agent1_prompt, str):
            batch_size = 1
        elif agent1_prompt is not None and isinstance(agent1_prompt, list):
            batch_size = len(agent1_prompt)
        elif agent1_prompt_embeds is not None:
            batch_size = agent1_prompt_embeds.shape[0]
        else:  # Fallback if only agent2 or agent3 prompt is given
            if agent2_prompt is not None and isinstance(agent2_prompt, str):
                batch_size = 1
            elif agent2_prompt is not None and isinstance(agent2_prompt, list):
                batch_size = len(agent2_prompt)
            elif agent2_prompt_embeds is not None:
                batch_size = agent2_prompt_embeds.shape[0]
            elif agent3_prompt is not None and isinstance(agent3_prompt, str):
                batch_size = 1
            elif agent3_prompt is not None and isinstance(agent3_prompt, list):
                batch_size = len(agent3_prompt)
            elif agent3_prompt_embeds is not None:
                batch_size = agent3_prompt_embeds.shape[0]
            else:
                raise ValueError(
                    "At least one agent prompt or its embeddings must be provided to determine batch_size."
                )

        device = self._execution_device
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None)
            if self.joint_attention_kwargs is not None
            else None
        )

        # Store agent data
        agents_initial_data = [
            {
                "id": "agent1",
                "prompt": agent1_prompt,
                "prompt_2": agent1_prompt_2,
                "bid": agent1_bid,
                "prompt_embeds": agent1_prompt_embeds,
                "pooled_prompt_embeds": agent1_pooled_prompt_embeds,
            },
            {
                "id": "agent2",
                "prompt": agent2_prompt,
                "prompt_2": agent2_prompt_2,
                "bid": agent2_bid,
                "prompt_embeds": agent2_prompt_embeds,
                "pooled_prompt_embeds": agent2_pooled_prompt_embeds,
            },
            {
                "id": "agent3",
                "prompt": agent3_prompt,
                "prompt_2": agent3_prompt_2,
                "bid": agent3_bid,
                "prompt_embeds": agent3_prompt_embeds,
                "pooled_prompt_embeds": agent3_pooled_prompt_embeds,
            },
        ]

        # Filter out agents with no prompt and no embeddings
        agents_active_data = [
            agent
            for agent in agents_initial_data
            if (
                agent["prompt"] is not None
                or (
                    agent["prompt_embeds"] is not None
                    and agent["pooled_prompt_embeds"] is not None
                )
            )
        ]

        if not agents_active_data:
            raise ValueError("No active agents with prompts or embeddings provided.")

        # Sort active agents by bid in descending order
        # If bids are equal, original order (agent1, agent2, agent3) is maintained due to stable sort
        sorted_agents_data = sorted(
            agents_active_data, key=lambda x: x["bid"], reverse=True
        )

        # Pad with dummy agents if less than 3 active agents (for simpler indexing, though logic should adapt)
        # This padding might be overly complex; simpler to just work with len(sorted_agents_data)
        # For now, let's assume we always want to attempt to use up to 3 if provided.
        # The core logic below will handle cases where fewer than 3 distinct prompts are used for s1, s1_s2, s1_s2_s3.

        s_agent1_data = sorted_agents_data[0] if len(sorted_agents_data) > 0 else None
        s_agent2_data = sorted_agents_data[1] if len(sorted_agents_data) > 1 else None
        s_agent3_data = sorted_agents_data[2] if len(sorted_agents_data) > 2 else None

        # Helper to prepend base prompt
        def _combine_prompt(p1, p2, base_p):
            res_p = ""
            if base_p:
                res_p = base_p
            if p1:
                res_p = f"{res_p} with {p1}" if res_p else p1
            if p2:  # for prompt_2
                res_p = (
                    f"{res_p} with {p2}" if res_p else p2
                )  # this logic might need refinement for prompt_2
            return res_p if res_p else None  # return None if all empty

        def _get_effective_prompt(agent_data, base_p, base_p2):
            if not agent_data:
                return None, None
            # Simplified: agent_prompt_2 is appended if it exists
            # A more complex handling of prompt_2 might be needed depending on its intended use in Flux
            main_p = agent_data["prompt"]
            sec_p = agent_data["prompt_2"]

            # Handle cases where prompts might be lists (take first element for simplicity if batch_size > 1 and combining)
            # This part requires careful thought if prompts are lists and base_prompt is a single string.
            # For now, assume prompts are strings or handled by encode_prompt if lists.

            # If base_prompt is a list and agent_prompt is a list, they should have compatible batch sizes or logic to handle it.
            # Assuming string prompts for combination simplicity here.

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

        # Encode prompts for s1, s1+s2, s1+s2+s3
        # s1_prompt
        s1_full_prompt, s1_full_prompt_2 = _get_effective_prompt(
            s_agent1_data, base_prompt, base_prompt_2
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
            if s_agent1_data
            and (s1_full_prompt or (s_agent1_data["prompt_embeds"] is not None))
            else (None, None, None)
        )

        # s1_s2_prompt (combined prompt of s_agent1 and s_agent2)
        s1_s2_full_prompt_parts = []
        s1_s2_full_prompt_2_parts = []

        if base_prompt:
            s1_s2_full_prompt_parts.append(base_prompt)
        if base_prompt_2:
            s1_s2_full_prompt_2_parts.append(base_prompt_2)

        if s_agent1_data and s_agent1_data["prompt"]:
            s1_s2_full_prompt_parts.append(s_agent1_data["prompt"])
        if s_agent1_data and s_agent1_data["prompt_2"]:
            s1_s2_full_prompt_2_parts.append(s_agent1_data["prompt_2"])
        if s_agent2_data and s_agent2_data["prompt"]:
            s1_s2_full_prompt_parts.append(s_agent2_data["prompt"])
        if s_agent2_data and s_agent2_data["prompt_2"]:
            s1_s2_full_prompt_2_parts.append(s_agent2_data["prompt_2"])

        s1_s2_combined_prompt = (
            ", ".join(p for p in s1_s2_full_prompt_parts if p)
            if s1_s2_full_prompt_parts
            else None
        )
        s1_s2_combined_prompt_2 = (
            ", ".join(p for p in s1_s2_full_prompt_2_parts if p)
            if s1_s2_full_prompt_2_parts
            else None
        )

        # For combined prompts, we generally don't use pre-computed embeddings unless they are for the exact combination
        (s1_s2_embeds, s1_s2_pooled_embeds, s1_s2_text_ids) = (
            self.encode_prompt(
                prompt=s1_s2_combined_prompt,
                prompt_2=s1_s2_combined_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
            if s1_s2_combined_prompt
            else (s1_embeds, s1_pooled_embeds, s1_text_ids)
        )  # Fallback to s1 if s2 not active for combo

        # s1_s2_s3_prompt (combined prompt of s_agent1, s_agent2, s_agent3)
        s1_s2_s3_full_prompt_parts = []
        s1_s2_s3_full_prompt_2_parts = []

        if base_prompt:
            s1_s2_s3_full_prompt_parts.append(base_prompt)
        if base_prompt_2:
            s1_s2_s3_full_prompt_2_parts.append(base_prompt_2)

        for agent_data in [s_agent1_data, s_agent2_data, s_agent3_data]:
            if agent_data and agent_data["prompt"]:
                s1_s2_s3_full_prompt_parts.append(agent_data["prompt"])
            if agent_data and agent_data["prompt_2"]:
                s1_s2_s3_full_prompt_2_parts.append(agent_data["prompt_2"])

        s1_s2_s3_combined_prompt = (
            " and ".join(p for p in s1_s2_s3_full_prompt_parts if p)
            if s1_s2_s3_full_prompt_parts
            else None
        )
        s1_s2_s3_combined_prompt_2 = (
            " and ".join(p for p in s1_s2_s3_full_prompt_2_parts if p)
            if s1_s2_s3_full_prompt_2_parts
            else None
        )

        (s1_s2_s3_embeds, s1_s2_s3_pooled_embeds, s1_s2_s3_text_ids) = (
            self.encode_prompt(
                prompt=s1_s2_s3_combined_prompt,
                prompt_2=s1_s2_s3_combined_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
            if s1_s2_s3_combined_prompt
            else (s1_s2_embeds, s1_s2_pooled_embeds, s1_s2_text_ids)
        )  # Fallback further

        # If all prompts somehow ended up None (e.g. only empty base_prompt provided)
        # This should be caught earlier by active_agents check or encode_prompt returning errors for None prompt.
        # For safety, ensure at least one set of embeddings is not None.
        if (
            s1_s2_s3_embeds is None
        ):  # This implies all individual prompts were also None or failed to encode
            if s1_s2_embeds is not None:  # Should not happen if logic is correct
                s1_s2_s3_embeds, s1_s2_s3_pooled_embeds, s1_s2_s3_text_ids = (
                    s1_s2_embeds,
                    s1_s2_pooled_embeds,
                    s1_s2_text_ids,
                )
            elif s1_embeds is not None:
                s1_s2_s3_embeds, s1_s2_s3_pooled_embeds, s1_s2_s3_text_ids = (
                    s1_embeds,
                    s1_pooled_embeds,
                    s1_text_ids,
                )
            else:
                raise ValueError(
                    "Failed to generate any valid prompt embeddings for the agents."
                )

        global GLOBAL_WORK, DO_TRUE_CFG
        GLOBAL_WORK = True  # Score composition is always happening with multiple agents
        DO_TRUE_CFG = True

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            s1_s2_s3_embeds.dtype,  # Use dtype from one of the valid embeddings
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        # sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        # Using the original sigmas if provided, else default retrieve_timesteps logic
        image_seq_len = (
            latents.shape[2] * self.vae_scale_factor
        )  # latents are H/scale, W/scale
        mu = calculate_shift(
            image_seq_len,  # This should be image_seq_len, not latent.shape[1]
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

        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # IP Adapter (simplified, assuming one set of IP adapter images for the whole process)
        # If per-agent IP, logic here would need significant changes based on which noise_pred it applies to.
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            # This was agent1_image_embeds in the original code.
            # Let's assume it's a general IP adapter effect.
            processed_ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
            self._joint_attention_kwargs["ip_adapter_image_embeds"] = (
                processed_ip_adapter_image_embeds
            )

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t
                timestep_expanded = t.expand(latents.shape[0]).to(latents.dtype)

                # Get noise predictions for s1, s1+s2, s1+s2+s3
                # These should default to the most comprehensive available if specific ones are None

                _s1_embeds = s1_embeds if s1_embeds is not None else s1_s2_s3_embeds
                _s1_pooled = (
                    s1_pooled_embeds
                    if s1_pooled_embeds is not None
                    else s1_s2_s3_pooled_embeds
                )
                _s1_ids = s1_text_ids if s1_text_ids is not None else s1_s2_s3_text_ids
                noise_pred_s1 = self.transformer(
                    hidden_states=latents,
                    timestep=timestep_expanded / 1000,
                    guidance=guidance,
                    pooled_projections=_s1_pooled,
                    encoder_hidden_states=_s1_embeds,
                    txt_ids=_s1_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                _s1_s2_embeds = (
                    s1_s2_embeds if s1_s2_embeds is not None else s1_s2_s3_embeds
                )
                _s1_s2_pooled = (
                    s1_s2_pooled_embeds
                    if s1_s2_pooled_embeds is not None
                    else s1_s2_s3_pooled_embeds
                )
                _s1_s2_ids = (
                    s1_s2_text_ids if s1_s2_text_ids is not None else s1_s2_s3_text_ids
                )
                noise_pred_s1_s2 = self.transformer(
                    hidden_states=latents,
                    timestep=timestep_expanded / 1000,
                    guidance=guidance,
                    pooled_projections=_s1_s2_pooled,
                    encoder_hidden_states=_s1_s2_embeds,
                    txt_ids=_s1_s2_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                noise_pred_s1_s2_s3 = self.transformer(
                    hidden_states=latents,
                    timestep=timestep_expanded / 1000,
                    guidance=guidance,
                    pooled_projections=s1_s2_s3_pooled_embeds,
                    encoder_hidden_states=s1_s2_s3_embeds,
                    txt_ids=s1_s2_s3_text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # Generalized Score Composition for n=3
                sb1 = (
                    sorted_agents_data[0]["bid"] if len(sorted_agents_data) > 0 else 0.0
                )
                sb2 = (
                    sorted_agents_data[1]["bid"] if len(sorted_agents_data) > 1 else 0.0
                )
                sb3 = (
                    sorted_agents_data[2]["bid"] if len(sorted_agents_data) > 2 else 0.0
                )

                # Ensure bids are non-negative
                sb1, sb2, sb3 = max(0, sb1), max(0, sb2), max(0, sb3)

                S3 = sb1 + sb2 + sb3

                if S3 < 1e-9:  # All bids are effectively zero
                    noise_pred = noise_pred_s1_s2_s3  # Default to the most inclusive prompt combination
                else:
                    # P_dom_A = (sb1 + sb2) / S3
                    # w_A is the weight for the s_{1,2}^* part (composition of top two agents)
                    # w_A = 2 * P_dom_A - 1
                    # P_dom_A ranges [2/3, 1] if sb1>=sb2>=sb3>0, so w_A ranges [1/3, 1]
                    # If sb3 is largest, sorting puts it as sb1. If sb1>0, sb2=0, sb3=0, P_dom_A=1, w_A=1.
                    p_dom_A = (sb1 + sb2) / S3
                    w_A = 2 * p_dom_A - 1

                    S2 = sb1 + sb2
                    if (
                        S2 < 1e-9
                    ):  # sb1 and sb2 are zero (implies only sb3 could be non-zero, which means S3=sb3)
                        # This configuration implies sb1 (after sort) is actually the agent with bid sb3.
                        # If sb1 (highest bid) = 0, then all bids are 0, handled by S3 < 1e-9.
                        # So if S2 < 1e-9 here, it means sb1 > 0 is not possible.
                        # This path (S2 < 1e-9 inside S3 >= 1e-9) implies sb1=0, sb2=0, and sb3 > 0.
                        # After sorting, sb1_sorted = sb3_orig, sb2_sorted=0, sb3_sorted=0.
                        # Then p_dom_A = (sb1_sorted + 0) / sb1_sorted = 1. So w_A = 1.
                        # And S2_calc = sb1_sorted + 0. p_dom_B = sb1_sorted / sb1_sorted = 1. So w_B = 1.
                        # s_1_2_intermediate = (1-1)*noise_pred_s1_s2 + 1*noise_pred_s1 = noise_pred_s1
                        # noise_pred = (1-1)*noise_pred_s1_s2_s3 + 1*s_1_2_intermediate = noise_pred_s1.
                        # This seems correct: if only one agent has a bid, its noise_pred is used.
                        w_B = 1.0  # If sb1+sb2 is zero, but sb1 might be positive if sb2 is negative (not allowed)
                    # If sb1 and sb2 are zero, then s_agent1 is dominant in {s_agent1, s_agent2} group (vacuously true or weight 1)
                    # If sb1 > 0 and sb2 = 0, S2 = sb1. p_dom_B = sb1/sb1 = 1. w_B = 1.
                    else:
                        # P_dom_B = sb1 / S2
                        # w_B is the weight for the s_1^* part (most dominant agent in top two)
                        # w_B = 2 * P_dom_B - 1
                        # P_dom_B ranges [0.5, 1] if sb1>=sb2>0, so w_B ranges [0, 1]
                        p_dom_B = sb1 / S2
                        w_B = 2 * p_dom_B - 1

                    # Clamping weights to [0,1] for robustness, though theory suggests they should fall in range.
                    w_A = max(0.0, min(1.0, w_A))
                    w_B = max(0.0, min(1.0, w_B))

                    # Recursive application:
                    # s_{1,2}^* (intermediate) = (1-w_B) * s(c_{1,2}) + w_B * s(c_1)
                    s_1_2_intermediate_noise_pred = (
                        1 - w_B
                    ) * noise_pred_s1_s2 + w_B * noise_pred_s1

                    # s_{1,3}^* (final) = (1-w_A) * s(c_{1,2,3}) + w_A * s_{1,2}^*
                    noise_pred = (
                        1 - w_A
                    ) * noise_pred_s1_s2_s3 + w_A * s_1_2_intermediate_noise_pred

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k_cb in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k_cb] = locals()[k_cb]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    # Potentially update other things like prompt_embeds if callback modifies them
                    # For now, assume only latents is primary output.

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    # xm.mark_step() # type: ignore
                    pass

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(
                latents, height, width, self.vae_scale_factor
            )  # Ensure this exists or is no-op
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return FluxPipelineOutput(images=image)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# new_cache_path = "/n/holylabs/LABS/sham_lab/Lab/lillian/mit6.S982/projects/diffusion_auctions/.hfcache"
pipe_auction = FluxPipelineAuction.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=TORCH_DTYPE,
    # cache_dir=new_cache_path
).to(device)


def generate_and_save_flux_image_3_agents(
    pipe_auction, data_item, index, output_dir, bids, sample_idx=0
):
    agent1_prompt = data_item.get("agent1_prompt", "")
    agent2_prompt = data_item.get("agent2_prompt", "")
    agent3_prompt = data_item.get("agent3_prompt", "")  # New agent
    base_prompt = data_item.get("base_prompt", "")  # Base prompt

    bid1, bid2, bid3 = bids

    filename_base = (
        f"idx{index:03d}_b1_{bid1:.2f}_b2_{bid2:.2f}_b3_{bid3:.2f}_s{sample_idx:02d}"
    )
    output_path = os.path.join(output_dir, f"{filename_base}.png")

    print(
        f"Generating item {index}, sample {sample_idx}: Bids=({bid1:.2f}, {bid2:.2f}, {bid3:.2f})"
    )
    print(
        f"  A1: {agent1_prompt[:30]}... | A2: {agent2_prompt[:30]}... | A3: {agent3_prompt[:30]}..."
    )

    # Call the pipeline with 3 agents and their bids
    try:
        images = pipe_auction(
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

        # Save the image
        if images and hasattr(images, "images") and len(images.images) > 0:
            images.images[0].save(output_path)
            print(f"Saved image to {output_path}")
        else:
            print(f"No image generated for {output_path}")

    except Exception as e:
        print(f"Error generating image for {filename_base}: {e}")
        output_path = None  # Indicate failure

    return output_path


results_list = []
num_items_to_process = (
    NUM_PROMPTS_TO_PROCESS if NUM_PROMPTS_TO_PROCESS is not None else len(prompts)
)
total_images = (
    num_items_to_process
    * len(BIDDING_COMBINATIONS_3_AGENT)
    * NUM_SAMPLES_PER_COMBINATION
)
print("Starting 3-agent image generation:")
print(f"  - {num_items_to_process} prompts")
print(f"  - {len(BIDDING_COMBINATIONS_3_AGENT)} bidding combinations")
print(f"  - {NUM_SAMPLES_PER_COMBINATION} samples per combination")
print(f"  - Total images to generate: {total_images}")


# Bidding combinations are now defined in the configuration section above

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Image outputs will be saved to: {OUTPUT_DIR}")

generation_results = []

for i, item_data in enumerate(
    tqdm(prompts[:num_items_to_process], desc="Processing Prompts")
):
    prompt_specific_output_dir = os.path.join(OUTPUT_DIR, f"prompt_{i:03d}")
    os.makedirs(prompt_specific_output_dir, exist_ok=True)

    for bids_tuple in BIDDING_COMBINATIONS_3_AGENT:
        for sample_idx in range(NUM_SAMPLES_PER_COMBINATION):
            generated_image_path = generate_and_save_flux_image_3_agents(
                pipe_auction,
                item_data,
                i,
                prompt_specific_output_dir,
                bids_tuple,
                sample_idx,
            )
            if generated_image_path:
                generation_results.append(
                    {
                        "item_index": i,
                        "bids": bids_tuple,
                        "sample_index": sample_idx,
                        "agent1_prompt": item_data["agent1_prompt"],
                        "agent2_prompt": item_data["agent2_prompt"],
                        "agent3_prompt": item_data["agent3_prompt"],
                        "base_prompt": item_data["base_prompt"],
                        "image_path": generated_image_path,
                    }
                )

results_filename = os.path.join(OUTPUT_DIR, "generation_log.json")
with open(results_filename, "w") as f:
    json.dump(generation_results, f, indent=2)
print(f"\nSaved generation log to {results_filename}")
print("3-agent image generation example finished.")
