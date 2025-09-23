# VLM Quality Assessment Setup Guide

This guide helps you set up Vision Language Model (VLM) quality assessment for the diffusion auctions project.

## Overview

The VLM quality assessment system enhances the existing CLIP-based alignment analysis by adding sophisticated image quality evaluation using state-of-the-art vision language models. This provides detailed quality metrics across multiple dimensions for your generated images.

## Prerequisites

- **GPU**: NVIDIA A100, L40S, or similar (8GB+ VRAM recommended)
- **CUDA**: Compatible CUDA installation
- **Python**: 3.8+
- **Storage**: ~20GB for model weights

## Installation

### 1. Install Core Dependencies

```bash
# Install latest transformers from source (required for Qwen2.5-VL)
pip install git+https://github.com/huggingface/transformers

# Install VLM-specific dependencies
pip install qwen-vl-utils[decord]
pip install flash-attn>=2.0.0
pip install accelerate
pip install einops timm

# Install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install All Requirements

```bash
# From project root directory
pip install -r requirements.txt
```

## Configuration

### 1. VLM Configuration File

The VLM system is configured via `config/vlm_config.json`. Key settings:

```json
{
  "primary_model": {
    "name": "qwen2.5-vl-7b",
    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    "enabled": true
  },
  "processing": {
    "batch_size": 4,
    "enable_caching": true
  }
}
```

### 2. Model Selection

**Primary Model (Recommended):**
- **Qwen2.5-VL-7B**: Best performance-to-memory ratio, ~29GB VRAM
- Excellent image quality assessment capabilities
- Native dynamic resolution support

**Fallback Models:**
- **InternVL-1.5**: Alternative high-performance option
- **LLaVA-Next-13B**: Backup option (disabled by default)

### 3. Quality Assessment Dimensions

The system evaluates 5 quality dimensions on a 0.0-1.0 scale:

1. **Overall Sharpness**: Image clarity and focus
2. **Visual Coherence**: Realism and consistency
3. **Color Quality**: Color balance and saturation
4. **Composition**: Artistic composition and framing
5. **Detail Preservation**: Fine detail retention

## Usage

### 1. Basic Usage (2-Agent)

```bash
cd scripts

# CLIP-only analysis (existing functionality)
python calculate_alignment_2_agent.py

# Enable VLM quality assessment
python calculate_alignment_2_agent.py --enable_vlm

# Custom VLM configuration
python calculate_alignment_2_agent.py --enable_vlm --vlm_config ../config/custom_vlm_config.json
```

### 2. Basic Usage (3-Agent)

```bash
cd scripts

# CLIP-only analysis (existing functionality)
python calculate_alignment_3_agent.py

# Enable VLM quality assessment
python calculate_alignment_3_agent.py --enable_vlm
```

### 3. Filtered Processing

```bash
# Process specific prompt
python calculate_alignment_2_agent.py --enable_vlm --prompt_index 0

# Process specific bid combination
python calculate_alignment_2_agent.py --enable_vlm --bid_combination "1.0,0.0"

# Process specific sample
python calculate_alignment_2_agent.py --enable_vlm --sample_index 5
```

## Output Format

### Enhanced JSON Structure

The VLM system integrates seamlessly with existing output:

```json
{
  "metadata": {
    "prompt_index": 0,
    "bids": [0.6, 0.4],
    "sample_index": 0
  },
  "alignment_scores": {
    "base_alignment": 0.85,
    "agent1_alignment": 0.92,
    "agent2_alignment": 0.78
  },
  "quality_assessment": {
    "clip_quality": 0.82,
    "vlm_quality": {
      "overall_sharpness": 0.87,
      "visual_coherence": 0.91,
      "color_quality": 0.85,
      "composition": 0.89,
      "detail_preservation": 0.83,
      "overall_quality": 0.87,
      "reasoning": "High quality image with excellent clarity...",
      "model_used": "qwen2.5-vl-7b"
    }
  },
  "welfare_metrics": {
    "weighted_alignment": 0.856,
    "total_welfare": 1.70
  }
}
```

## Performance Considerations

### Memory Management

- **Qwen2.5-VL-7B**: ~29GB VRAM
- **Batch Processing**: Configured for 4 images per batch
- **Automatic Cleanup**: GPU memory cleared between batches

### Processing Speed

- **VLM Assessment**: ~2-5 seconds per image
- **Caching**: Results cached to avoid redundant processing
- **Fallback**: Graceful degradation to CLIP-only if VLM fails

### Cost Optimization

- **Local Hosting**: No API costs (unlike GPT-4V)
- **Smart Caching**: Avoids repeated assessments
- **Configurable**: Easy to disable VLM for faster processing

## Troubleshooting

### Common Issues

#### 1. Model Loading Fails

```bash
# Check transformers version
pip show transformers  # Should be >= 4.49.0

# Reinstall latest transformers
pip install git+https://github.com/huggingface/transformers --force-reinstall
```

#### 2. CUDA/Flash Attention Issues

```bash
# Install flash attention separately
pip install flash-attn --no-build-isolation

# Check CUDA compatibility
python -c "import torch; print(torch.cuda.is_available())"
```

#### 3. Memory Issues

Reduce batch size in config:
```json
{
  "processing": {
    "batch_size": 2  // Reduce from 4
  }
}
```

#### 4. Model Download Issues

```bash
# Ensure adequate storage space (20GB+)
df -h

# Check internet connection for model downloads
curl -I https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
```

### Error Messages

**"No VLM models could be loaded"**
- Check GPU availability and VRAM
- Verify transformers version >= 4.49.0
- Check internet connection for model downloads

**"Failed to load VLM assessor"**
- Script continues with CLIP-only assessment
- Check error details in logs: `logs/vlm_assessment.log`

**"VLM assessment failed for image"**
- Individual image processing failed
- Falls back to default quality scores
- Check image file integrity

## Performance Monitoring

### Logs

VLM assessment logs are saved to:
- **Location**: `logs/vlm_assessment.log`
- **Level**: Configurable in `vlm_config.json`
- **Content**: Model loading, processing progress, errors

### Cache Management

Cache files stored in:
- **Location**: `vlm_cache/`
- **Format**: JSON files with MD5 hashes
- **Cleanup**: Manual deletion if needed

```bash
# Clear VLM cache
rm -rf vlm_cache/*
```

## Advanced Configuration

### Custom Prompts

Modify quality assessment prompts in `config/vlm_config.json`:

```json
{
  "quality_assessment": {
    "prompt_template": "Your custom quality assessment prompt..."
  }
}
```

### Multiple Model Comparison

Enable multiple models for comparison:

```json
{
  "fallback_models": [
    {
      "name": "internvl-1.5",
      "enabled": true
    },
    {
      "name": "llava-next-13b",
      "enabled": true
    }
  ]
}
```

### Batch Size Optimization

Optimize batch size for your hardware:

```json
{
  "processing": {
    "batch_size": 8,  // Increase for more VRAM
    "timeout_seconds": 60  // Adjust timeout
  }
}
```

## Integration with Existing Workflow

The VLM system is designed to seamlessly integrate with your existing alignment calculation workflow:

1. **Backward Compatible**: All existing scripts work unchanged
2. **Optional**: VLM assessment is opt-in via `--enable_vlm` flag
3. **Graceful Fallback**: Falls back to CLIP-only if VLM fails
4. **Preserves Output**: Maintains existing JSON structure with additions

## Next Steps

1. **Run Test**: Try VLM assessment on a small subset first
2. **Monitor Performance**: Check GPU usage and processing speed
3. **Adjust Configuration**: Optimize batch size and model selection
4. **Scale Up**: Process your full 10,000 image dataset

For questions or issues, check the logs at `logs/vlm_assessment.log` or refer to the troubleshooting section above.