# Cluster Setup Guide for HuggingFace Cache

This guide explains how to properly set up and use the HuggingFace model cache on cluster environments to avoid slow downloads.

## Quick Start

### Option 1: Use the Automated Wrapper (Recommended)
```bash
cd scripts
./run_with_cache.sh
```

This automatically sets up the cache and runs the generation script.

### Option 2: Manual Setup
```bash
cd scripts
source setup_cache.sh
python generate_images_3_agent.py
```

## Cache Management

### Check Cache Status
```bash
python manage_cache.py info
```

### List Cached Models
```bash
python manage_cache.py list
```

### Check Cache Size
```bash
python manage_cache.py usage
```

### Clean Specific Model (if corrupted)
```bash
# Clean the FLUX model if download was interrupted
python manage_cache.py clean black-forest-labs/FLUX.1-schnell
```

### Clean All Cache
```bash
python manage_cache.py clean
```

## How It Works

### Cache Location Priority
The setup automatically tries these locations in order:
1. `/scratch/$USER/hf-cache` (fastest, recommended)
2. `/tmp/$USER/hf-cache`
3. `/dev/shm/$USER/hf-cache`
4. `/var/tmp/$USER/hf-cache`

### Environment Variables Set
- `HF_HOME`: Base cache directory
- `HF_HUB_CACHE`: Model repository cache
- `TRANSFORMERS_CACHE`: Transformers library cache

## Troubleshooting

### Interrupted Downloads
If a model download was interrupted, you'll see partial files. Clean them:
```bash
python manage_cache.py clean black-forest-labs/FLUX.1-schnell
```

### Permission Issues
Make sure your user has write access to the cache location:
```bash
ls -la /scratch/$USER/
```

### NFS vs Local Storage
Check if your cache is on local storage:
```bash
df -h $HF_HOME
```

Look for filesystem types like `ext4`, `xfs` (local) vs `nfs` (network).

### Custom Cache Location
If you need to specify a custom location:
```bash
export HF_CACHE_BASE=/your/custom/path
source setup_cache.sh
```

## Performance Comparison

| Storage Type | First Download | Subsequent Runs |
|-------------|----------------|-----------------|
| NFS/Network | 5-30 minutes   | 2-5 minutes     |
| Local SSD   | 2-5 minutes    | 30 seconds      |

## File Structure

```
scripts/
├── setup_cache.sh          # Environment setup
├── run_with_cache.sh       # Wrapper script
├── manage_cache.py         # Cache management utility
├── generate_images_3_agent.py  # Modified with auto-cache
└── generate_images_3_agent_multigpu.py  # Original multi-GPU script
```

## Integration with Existing Scripts

The `generate_images_3_agent.py` script now automatically:
1. Detects available local storage
2. Sets up cache directories
3. Configures environment variables
4. Reports cache location before starting

No changes needed to your workflow - just run the script normally.