#!/usr/bin/env python3
"""
HuggingFace Cache Management Utility

This script helps manage the HuggingFace model cache on cluster environments,
including listing cached models, cleaning broken downloads, and cache statistics.

Usage:
    python manage_cache.py list                    # List all cached models
    python manage_cache.py usage                   # Show cache disk usage
    python manage_cache.py clean [model_name]      # Clean specific model or all
    python manage_cache.py info                    # Show cache configuration

Authors: Lillian Sun, Warren Zhu, Henry Huang
"""

import os
import argparse
import shutil
from pathlib import Path


def get_cache_dir():
    """Get the HuggingFace cache directory."""
    # Check environment variables in order of preference
    cache_dir = os.environ.get('HF_HUB_CACHE')
    if not cache_dir:
        cache_dir = os.environ.get('HF_HOME', '~/.cache/huggingface')
        cache_dir = os.path.join(cache_dir, 'hub')

    return Path(cache_dir).expanduser()


def format_size(size_bytes):
    """Format bytes to human readable format."""
    if size_bytes == 0:
        return "0B"

    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}PB"


def get_directory_size(path):
    """Get total size of directory."""
    total = 0
    try:
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total += os.path.getsize(filepath)
                except (OSError, IOError):
                    pass
    except (OSError, IOError):
        pass
    return total


def list_cached_models():
    """List all cached models with their sizes."""
    cache_dir = get_cache_dir()

    if not cache_dir.exists():
        print(f"Cache directory does not exist: {cache_dir}")
        return

    print(f"HuggingFace cache directory: {cache_dir}")
    print("-" * 80)

    model_dirs = list(cache_dir.glob("models--*"))
    if not model_dirs:
        print("No cached models found.")
        return

    total_size = 0
    models_info = []

    for model_dir in model_dirs:
        if model_dir.is_dir():
            model_name = model_dir.name.replace("models--", "").replace("--", "/")
            size = get_directory_size(model_dir)
            total_size += size
            models_info.append((model_name, size, model_dir))

    # Sort by size (largest first)
    models_info.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Model Name':<50} {'Size':<15} {'Path'}")
    print("-" * 80)

    for model_name, size, path in models_info:
        print(f"{model_name:<50} {format_size(size):<15} {path.name}")

    print("-" * 80)
    print(f"Total cached models: {len(models_info)}")
    print(f"Total cache size: {format_size(total_size)}")


def show_cache_usage():
    """Show cache disk usage statistics."""
    cache_dir = get_cache_dir()

    if not cache_dir.exists():
        print(f"Cache directory does not exist: {cache_dir}")
        return

    total_size = get_directory_size(cache_dir)

    # Get filesystem stats if possible
    try:
        statvfs = os.statvfs(cache_dir)
        total_space = statvfs.f_frsize * statvfs.f_blocks
        free_space = statvfs.f_frsize * statvfs.f_available
        used_space = total_space - free_space

        print(f"Cache directory: {cache_dir}")
        print(f"Cache size: {format_size(total_size)}")
        print(f"Filesystem total: {format_size(total_space)}")
        print(f"Filesystem used: {format_size(used_space)} ({used_space/total_space*100:.1f}%)")
        print(f"Filesystem free: {format_size(free_space)} ({free_space/total_space*100:.1f}%)")
        print(f"Cache as % of filesystem: {total_size/total_space*100:.2f}%")

    except (OSError, AttributeError):
        print(f"Cache directory: {cache_dir}")
        print(f"Cache size: {format_size(total_size)}")
        print("Filesystem stats not available")


def clean_cache(model_name=None):
    """Clean cache for specific model or all models."""
    cache_dir = get_cache_dir()

    if not cache_dir.exists():
        print(f"Cache directory does not exist: {cache_dir}")
        return

    if model_name:
        # Clean specific model
        model_pattern = f"models--{model_name.replace('/', '--')}"
        model_dirs = list(cache_dir.glob(model_pattern))

        if not model_dirs:
            print(f"Model not found in cache: {model_name}")
            print("Available models:")
            list_cached_models()
            return

        for model_dir in model_dirs:
            size = get_directory_size(model_dir)
            print(f"Removing {model_dir.name} ({format_size(size)})...")
            shutil.rmtree(model_dir)
            print(f"Removed: {model_name}")

    else:
        # Clean all cache
        print(f"WARNING: This will remove ALL cached models from {cache_dir}")
        response = input("Are you sure? (y/N): ")

        if response.lower() == 'y':
            total_size = get_directory_size(cache_dir)
            print(f"Removing {format_size(total_size)} of cached data...")

            for item in cache_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

            print("Cache cleared successfully")
        else:
            print("Cache cleaning cancelled")


def show_cache_info():
    """Show cache configuration and environment info."""
    print("HuggingFace Cache Configuration")
    print("=" * 50)

    # Environment variables
    env_vars = ['HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE', 'HF_DATASETS_CACHE']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")

    print()

    # Actual cache directory
    cache_dir = get_cache_dir()
    print(f"Effective cache directory: {cache_dir}")
    print(f"Cache directory exists: {cache_dir.exists()}")

    if cache_dir.exists():
        print(f"Cache directory writable: {os.access(cache_dir, os.W_OK)}")
        print(f"Cache directory readable: {os.access(cache_dir, os.R_OK)}")

    # Check for environment setup script
    possible_env_files = [
        cache_dir.parent / "hf_env.sh",
        Path("~/.hf_env.sh").expanduser(),
        Path("/tmp/hf_env.sh"),
    ]

    print("\nEnvironment setup scripts:")
    for env_file in possible_env_files:
        if env_file.exists():
            print(f"  Found: {env_file}")
        else:
            print(f"  Missing: {env_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage HuggingFace model cache",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # List command
    subparsers.add_parser('list', help='List all cached models with sizes')

    # Usage command
    subparsers.add_parser('usage', help='Show cache disk usage statistics')

    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean cache')
    clean_parser.add_argument('model', nargs='?', help='Model name to clean (e.g., black-forest-labs/FLUX.1-schnell)')

    # Info command
    subparsers.add_parser('info', help='Show cache configuration')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == 'list':
        list_cached_models()
    elif args.command == 'usage':
        show_cache_usage()
    elif args.command == 'clean':
        clean_cache(args.model)
    elif args.command == 'info':
        show_cache_info()


if __name__ == '__main__':
    main()