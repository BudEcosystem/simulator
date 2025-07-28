#!/usr/bin/env python3
"""
CLI tool for managing LLM settings.
"""

import sys
import argparse
from typing import Optional

from .settings_manager import LLMSettingsManager


def get_settings():
    """Get current LLM settings."""
    settings = LLMSettingsManager()
    base_url = settings.get('llm_base_url')
    model = settings.get('llm_model')
    
    print("\nCurrent LLM Settings:")
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")


def update_settings(base_url: Optional[str] = None, model: Optional[str] = None):
    """Update LLM settings."""
    settings = LLMSettingsManager()
    results = {}
    
    if base_url is not None:
        results['base_url'] = settings.set('llm_base_url', base_url)
    
    if model is not None:
        results['model'] = settings.set('llm_model', model)
    
    print("\nUpdate Results:")
    for key, success in results.items():
        print(f"{key}: {'Success' if success else 'Failed'}")
    
    if results:
        print("\nNew Settings:")
        get_settings()


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Manage LLM settings")
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Get command
    get_parser = subparsers.add_parser('get', help='Get current settings')
    
    # Set command
    set_parser = subparsers.add_parser('set', help='Update settings')
    set_parser.add_argument('--base-url', help='Set LLM base URL')
    set_parser.add_argument('--model', help='Set default LLM model')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'get':
        get_settings()
    elif args.command == 'set':
        if not args.base_url and not args.model:
            print("Error: Please provide at least one setting to update (--base-url or --model)")
            sys.exit(1)
        update_settings(args.base_url, args.model)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main() 