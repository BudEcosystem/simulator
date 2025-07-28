"""
Command-line interface for managing models in the database.
"""

import argparse
import json
import sys
from typing import Optional
from tabulate import tabulate

from .connection import DatabaseConnection
from .model_manager import ModelManager
from .hf_integration import HuggingFaceModelImporter
from .model_loader import patch_genz_model_dict, DynamicModelCollection


def format_size(num_bytes: Optional[int]) -> str:
    """Format byte size in human-readable format."""
    if num_bytes is None:
        return "N/A"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def format_number(num: Optional[int]) -> str:
    """Format large numbers with suffixes."""
    if num is None:
        return "N/A"
    
    if num < 1000:
        return str(num)
    elif num < 1_000_000:
        return f"{num/1000:.1f}K"
    elif num < 1_000_000_000:
        return f"{num/1_000_000:.1f}M"
    else:
        return f"{num/1_000_000_000:.1f}B"


class ModelCLI:
    """CLI for model management."""
    
    def __init__(self):
        self.db = DatabaseConnection()
        self.manager = ModelManager(self.db)
        self.importer = HuggingFaceModelImporter(self.manager)
    
    def list_models(self, args):
        """List all models in the database."""
        filters = {}
        if args.type:
            filters['model_type'] = args.type
        if args.attention:
            filters['attention_type'] = args.attention
        if args.source:
            filters['source'] = args.source
        
        models = self.manager.list_models(**filters)
        
        if not models:
            print("No models found.")
            return
        
        # Prepare table data
        headers = ['Model ID', 'Type', 'Attention', 'Parameters', 'Source']
        rows = []
        
        for model in models:
            rows.append([
                model['model_id'],
                model['model_type'] or 'N/A',
                model['attention_type'] or 'N/A',
                format_number(model['parameter_count']),
                model['source']
            ])
        
        print(f"\nFound {len(models)} models:")
        print(tabulate(rows, headers=headers, tablefmt='grid'))
    
    def show_model(self, args):
        """Show detailed information about a model."""
        model = self.manager.get_model(args.model_id)
        
        if not model:
            print(f"Model '{args.model_id}' not found.")
            return
        
        print(f"\nModel: {model['model_id']}")
        print(f"Name: {model['model_name']}")
        print(f"Source: {model['source']}")
        print(f"Type: {model['model_type'] or 'N/A'}")
        print(f"Attention: {model['attention_type'] or 'N/A'}")
        print(f"Parameters: {format_number(model['parameter_count'])}")
        print(f"Created: {model['created_at']}")
        print(f"Updated: {model['updated_at']}")
        
        # Show quality metrics
        metrics = self.manager.get_quality_metrics(args.model_id)
        if metrics:
            print("\nQuality Metrics:")
            metric_rows = []
            for metric in metrics:
                shots_str = f"{metric['shots']} shots" if metric['shots'] else "N/A"
                metric_rows.append([
                    metric['metric_name'],
                    f"{metric['metric_value']:.3f}",
                    shots_str
                ])
            print(tabulate(metric_rows, headers=['Metric', 'Value', 'Shots'], tablefmt='simple'))
        
        # Show cached values
        if args.show_cache:
            print("\nCached Values:")
            # This would require extending the manager to list cache entries
            print("(Cache listing not implemented)")
        
        # Show config if requested
        if args.show_config:
            config = self.manager.get_model_config(args.model_id)
            print("\nConfiguration:")
            print(json.dumps(config, indent=2))
    
    def import_model(self, args):
        """Import a model from HuggingFace."""
        print(f"Importing model '{args.model_id}' from HuggingFace...")
        
        success = self.importer.import_model(args.model_id, force_update=args.force)
        
        if success:
            print(f"✓ Successfully imported {args.model_id}")
            
            # Validate import
            validation = self.importer.validate_import(args.model_id)
            if validation['errors']:
                print("\nWarnings:")
                for error in validation['errors']:
                    print(f"  - {error}")
        else:
            print(f"✗ Failed to import {args.model_id}")
    
    def delete_model(self, args):
        """Delete a model from the database."""
        model = self.manager.get_model(args.model_id)
        
        if not model:
            print(f"Model '{args.model_id}' not found.")
            return
        
        if not args.yes:
            confirm = input(f"Are you sure you want to delete '{args.model_id}'? [y/N] ")
            if confirm.lower() != 'y':
                print("Cancelled.")
                return
        
        success = self.manager.delete_model(args.model_id, hard_delete=args.hard)
        
        if success:
            action = "permanently deleted" if args.hard else "soft deleted"
            print(f"✓ Model '{args.model_id}' {action}.")
        else:
            print(f"✗ Failed to delete model '{args.model_id}'.")
    
    def sync_models(self, args):
        """Sync models with HuggingFace or static definitions."""
        if args.static:
            # Sync static models to database
            print("Syncing static models to database...")
            
            # Patch GenZ to get dynamic collection
            if patch_genz_model_dict():
                from Models.default_models import MODEL_DICT
                if isinstance(MODEL_DICT, DynamicModelCollection):
                    synced = MODEL_DICT.sync_static_models_to_db()
                    print(f"✓ Synced {synced} static models to database.")
                else:
                    print("✗ Failed to get dynamic MODEL_DICT.")
            else:
                print("✗ Failed to patch GenZ MODEL_DICT.")
        else:
            # Sync existing HF models
            print("Syncing HuggingFace models...")
            results = self.importer.sync_existing_models()
            
            success_count = sum(1 for v in results.values() if v)
            print(f"\n✓ Successfully synced {success_count}/{len(results)} models.")
            
            if success_count < len(results):
                print("\nFailed models:")
                for model_id, success in results.items():
                    if not success:
                        print(f"  - {model_id}")
    
    def stats(self, args):
        """Show database statistics."""
        stats = self.manager.get_stats()
        db_info = self.db.get_db_info()
        
        print("\nDatabase Statistics")
        print("=" * 40)
        print(f"Database Path: {db_info['db_path']}")
        print(f"Schema Version: {db_info['schema_version']}")
        print(f"Total Models: {stats['total_models']}")
        
        print("\nModels by Type:")
        for model_type, count in stats['by_type'].items():
            print(f"  {model_type}: {count}")
        
        print("\nModels by Attention:")
        for attention_type, count in stats['by_attention'].items():
            print(f"  {attention_type}: {count}")
        
        print("\nModels by Source:")
        for source, count in stats['by_source'].items():
            print(f"  {source}: {count}")
        
        print("\nParameter Ranges:")
        ranges = stats['parameter_ranges']
        print(f"  Min: {format_number(ranges['min'])}")
        print(f"  Max: {format_number(ranges['max'])}")
        print(f"  Avg: {format_number(int(ranges['avg']) if ranges['avg'] else None)}")
        
        print("\nTable Row Counts:")
        for table, count in db_info['table_counts'].items():
            print(f"  {table}: {count}")
    
    def clear_cache(self, args):
        """Clear model cache."""
        if args.model_id:
            cleared = self.manager.clear_cache(args.model_id)
            print(f"✓ Cleared {cleared} cache entries for model '{args.model_id}'.")
        else:
            cleared = self.manager.clear_cache()
            print(f"✓ Cleared {cleared} expired cache entries.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='GenZ Model Database Management CLI'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List models')
    list_parser.add_argument('--type', help='Filter by model type')
    list_parser.add_argument('--attention', help='Filter by attention type')
    list_parser.add_argument('--source', help='Filter by source')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show model details')
    show_parser.add_argument('model_id', help='Model ID')
    show_parser.add_argument('--config', dest='show_config', action='store_true',
                           help='Show full configuration')
    show_parser.add_argument('--cache', dest='show_cache', action='store_true',
                           help='Show cached values')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import model from HuggingFace')
    import_parser.add_argument('model_id', help='HuggingFace model ID')
    import_parser.add_argument('--force', action='store_true',
                             help='Force update if model exists')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a model')
    delete_parser.add_argument('model_id', help='Model ID')
    delete_parser.add_argument('--hard', action='store_true',
                             help='Permanently delete (default is soft delete)')
    delete_parser.add_argument('-y', '--yes', action='store_true',
                             help='Skip confirmation')
    
    # Sync command
    sync_parser = subparsers.add_parser('sync', help='Sync models')
    sync_parser.add_argument('--static', action='store_true',
                           help='Sync static models to database')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    
    # Clear cache command
    cache_parser = subparsers.add_parser('clear-cache', help='Clear model cache')
    cache_parser.add_argument('model_id', nargs='?', help='Model ID (clears all if not specified)')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create CLI instance
    cli = ModelCLI()
    
    # Execute command
    command_map = {
        'list': cli.list_models,
        'show': cli.show_model,
        'import': cli.import_model,
        'delete': cli.delete_model,
        'sync': cli.sync_models,
        'stats': cli.stats,
        'clear-cache': cli.clear_cache,
    }
    
    try:
        command_map[args.command](args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 