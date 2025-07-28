#!/usr/bin/env python3
"""
Automatic build refresh system for GenZ development.
Watches for changes in GenZ source files and automatically reinstalls the package.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class GenZWatcher(FileSystemEventHandler):
    """Watches GenZ source files and triggers rebuild on changes."""
    
    def __init__(self, project_root):
        self.project_root = project_root
        self.last_rebuild = 0
        self.rebuild_delay = 1.0  # Minimum seconds between rebuilds
        
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
            
        # Only watch Python files and pyproject.toml
        if event.src_path.endswith(('.py', 'pyproject.toml')):
            current_time = time.time()
            if current_time - self.last_rebuild > self.rebuild_delay:
                self.rebuild()
                self.last_rebuild = current_time
    
    def on_created(self, event):
        """Handle file creation events."""
        self.on_modified(event)
    
    def rebuild(self):
        """Reinstall the GenZ package in development mode."""
        print("\n🔄 Detected changes, rebuilding GenZ...")
        
        try:
            # Run pip install in editable mode
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-e', self.project_root],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                print("✅ GenZ rebuilt successfully!")
            else:
                print(f"❌ Build failed:\n{result.stderr}")
                
        except Exception as e:
            print(f"❌ Error during rebuild: {e}")

def main():
    """Main entry point for the watcher."""
    project_root = Path(__file__).parent
    
    print(f"👀 Watching GenZ source files in: {project_root}")
    print("📦 Package will be automatically rebuilt on changes")
    print("🛑 Press Ctrl+C to stop\n")
    
    # Directories to watch
    watch_dirs = [
        project_root / 'GenZ',
        project_root / 'Systems',
    ]
    
    # Create event handler and observer
    event_handler = GenZWatcher(str(project_root))
    observer = Observer()
    
    # Add watchers for each directory
    for directory in watch_dirs:
        if directory.exists():
            observer.schedule(event_handler, str(directory), recursive=True)
            print(f"📁 Watching: {directory}")
    
    # Also watch pyproject.toml
    observer.schedule(event_handler, str(project_root), recursive=False)
    
    # Start watching
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n\n🛑 Watcher stopped.")
    
    observer.join()

if __name__ == '__main__':
    main()