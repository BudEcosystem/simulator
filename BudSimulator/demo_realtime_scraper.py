#!/usr/bin/env python3
"""
Demonstration script for the real-time HuggingFace scraper.
This script shows how the scraper saves JSON data in real-time after each page.
"""

import sys
import os
from pathlib import Path
import time
import os.path

# Add the model_finder directory to the path
sys.path.insert(0, str(Path(__file__).parent / "model_finder"))

from hf_model_list_scraper import HuggingFaceSeleniumScraper


def demo_realtime_scraping():
    """Demonstrate real-time scraping with 3 pages"""
    print("🚀 Real-Time HuggingFace Scraper Demo")
    print("=" * 45)
    print("This demo will scrape 3 pages and show real-time JSON updates")
    print()
    
    # Create scraper instance
    scraper = HuggingFaceSeleniumScraper(headless=True)
    
    # Check ChromeDriver first
    if not scraper.check_chromedriver():
        print("❌ ChromeDriver not available. Please install it first.")
        return False
    
    try:
        # Show session file information
        session_files = scraper.get_current_session_files()
        print(f"📁 Session Files Created:")
        print(f"   🔄 Real-time JSON: {os.path.basename(session_files['json_file'])}")
        print(f"   📝 Progress log: {os.path.basename(session_files['progress_file'])}")
        print(f"   📂 Data directory: {session_files['data_dir']}")
        print()
        
        print("🔍 Starting scraping (3 pages)...")
        print("💡 Watch the data directory for real-time updates!")
        print()
        
        # Scrape 3 pages to demonstrate real-time updates
        models = scraper.scrape_all_models(max_pages=3, scroll_each_page=False)
        
        if models:
            print(f"\n🎉 Demo completed!")
            print(f"📊 Total models scraped: {len(models)}")
            
            # Show file sizes and modification times
            json_file = session_files['json_file']
            progress_file = session_files['progress_file']
            
            if os.path.exists(json_file):
                json_size = os.path.getsize(json_file)
                json_mtime = os.path.getmtime(json_file)
                print(f"\n📄 Real-time JSON file:")
                print(f"   Size: {json_size:,} bytes")
                print(f"   Last modified: {time.ctime(json_mtime)}")
            
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    progress_lines = f.readlines()
                print(f"\n📝 Progress log ({len(progress_lines)} lines):")
                for line in progress_lines[-5:]:  # Show last 5 lines
                    print(f"   {line.strip()}")
            
            # Show modality statistics
            print(f"\n📊 Modality Distribution:")
            modality_counts = {}
            for model in models:
                modality = model.get('modality', 'Unknown')
                modality_counts[modality] = modality_counts.get(modality, 0) + 1
            
            for modality, count in sorted(modality_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   {modality}: {count} models")
            
            return True
            
        else:
            print("❌ No models were scraped.")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
        session_files = scraper.get_current_session_files()
        if scraper.models:
            print(f"💾 Partial results saved in: {os.path.basename(session_files['json_file'])}")
        return False
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        return False
    
    finally:
        scraper.close_driver()


def monitor_realtime_files():
    """Monitor the data directory for real-time updates"""
    data_dir = Path(__file__).parent / "model_finder" / "data"
    
    print(f"\n👀 Monitoring {data_dir} for real-time files...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        last_files = set()
        while True:
            current_files = set(f.name for f in data_dir.glob("*") if f.is_file())
            
            # Check for new files
            new_files = current_files - last_files
            if new_files:
                for file in new_files:
                    file_path = data_dir / file
                    size = file_path.stat().st_size
                    print(f"📄 New file: {file} ({size:,} bytes)")
            
            # Check for updated files
            for file in current_files:
                file_path = data_dir / file
                if file_path.suffix == '.json':
                    size = file_path.stat().st_size
                    mtime = file_path.stat().st_mtime
                    print(f"🔄 Updated: {file} ({size:,} bytes, {time.strftime('%H:%M:%S', time.localtime(mtime))})")
            
            last_files = current_files
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped")


if __name__ == "__main__":
    print("HuggingFace Scraper Real-Time Demo")
    print("="*40)
    
    choice = input("\nChoose an option:\n1. Run scraping demo (3 pages)\n2. Monitor data directory\n\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        success = demo_realtime_scraping()
        if success:
            print("\n✅ Demo completed successfully!")
            print("Check the model_finder/data/ directory for the generated files.")
        else:
            print("\n❌ Demo failed.")
    elif choice == "2":
        monitor_realtime_files()
    else:
        print("Invalid choice. Please run again and select 1 or 2.") 