#!/usr/bin/env python3
"""
BudSimulator Automated Setup Script

This script automates the entire setup process for BudSimulator including:
- Environment setup
- Dependency installation
- Database initialization
- Configuration management
- System testing
- Server launch

Usage: python setup.py
"""

import os
import sys
import json
import shutil
import subprocess
import platform
import time
import socket
import webbrowser
from pathlib import Path
from typing import Dict, Any, Optional

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class BudSimulatorSetup:
    """Automated setup for BudSimulator."""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.config = {}
        self.processes = []
        
    def print_header(self, text: str):
        """Print a formatted header."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")
        
    def print_step(self, text: str):
        """Print a step message."""
        print(f"{Colors.CYAN}➤ {text}{Colors.ENDC}")
        
    def print_success(self, text: str):
        """Print a success message."""
        print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")
        
    def print_error(self, text: str):
        """Print an error message."""
        print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")
        
    def print_warning(self, text: str):
        """Print a warning message."""
        print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")
        
    def check_python_version(self):
        """Check if Python version is compatible."""
        self.print_step("Checking Python version...")
        
        if sys.version_info < (3, 8):
            self.print_error("Python 3.8 or higher is required!")
            sys.exit(1)
            
        self.print_success(f"Python {sys.version.split()[0]} detected")
        
    def check_node_npm(self):
        """Check if Node.js and npm are installed."""
        self.print_step("Checking Node.js and npm...")
        
        try:
            node_version = subprocess.check_output(['node', '--version'], text=True).strip()
            npm_version = subprocess.check_output(['npm', '--version'], text=True).strip()
            
            self.print_success(f"Node.js {node_version} and npm {npm_version} detected")
            return True
        except:
            self.print_error("Node.js and npm are required but not found!")
            self.print_warning("Please install Node.js from https://nodejs.org/")
            return False
            
    def create_virtual_environment(self):
        """Create and activate virtual environment."""
        self.print_step("Setting up Python virtual environment...")
        
        venv_path = self.root_dir / "env"
        
        if not venv_path.exists():
            subprocess.run([sys.executable, "-m", "venv", "env"], cwd=self.root_dir)
            self.print_success("Virtual environment created")
        else:
            self.print_success("Virtual environment already exists")
            
        # Get activation command based on OS
        if platform.system() == "Windows":
            activate_cmd = str(venv_path / "Scripts" / "activate.bat")
            pip_path = str(venv_path / "Scripts" / "pip")
            python_path = str(venv_path / "Scripts" / "python")
        else:
            activate_cmd = f"source {venv_path / 'bin' / 'activate'}"
            pip_path = str(venv_path / "bin" / "pip")
            python_path = str(venv_path / "bin" / "python")
            
        return pip_path, python_path, activate_cmd
        
    def install_python_dependencies(self, pip_path: str):
        """Install Python dependencies."""
        self.print_step("Installing Python dependencies...")
        
        requirements_file = self.root_dir / "requirements.txt"
        
        # Create requirements.txt if it doesn't exist
        if not requirements_file.exists():
            requirements = [
                "fastapi>=0.104.1",
                "uvicorn>=0.24.0",
                "pydantic>=2.5.0",
                "python-multipart>=0.0.6",
                "aiofiles>=23.2.1",
                "httpx>=0.25.2",
                "beautifulsoup4>=4.12.2",
                "requests>=2.31.0",
                "Pillow>=10.1.0",
                "numpy>=1.24.0",
                "pytest>=7.4.3",
                "pytest-asyncio>=0.21.1",
                "python-dotenv>=1.0.0",
            ]
            
            with open(requirements_file, 'w') as f:
                f.write('\n'.join(requirements))
                
            self.print_success("Created requirements.txt")
        
        # Install dependencies
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], cwd=self.root_dir)
        self.print_success("Python dependencies installed")
        
    def install_frontend_dependencies(self):
        """Install frontend npm dependencies."""
        self.print_step("Installing frontend dependencies...")
        
        frontend_dir = self.root_dir / "frontend"
        
        if frontend_dir.exists():
            subprocess.run(["npm", "install"], cwd=frontend_dir)
            self.print_success("Frontend dependencies installed")
        else:
            self.print_warning("Frontend directory not found, skipping npm install")
            
    def setup_database(self, python_path: str):
        """Set up database with pre-populated data."""
        self.print_step("Setting up database...")
        
        # Create local data directory
        db_dir = self.root_dir / "data"
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if we have a pre-populated database
        prepop_db = self.root_dir / "data" / "prepopulated.db"
        
        if prepop_db.exists():
            # Database already exists
            self.print_success("Pre-populated database found")
        else:
            # Run database setup script
            setup_script = self.root_dir / "scripts" / "setup_database.py"
            
            if setup_script.exists():
                self.print_warning("Running database setup (this may take a while)...")
                subprocess.run([python_path, str(setup_script), "--skip-analysis"], cwd=self.root_dir)
                self.print_success("Database initialized")
            else:
                self.print_warning("Database setup script not found")
                
    def configure_llm(self):
        """Configure LLM settings."""
        self.print_header("LLM Configuration")
        
        print("Please provide your LLM configuration:")
        print("(Press Enter to use default values)\n")
        
        # Get LLM provider
        print("Available LLM providers:")
        print("1. OpenAI")
        print("2. Anthropic") 
        print("3. Local (Ollama)")
        print("4. Custom API")
        
        provider_choice = input("\nSelect provider (1-4) [1]: ").strip() or "1"
        
        provider_map = {
            "1": "openai",
            "2": "anthropic",
            "3": "ollama",
            "4": "custom"
        }
        
        provider = provider_map.get(provider_choice, "openai")
        
        # Get configuration based on provider
        if provider == "openai":
            api_key = input("OpenAI API Key: ").strip()
            model = input("Model name [gpt-4]: ").strip() or "gpt-4"
            api_url = "https://api.openai.com/v1/chat/completions"
            
        elif provider == "anthropic":
            api_key = input("Anthropic API Key: ").strip()
            model = input("Model name [claude-3-opus-20240229]: ").strip() or "claude-3-opus-20240229"
            api_url = "https://api.anthropic.com/v1/messages"
            
        elif provider == "ollama":
            api_key = ""
            model = input("Model name [llama2]: ").strip() or "llama2"
            api_url = input("Ollama URL [http://localhost:11434]: ").strip() or "http://localhost:11434"
            
        else:
            api_url = input("API URL: ").strip()
            api_key = input("API Key (if required): ").strip()
            model = input("Model name: ").strip()
        
        # Save configuration
        self.config = {
            "llm_provider": provider,
            "llm_api_key": api_key,
            "llm_model": model,
            "llm_api_url": api_url
        }
        
        # Create .env file
        env_file = self.root_dir / ".env"
        with open(env_file, 'w') as f:
            f.write(f"LLM_PROVIDER={provider}\n")
            f.write(f"LLM_API_KEY={api_key}\n")
            f.write(f"LLM_MODEL={model}\n")
            f.write(f"LLM_API_URL={api_url}\n")
            
        self.print_success("LLM configuration saved")
        
    def run_tests(self, python_path: str):
        """Run system tests."""
        self.print_step("Running system tests...")
        
        # Run pytest
        test_result = subprocess.run(
            [python_path, "-m", "pytest", "tests/", "-v", "--tb=short"],
            cwd=self.root_dir,
            capture_output=True,
            text=True
        )
        
        if test_result.returncode == 0:
            self.print_success("All tests passed!")
        else:
            self.print_warning("Some tests failed (this may be okay for initial setup)")
            
    def run_verification(self, python_path: str):
        """Run verification script."""
        self.print_step("Running system verification...")
        
        # Wait a bit for servers to fully start
        time.sleep(3)
        
        # Run verification script
        verify_result = subprocess.run(
            [python_path, "verify_setup.py"],
            cwd=self.root_dir
        )
        
        if verify_result.returncode == 0:
            self.print_success("System verification passed!")
        else:
            self.print_warning("Some verification checks failed (check output above)")
            
    def check_port_availability(self, port: int) -> bool:
        """Check if a port is available."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('', port))
            sock.close()
            return True
        except:
            return False
            
    def start_backend(self, python_path: str):
        """Start the backend API server."""
        self.print_step("Starting backend API server...")
        
        # Check if port 8000 is available
        if not self.check_port_availability(8000):
            self.print_warning("Port 8000 is already in use!")
            port = 8001
            while not self.check_port_availability(port):
                port += 1
            self.print_warning(f"Using port {port} instead")
        else:
            port = 8000
            
        # Set environment variable for backend port
        env = os.environ.copy()
        env['BACKEND_PORT'] = str(port)
        
        # Start API server using run_api.py
        backend_process = subprocess.Popen(
            [python_path, "run_api.py"],
            cwd=self.root_dir,
            env=env
        )
        
        self.processes.append(backend_process)
        self.config['backend_port'] = port
        
        # Wait for backend to start
        time.sleep(3)
        
        self.print_success(f"Backend API running on http://localhost:{port}")
        
    def start_frontend(self):
        """Start the frontend development server."""
        self.print_step("Starting frontend server...")
        
        frontend_dir = self.root_dir / "frontend"
        
        if not frontend_dir.exists():
            self.print_warning("Frontend directory not found!")
            return
            
        # Check if port 3000 is available
        if not self.check_port_availability(3000):
            self.print_warning("Port 3000 is already in use!")
            port = 3001
            while not self.check_port_availability(port):
                port += 1
            self.print_warning(f"Using port {port} instead")
        else:
            port = 3000
            
        # Update frontend API URL if backend is on different port
        if self.config.get('backend_port', 8000) != 8000:
            # Update API base URL in frontend
            api_config_file = frontend_dir / "src" / "config.js"
            if api_config_file.exists():
                with open(api_config_file, 'r') as f:
                    content = f.read()
                content = content.replace('localhost:8000', f'localhost:{self.config["backend_port"]}')
                with open(api_config_file, 'w') as f:
                    f.write(content)
                    
        # Set PORT environment variable
        env = os.environ.copy()
        env['PORT'] = str(port)
        
        # Start npm
        frontend_process = subprocess.Popen(
            ["npm", "start"],
            cwd=frontend_dir,
            env=env
        )
        
        self.processes.append(frontend_process)
        self.config['frontend_port'] = port
        
        # Wait for frontend to start
        time.sleep(5)
        
        self.print_success(f"Frontend running on http://localhost:{port}")
        
    def open_browser(self):
        """Open the application in browser."""
        url = f"http://localhost:{self.config.get('frontend_port', 3000)}"
        
        self.print_step(f"Opening browser at {url}")
        time.sleep(2)
        webbrowser.open(url)
        
    def print_final_instructions(self):
        """Print final setup instructions."""
        self.print_header("Setup Complete! 🎉")
        
        print(f"{Colors.GREEN}BudSimulator is now running!{Colors.ENDC}\n")
        
        print(f"{Colors.BOLD}Access Points:{Colors.ENDC}")
        print(f"  • Frontend: http://localhost:{self.config.get('frontend_port', 3000)}")
        print(f"  • Backend API: http://localhost:{self.config.get('backend_port', 8000)}")
        print(f"  • API Docs: http://localhost:{self.config.get('backend_port', 8000)}/docs\n")
        
        print(f"{Colors.BOLD}To stop the servers:{Colors.ENDC}")
        print("  Press Ctrl+C in this terminal\n")
        
        print(f"{Colors.BOLD}To restart later:{Colors.ENDC}")
        print("  1. Activate virtual environment:")
        if platform.system() == "Windows":
            print("     env\\Scripts\\activate")
        else:
            print("     source env/bin/activate")
        print("  2. Start backend:")
        print("     python run_api.py")
        print("  3. Start frontend (new terminal):")
        print("     cd frontend && npm start\n")
        
        print(f"{Colors.CYAN}Enjoy using BudSimulator!{Colors.ENDC}")
        
    def cleanup(self):
        """Clean up processes on exit."""
        self.print_warning("\nShutting down servers...")
        
        for process in self.processes:
            process.terminate()
            
        time.sleep(2)
        
        for process in self.processes:
            if process.poll() is None:
                process.kill()
                
        self.print_success("Servers stopped")
        
    def run(self):
        """Run the complete setup process."""
        try:
            self.print_header("BudSimulator Automated Setup")
            
            # Check prerequisites
            self.check_python_version()
            
            if not self.check_node_npm():
                return
                
            # Set up environment
            pip_path, python_path, activate_cmd = self.create_virtual_environment()
            
            # Install dependencies
            self.install_python_dependencies(pip_path)
            self.install_frontend_dependencies()
            
            # Set up database
            self.setup_database(python_path)
            
            # Configure LLM
            self.configure_llm()
            
            # Run tests
            self.run_tests(python_path)
            
            # Start servers
            self.start_backend(python_path)
            self.start_frontend()
            
            # Open browser
            self.open_browser()
            
            # Run verification
            self.run_verification(python_path)
            
            # Print instructions
            self.print_final_instructions()
            
            # Keep running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.cleanup()
                
        except Exception as e:
            self.print_error(f"Setup failed: {str(e)}")
            self.cleanup()
            sys.exit(1)


if __name__ == "__main__":
    setup = BudSimulatorSetup()
    setup.run() 