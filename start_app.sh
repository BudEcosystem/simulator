#!/bin/bash

echo "🚀 Starting BudSimulator Application"
echo "=================================="

# Start Backend
echo "📦 Starting Backend API Server..."
cd /Users/accubits/Desktop/bud/budsim/simulator/BudSimulator
source ../venv/bin/activate
python run_api.py &

echo "⏳ Waiting for backend to start..."
sleep 5

# Start Frontend  
echo "🎨 Starting Frontend Development Server..."
cd frontend
npm start

# This script will run the frontend in the foreground
# The backend runs in the background
# Press Ctrl+C to stop the frontend, then manually kill the backend if needed