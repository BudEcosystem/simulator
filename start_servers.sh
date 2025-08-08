#!/bin/bash

echo "Starting BudSimulator servers..."

# Kill any existing processes on ports 3000 and 8000
lsof -ti:3000 | xargs kill -9 2>/dev/null
lsof -ti:8000 | xargs kill -9 2>/dev/null

# Wait a moment for ports to be freed
sleep 2

# Start backend server
echo "Starting FastAPI backend on port 8000..."
cd /Users/accubits/Desktop/bud/budsim/simulator/BudSimulator
source ../venv/bin/activate
python run_api.py &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 5

# Check if backend is running
if curl -s http://localhost:8000/api/health > /dev/null; then
    echo "✅ Backend is running"
else
    echo "❌ Backend failed to start"
    exit 1
fi

# Start frontend server
echo "Starting React frontend on port 3000..."
cd /Users/accubits/Desktop/bud/budsim/simulator/BudSimulator/frontend
npm start &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo ""
echo "🚀 Both servers are starting..."
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for interrupt
trap 'echo "Stopping servers..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit' INT
wait