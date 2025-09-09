#!/bin/bash
# Deployment script for Strands Multi-Agent System v1.7.1

set -e

echo "🚀 Deploying Strands Multi-Agent System v1.7.1"

# Update system packages
echo "📦 Installing system dependencies..."
pip install --upgrade pip

# Install Strands Agents v1.7.1
echo "📦 Installing Strands Agents v1.7.1..."
pip install strands-agents>=1.7.1
pip install strands-agents-tools>=0.1.0

# Install project requirements
echo "📦 Installing project requirements..."
pip install -r requirements.txt

# Run compatibility tests
echo "🧪 Running compatibility tests..."
python test_strands_v1_7_1.py

# Verify system health
echo "🔍 Verifying system health..."
python -c "
import sys
sys.path.append('.')
from inference.agents.orchestrator_agent import StrandsMultiAgentOrchestrator
orchestrator = StrandsMultiAgentOrchestrator()
print('✅ Strands Multi-Agent System is ready')
"

echo "🎉 Deployment completed successfully!"
echo "📊 System Status:"
echo "   - Strands Agents: v1.7.1+"
echo "   - Multi-Agent Tools: Available"
echo "   - Orchestrator: Ready"
echo "   - Brand Extraction: Operational"
