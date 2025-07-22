#!/bin/bash

echo "Setting up Resume Bias Detection Project..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

echo "✅ pip3 found: $(pip3 --version)"

# Install dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# OpenAI API Configuration
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Override default model (uncomment to use)
# OPENAI_MODEL=gpt-3.5-turbo

# Optional: Set custom processing parameters (uncomment to override defaults)
# BATCH_SIZE=5
# DELAY_BETWEEN_REQUESTS=2
# MAX_RETRIES=5
EOF
    echo "✅ Created .env file"
    echo "⚠️  Please edit .env file and add your OpenAI API key"
else
    echo "✅ .env file already exists"
fi

# Create results directory
mkdir -p results
echo "✅ Created results directory"

# Run tests
echo "Running system tests..."
python3 test_system.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Installation completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Edit .env file and add your OpenAI API key"
    echo "2. Run: python3 main.py --step full --num-resumes 10"
    echo ""
else
    echo ""
    echo "⚠️  Installation completed but some tests failed."
    echo "Please check the error messages above."
    echo ""
fi 