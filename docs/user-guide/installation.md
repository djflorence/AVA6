# Installation Guide

This guide provides detailed instructions for installing AVA6 in different environments.

## System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: Version 3.10 or higher
- **RAM**: Minimum 4GB, 8GB+ recommended
- **Storage**: At least 1GB of free disk space
- **API Keys**: Access to OpenAI API or other supported LLM providers

## Standard Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/djflorence/AVA6.git
cd AVA6
```

### Step 2: Set Up a Virtual Environment (Recommended)

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

```bash
cp .env.example .env
```

Edit the `.env` file with your preferred text editor and add your API keys and other configuration options.

## Docker Installation

AVA6 can also be run using Docker for easier deployment and isolation.

### Step 1: Build the Docker Image

```bash
docker build -t ava6 .
```

### Step 2: Run the Container

```bash
docker run -it --env-file .env -v ./data:/app/data ava6
```

## Installation for Development

If you plan to contribute to AVA6 or customize it extensively, follow these additional steps:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install
```

## Troubleshooting

### Common Issues

#### Missing Dependencies
If you encounter errors about missing packages, try updating pip and reinstalling:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### API Key Issues
If you see authentication errors, check that your API keys in the `.env` file are correct and have sufficient permissions.

#### Memory Issues
If the application crashes due to memory constraints, try:
- Reducing the model context size in the configuration
- Using a smaller model
- Increasing your system's swap space

## Next Steps

After installation, proceed to the [Configuration Guide](configuration.md) to set up AVA6 according to your needs. 