# Startup_Trend_Analysis_RAG
An AI-powered startup trend analysis tool using RAG (Retrieval-Augmented Generation) with a Streamlit interface.

## Features

- 🔍 **RAG-powered Research**: Uses ChromaDB for vector search and retrieval
- 🤖 **Multi-LLM Support**: Works with OpenAI GPT and Anthropic Claude
- 📊 **Interactive Dashboard**: Beautiful Streamlit interface
- 📈 **Keyword Analysis**: Automatic keyword extraction and visualization
- 📚 **Source Tracking**: Shows relevance and sources for all insights

## Setup

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Prepare your corpus**:
   - Create a `./corpus` directory
   - Add your research documents (PDF, TXT, MD, JSON, CSV files)

3. **Set up API keys**:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key from https://platform.openai.com/
   - (Optional) Add your Anthropic API key from https://console.anthropic.com/

## Running the App

Start the Streamlit application:

```bash
streamlit run app.py
```

Or directly:

```bash
streamlit run main.py
```

## Usage

1. **Auto-Initialize**: Pipeline automatically loads from your .env file
2. **Ask**: Enter your startup trend question
3. **Analyze**: Click "Analyze" to get AI-powered insights
4. **Explore**: Browse enhanced results across tabs:
   - 📊 **Executive Summary**: Action-oriented business analysis
   - 🔍 **Research Details**: Comprehensive research findings
   - 📈 **Trend Analysis**: Interactive keyword charts and metrics
   - 📚 **RAG Sources**: Source relevance analysis with expandable content

## Example Questions

- "What are the most investable AI-native B2B startup trends for 2025?"
- "Which emerging technologies show the strongest product-market fit signals?"
- "What regulatory changes will impact fintech startups in 2025?"

## Configuration

The pipeline supports various configuration options in the `Config` class:

- **Models**: Choose between OpenAI GPT-4 and Anthropic Claude
- **Chunk Size**: Adjust document chunking for better context
- **Search Results**: Configure number of retrieved documents
- **Temperature**: Control AI response creativity

## Architecture

- **Document Processing**: Chunked text processing with overlap
- **Vector Store**: ChromaDB with OpenAI embeddings
- **LLM Agents**: Research and Analysis agents using LangGraph
- **Interface**: Streamlit web application

## Requirements

- Python 3.13+
- OpenAI API key (required)
- Anthropic API key (optional)
- Documents in `./corpus` directory
