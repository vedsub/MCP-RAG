# MCP-RAG

A Multi-Modal RAG (Retrieval-Augmented Generation) system using Model Context Protocol (MCP) to integrate various data sources and analysis tools.

## Features

- **Multimodal Analysis**: Process video, audio, image, and document files.
- **Web Search**: Integrated Tavily search for real-time information.
- **ArXiv Research**: Search and retrieve academic papers from ArXiv.
- **Agentic Workflow**: Sequential processing of media files with synthesis capabilities.

## Prerequisites

- Python 3.10+
- `uv` (recommended for dependency management) or `pip`
- API keys for:
    - OpenAI/Groq (LLM)
    - Tavily (Web Search)
    - AssemblyAI (Audio processing - optional)
    - Gemini (Multimodal analysis - optional)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd MCP-RAG
   ```

2. Install dependencies:
   ```bash
   uv sync
   # OR
   pip install -r pyproject.toml
   ```

3. Configure environment variables:
   Copy `.env.example` to `.env` and fill in your API keys.
   ```bash
   cp .env.example .env
   ```

## Configuration

The system is configured via `config.py` and environment variables.
Key configurations include:
- **LLM**: Defaults to Groq (`llama-3.1-8b-instant`).
- **Embeddings**: Uses `all-MiniLM-L6-v2` locally.
- **Vector DB**: ChromaDB for storage.
- **MCP Servers**: Configurable URLs for Web, ArXiv, and Multimodal servers.

## Usage

### Running MCP Servers

Start the web research server:
```bash
python mcp_server/web_server.py
```

Start the ArXiv research server:
```bash
python mcp_server/arxiv_server.py
```

### Running the Agent

*Note: Entry points `main.py` and `app.py` are currently placeholders. Access the agent logic via `agents/multimodal_agent.py`.*

To use the `MultiModalResearchAgent`:

```python
import asyncio
from agents.multimodal_agent import MultiModalResearchAgent

async def main():
    agent = await MultiModalResearchAgent.create()
    result = await agent.research("Your research query here")
    print(result.synthesis)

if __name__ == "__main__":
    asyncio.run(main())
```

## Project Structure

- `agents/`: Core agent logic (e.g., `multimodal_agent.py`).
- `mcp_server/`: MCP server implementations (Web, ArXiv).
- `mcp_client/`: Client for connecting to MCP servers.
- `util/`: Utility functions (logger, etc.).
- `data/`: Directory for storing/processing data.
