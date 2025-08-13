# News Research Tool 📉

<img width="1917" height="1014" alt="nr1" src="https://github.com/user-attachments/assets/9509bdf6-3ce7-4ae3-91a3-61f3fb541b1f" />

<img width="896" height="448" alt="nr2" src="https://github.com/user-attachments/assets/43a8135b-2049-45da-9530-d187ec0d04c7" />

<img width="872" height="492" alt="nr3" src="https://github.com/user-attachments/assets/aa79bc76-2dfb-4f6b-ba4d-39f1af07b217" />


A Streamlit-powered AI tool that helps mutual fund and equity research analysts quickly extract insights from multiple news articles. Using OpenAI embeddings and FAISS, this tool indexes content, allowing users to ask questions and get instant, sourced answers.

## Features

- Process multiple news article URLs at once
- Automatically split and embed article text using OpenAI embeddings
- Store embeddings in FAISS vector database for fast retrieval
- Query the vector store to get AI-generated answers with sources
- Intuitive Streamlit interface with gradient headings and dark theme for readability

## Tech Stack

- **Streamlit**: Frontend interface
- **LangChain**: LLM & chain management
- **OpenAI**: LLM & embeddings
- **FAISS**: Vector database for fast similarity search
- **Python**: Backend scripting
- **dotenv**: Environment variable management

## Installation

```bash
git clone https://github.com/yourusername/news-research-tool.git
cd news-research-tool
pip install -r requirements.txt
