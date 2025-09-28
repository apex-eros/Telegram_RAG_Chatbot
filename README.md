# Telegram_RAG_Chatbot

A Telegram RAG chatbot powered by Groq LLMs and LangChain. Upload PDF/DOCX files and ask questions directly in Telegram with document-aware answers. Includes evaluation using ROUGE.

## Features
- Upload and query PDF/DOCX files
- RAG-powered chatbot with Groq LLMs + LangChain
- Seamless Telegram integration
- Response evaluation using ROUGE metrics

## Tech Stack
- **Python**
- **LangChain**
- **Groq LLMs**
- **Telegram Bot API**

## Setup
1. Clone the repo:
   ```bash
   git clone https://github.com/apex-eros/Telegram_RAG_Chatbot.git
   cd Telegram_RAG_Chatbot

2. Install dependencies:
pip install -r requirements.txt

3. Set environment variables in .env
TELEGRAM_BOT_TOKEN=your_telegram_token
GROQ_API_KEY=your_groq_api_key

4.Run the bot
python main.py

Usage

Start the bot in Telegram

Upload a PDF/DOCX file

Ask questions about the uploaded document

Get context-aware answers instantly

Evaluation

The chatbot’s responses are evaluated using ROUGE metrics for relevance and quality.
