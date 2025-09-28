import logging,tempfile,re,os,docx
from aiogram import Bot,Dispatcher,executor,types
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from io import BytesIO

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
_selected_model = None

client=Groq(api_key=api_key)

#Initiliaze bot & dispatcher
bot = Bot(token=TG_BOT_TOKEN)
dp = Dispatcher(bot)

from collections import defaultdict

# Per-user state dicts
user_sessions = defaultdict(lambda: {
    "doc_vectorstore": None,
    "user_file_content": None,
    "ref": {"messages": []}
})

def clear_convo(user_id):
    user_sessions[user_id]["doc_vectorstore"] = None
    user_sessions[user_id]["user_file_content"] = None
    user_sessions[user_id]["ref"]["messages"] = []

def format_bold(text: str) -> str:
    """Convert **text** to <b>text</b> for Telegram HTML"""
    return re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)

def get_model_name():
    global _selected_model

    #If we've already found a working model, reuse it
    if _selected_model:
        print(f"Using cached model: {_selected_model}")
        return _selected_model
    
    models = [
        "llama-3.1-8b-instant",
        "llama-3.1-70b-instant",
        "mixtral-8x7b",
        "gemma2-9b-it",
        "deepseek-r1-distill-llama-70b",
        "qwen/qwen3-32b"
    ]
    for model in models:
        try:
            # TEST
            print(f"Trying model: {model}")
            test = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5
            )
            print(f"Selected model: {model}")
            _selected_model = model  
            return model  #Returns the first working model
        except Exception as e:
            print(f"Model {model} failed: {e}")
            continue

    raise RuntimeError("No available models found!")

@dp.message_handler(content_types=types.ContentType.DOCUMENT)
async def handle_document(message: types.Message):
    user_id = message.from_user.id
    session = user_sessions[user_id]
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB limit
    MAX_WORD_COUNT = 15000            # 15,000 words max

    file = await bot.get_file(message.document.file_id)
    # Check file size before downloading
    if file.file_size and file.file_size > MAX_FILE_SIZE:
        await message.reply(
            f"PDF/DOC too big to process (>{MAX_FILE_SIZE // 1024 // 1024} MB). Please upload a smaller file."
        )
        return
    if session["doc_vectorstore"] is not None:
        await message.reply("⚠️ I can only parse one document at a time. Please clear conversation or wait before sending another.")
        return

    file_path = file.file_path
    file_extension = message.document.file_name.split('.')[-1].lower()
    downloaded_file = await bot.download_file(file_path)

    docs=[]
    if file_extension == "pdf":
        # ✅ FIX: Save BytesIO to a temporary file for PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(downloaded_file.read())
            temp_file_path = temp_file.name
        reader = PyPDFLoader(temp_file_path)
        docs = reader.load()

    elif file_extension in ["docx", "doc"]:
        file_stream = BytesIO(downloaded_file.read())
        doc = docx.Document(file_stream)
        docs = [Document(page_content=para.text) for para in doc.paragraphs if para.text.strip()]
    else:
        await message.reply("⚠️ Unsupported file type. Only PDF or DOC/DOCX allowed.")
        return
    
    if not docs:
        await message.reply("⚠️ Could not extract any text from the document.")
        return
    
    # Combine all text for word count check
    full_text = "\n".join([d.page_content for d in docs])
    word_count = len(full_text.split())

    if word_count > MAX_WORD_COUNT:
        await message.reply(
            f"Document too long to process ({word_count} words, max {MAX_WORD_COUNT}). Please upload a shorter file."
        )
        return
    
    # Store raw content for reference
    session["user_file_content"] = full_text

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  # You can replace with OpenAI embeddings if preferred
    session["doc_vectorstore"] = FAISS.from_documents(chunks, embeddings)

    await message.reply(f"Document received and processed successfully! You can now ask questions related to this document.")

@dp.message_handler(commands=['clear'])
async def clear(message: types.Message):
    user_id = message.from_user.id
    clear_convo(user_id)
    await message.reply("Your session and uploaded document are cleared. Start fresh!😊")

@dp.message_handler(commands=['start'])
async def welcome(message: types.Message):
    """
    This handler receives msg with '/start' command and responds with a wlcm msg
    """
    await message.reply("**Welcome to RAG-Groq Bot!**\n\n"
        "Upload a PDF or DOCX file, then ask questions about it.\n"
        "Type /clear to reset.\nType /help for commands.",
        parse_mode="Markdown")

@dp.message_handler(commands=['help'])
async def help(message: types.Message):
    """
    This handler receives msg with '/help' command and responds with help menu
    """
    cmd = """
    **Help Menu**
    Here are the commands you can use:
    **/start** – Start an interaction and get a welcome message.
    **/clear** – Clear conversation memory with uploaded docs & start fresh.
    **/help** – Show this help menu.
    💡 Tip: Just send me a message, and I'll reply using AI!
    Hope this helped you 😊
    """
    await message.reply(cmd,parse_mode="Markdown")

@dp.message_handler()
async def aibot(message: types.Message):
    """
    A handler to process user's input and generate a response using Groq API
    """
    print(f">>> USER: \n\t{message.text}")
    user_id = message.from_user.id
    session = user_sessions[user_id]
    user_question = message.text
    # Build prompt for Groq using full memory
    prompt_messages = session["ref"]["messages"].copy()
    prompt_messages.append({"role": "user", "content": user_question})
    if session["doc_vectorstore"]:
        # RAG: answer based on uploaded document
        retrieved_docs = session["doc_vectorstore"].similarity_search(user_question, k=3)
        context_text = "\n".join([d.page_content for d in retrieved_docs])
        prompt_messages[-1]["content"] = (
            f"Answer the question using ONLY the context below.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {user_question}\nAnswer:"
        )
    model = get_model_name()
    response=client.chat.completions.create(
        model=model,
        temperature=0.3,
        top_p=0.4,
        messages=prompt_messages)
    answer = response.choices[0].message.content
    # Save conversation for memory
    session["ref"]["messages"].append({"role": "user", "content": user_question})
    session["ref"]["messages"].append({"role": "assistant", "content": answer})

    # Limit memory to last MAX_MEMORY messages
    MAX_MEMORY = 5
    session["ref"]["messages"] = session["ref"]["messages"][-MAX_MEMORY:]

    formatted_answer = format_bold(answer)
    await bot.send_message(chat_id=message.chat.id,text=formatted_answer,parse_mode="HTML")
    print(f">>> GROQ: \n\t{answer}")

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    print("LIVEBOT is running...")
    executor.start_polling(dp,skip_updates=False)

