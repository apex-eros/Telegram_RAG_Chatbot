import logging
from aiogram import Bot,Dispatcher,executor,types
from dotenv import load_dotenv
import os

load_dotenv()
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")

#Config logging
logging.basicConfig(level=logging.INFO)

#Initiliaze bot and dispatcher
bot=Bot(token=TG_BOT_TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start','help'])
async def start_handler(message: types.Message):
    """
    This handler receives msg with '/start' or '/help' command
    """
    await message.reply("Hi\nI am your LIVEBOT.How you feeling today?")

@dp.message_handler()
async def repeat(message: types.Message):
    """
    This will return repeat
    """
    await message.answer(message.text)

if __name__=='__main__':
    executor.start_polling(dp,skip_updates=True)
