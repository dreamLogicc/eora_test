import asyncio

from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from aiogram.filters import Command
from loguru import logger
from config import BOT_TOKEN, CHROMA_PATH, COLLECTION_NAME, GIGACHAT_CLIENT_SECRET,JSON_PATH
from api_utils.gigachat_api_utils import get_token, get_answer
from parser.links import LINKS
from vec_db.vec_db import get_context, initialize_db

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

db = asyncio.run(initialize_db(CHROMA_PATH, COLLECTION_NAME, JSON_PATH, LINKS))


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    welcome_message = (
        "👋 <b>Привет</b>"
        "Я — виртуальный помощник компании <b>EORA</b>. \n\n"
        "Готов ответить на ваши вопросы."
    )
    await message.answer(welcome_message, parse_mode=ParseMode.HTML)


@dp.message(F.text)
async def handle_text(message: types.Message):

    context = get_context(db, message.text)
    prompt = f"""
            Ты — представитель компании EORA. Отвечай на вопросы от первого лица множественного числа: 
            "Мы разработали...", "Наша команда внедрила...", "В нашем проекте...".  

            Контекст для ответа:  
            {context}  

            Вопрос: {message.text}  
            Ответ должен быть кратким и содержать ссылки на источники в формате [номер](url).
            Пример:

            Вопрос: "Какие технологии вы использовали в проекте для Lamoda?"
            Ответ:
            "Мы разработали систему поиска похожей одежды на основе трансформенных моделей для изображений и API для поиска [1](url).
            Наша команда также создала удобную систему разметки, которую можно применять в других проектах [1](url)."
    """
    try:
        token = await get_token(GIGACHAT_CLIENT_SECRET)
        answer =  await get_answer(prompt, token)
        await message.answer(answer, parse_mode=ParseMode.MARKDOWN)
    except Exception as ex:
        print('Ошибка: ', ex)
        await message.answer('Что-то пошло не так(')


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    logger.success('Бот запущен')
    asyncio.run(main())
