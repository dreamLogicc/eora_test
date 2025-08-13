import asyncio
import io
import os
import sys

from utils.links import LINKS
from utils.parser import parse_links
from vec_db.vec_db import generate_vecdb, connect_to_vecdb, chunks_from_md, get_context
from loguru import  logger
from config import CHROMA_PATH, COLLECTION_NAME, GIGACHAT_CLIENT_SECRET
from api_utils.gigachat_api_utils import get_answer, get_token

sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')


async def initialize_db():
    if not os.path.exists(CHROMA_PATH):
        logger.info('Сбор документов...')
        documents = chunks_from_md(await parse_links(LINKS))
        return generate_vecdb(CHROMA_PATH, COLLECTION_NAME, documents)
    return connect_to_vecdb(CHROMA_PATH, COLLECTION_NAME)

db = asyncio.run(initialize_db())

async def main():
    print("\n" + "=" * 50)
    print("RAG QA System (для выхода введите 'quit' или 'exit')")
    print("=" * 50 + "\n")

    while True:
        query = input("[Вопрос] ")

        if query.lower() in ('quit', 'exit', 'выход'):
            print("Завершение работы системы...")
            break

        if not query:
            print("Пожалуйста, введите вопрос.")
            continue

        context = get_context(db, query)
        prompt = f"""
            Ты — представитель компании EORA. Отвечай на вопросы от первого лица множественного числа: 
            "Мы разработали...", "Наша команда внедрила...", "В нашем проекте...".  

            Контекст для ответа:  
            {context}  

            Вопрос: {query}  
            Ответ должен быть кратким и содержать ссылки на источники в формате [номер].
            Пример:

            Вопрос: "Какие технологии вы использовали в проекте для Lamoda?"
            Ответ:
            "Мы разработали систему поиска похожей одежды на основе трансформенных моделей для изображений и API для поиска [1].
            Наша команда также создала удобную систему разметки, которую можно применять в других проектах [1]."
            Источники:
            [1] - url
            """

        token = await get_token(GIGACHAT_CLIENT_SECRET)
        print("\n[Ответ] ", await get_answer(prompt, token))
        print("-" * 50 + "\n")



if __name__ == "__main__":
    asyncio.run(main())

