import asyncio
import io
import sys

from parser.links import LINKS
from vec_db.utils import get_context
from vec_db.vec_db import initialize_db
from config import CHROMA_PATH, COLLECTION_NAME, GIGACHAT_CLIENT_SECRET, JSON_PATH
from api_utils.gigachat_api_utils import get_answer, get_token

sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')

db = asyncio.run(initialize_db(CHROMA_PATH, COLLECTION_NAME, JSON_PATH, LINKS))

async def main():
    print("\n" + "=" * 50)
    print("RAG QA System (для выхода введите 'quit' или 'exit')")
    print("=" * 50 + "\n")

    while True:
        query = input("[Вопрос] ").strip()

        if query.lower() in ('quit', 'exit', 'выход'):
            print("Завершение работы...")
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
        try:
            token = await get_token(GIGACHAT_CLIENT_SECRET)
            print("\n[Ответ] ", await get_answer(prompt, token))
        except Exception as ex:
            print('Ошибка: ', ex)
        print("-" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
