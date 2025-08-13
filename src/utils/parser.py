import requests
import re

from langchain_core.documents import Document
from typing import List
from tqdm import tqdm
from bs4 import BeautifulSoup
from config import GIGACHAT_CLIENT_SECRET
from api_utils.gigachat_api_utils import get_answer, get_token

async def parse(url: str) -> str:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html5lib')

    for script in soup(["script", "style", "noscript", "meta", "link"]):
        script.decompose()

    raw_text = soup.get_text(separator=' ', strip=True)

    clean_text = re.sub(r'\[\{.*?\}\]', '', raw_text, flags=re.DOTALL)

    clean_text = '\n'.join([line.strip() for line in clean_text.split('\n') if line.strip()])

    prompt = f"""Анализируй предоставленный текст и выделяй только важные моменты, сохраняя смысловую структуру. Действуй по следующим правилам:
        1. Удаляй:
        - Все контактные данные (телефоны, email, адреса)
        - Технические блоки (JSON, HTML-теги, коды форм)
        - Уведомления о cookies и политиках конфиденциальности
        - Повторяющиеся элементы навигации и меню
        - Рекламные призывы ("Получить консультацию" и подобные)
        
        2. Сохраняй:
        - Заголовки и подзаголовки
        - Проблемы и решения
        - Технологии и инструменты
        - Ключевые преимущества
        - Команду проекта (если есть)
        - Уникальные особенности проекта
        
        3. Форматируй результат:
        - Используй Markdown-разметку
        - Заголовки выделяй через ###
        - Списки оформляй через дефисы
        - Сохраняй оригинальную структуру разделов
        - Удаляй лишние переносы строк
        
        4. Особые указания:
        - Если встречается описание кейса - сохрани полную структуру (Проблема/Решение/Результат)
        - Технические детали оставляй только если они существенны
        - Названия инструментов/технологий выделяй курсивом
        
        Пример структуры вывода:
        
        ### Проект: Название проекта/кейса
        
        ### Проблема
        - Описание проблемы
        - Кто сталкивается
        
        ### Решение
        - Какое решение предложено
        - Основные компоненты системы
        
        ### Технологии
        - *PyTorch* для машинного обучения
        - Собственный инструмент *ORI MarkUp*
        
        ### Команда
        - Имя (роль)
        
        
        Теперь обработай следующий текст:
        {clean_text}
    """

    token = await get_token(GIGACHAT_CLIENT_SECRET)
    response = await get_answer(prompt, token)

    return response


async def parse_links(links: List[str]) -> List[Document]:
    data = []
    for link in tqdm(links):
        data.append(
            {
                'text': await parse(link),
                'source': link
            }
        )
    return [Document(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(data)]
