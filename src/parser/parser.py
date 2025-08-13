import json

import requests
import re

import aiohttp
from langchain_core.documents import Document
from typing import List
from tqdm import tqdm
from bs4 import BeautifulSoup
from config import GIGACHAT_CLIENT_SECRET, JSON_PATH
from api_utils.gigachat_api_utils import get_answer, get_token
from loguru import logger


async def query(url: str) -> str:
    """Отправляет асинхронный POST-запрос по указанному URL и возвращает ответ в формате JSON.

    Args:
        url (str): URL-адрес, на который отправляется POST-запрос.
        headers (dict): Заголовки HTTP-запроса.
        payload (dict | str): Тело запроса. Может быть словарём (будет автоматически
            сериализовано в JSON) или строкой.

    Returns:
        dict: Ответ сервера, декодированный из JSON.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            resp = await response.text()
            return resp

async def parse(url: str) -> str:
    """Извлекает и структурирует значимый контент с веб-страницы с помощью парсинга и GigaChat.

    Args:
        url (str): URL веб-страницы, которую необходимо обработать.

    Returns:
        str: Структурированный текст в формате Markdown, содержащий только
        релевантную информацию,очищенный от шума и оформленный по заданному шаблону.
    """
    try:
        response = await query(url)
    except Exception as ex:
        logger.error(ex)
    soup = BeautifulSoup(response, 'html5lib')

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
    try:
        token = await get_token(GIGACHAT_CLIENT_SECRET)
        response = await get_answer(prompt, token)
        return response
    except Exception as ex:
        logger.error(ex)


async def parse_links(links: List[str]) -> List[Document]:
    """Асинхронно обрабатывает список веб-ссылок: парсит содержимое, очищает и структурирует текст через LLM.

    Args:
        links (List[str]): Список URL-адресов, которые необходимо распарсить.

    Returns:
        List[langchain_core.documents.Document]: Список объектов Document, каждый из которых содержит:
            - page_content: структурированный текст, очищенный и обработанный GigaChat,
            - metadata: словарь с ключом 'source' — оригинальной ссылкой.
    """
    data = []
    for link in tqdm(links):
        data.append(
            {
                'text': await parse(link),
                'source': link
            }
        )
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return [Document(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(data)]

