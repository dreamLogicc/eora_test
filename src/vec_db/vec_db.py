import json
import os

import torch

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger
from langchain_chroma import Chroma
from typing import List

from parser.parser import parse_links
from vec_db.utils import chunks_from_md, dicts_to_documents


def generate_vecdb(chroma_path: str, collection_name: str, data: List[Document]) -> Chroma:
    """Создаёт векторную базу данных Chroma из списка документов с использованием HuggingFace-эмбеддингов.

    Args:
        chroma_path (str): Путь к директории, где будет сохранена.
        collection_name (str): Имя коллекции в Chroma.
        data: Список объектов Document, каждый из которых содержит текст (page_content) и метаданные (metadata).

    Returns:
        Chroma: Экземпляр векторной базы данных Chroma
    """
    try:

        logger.info("Загрузка модели эмбеддингов...")
        embeddings = HuggingFaceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Создание Chroma DB...")
        chroma_db = Chroma.from_texts(
            texts=[item.page_content for item in data],
            metadatas=[item.metadata for item in data],
            embedding=embeddings,
            persist_directory=chroma_path,
            collection_name=collection_name,
        )
        return chroma_db
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        raise


def connect_to_vecdb(chroma_path: str, collection_name: str) -> Chroma:
    """Подключается к существующей векторной базе данных Chroma.

        Args:
            chroma_path (str): Путь к директории, где хранится сохранённая
                векторная база данных.
            collection_name (str): Имя коллекции.

        Returns:
            Chroma: Экземпляр векторной базы данных Chroma.
    """
    try:
        logger.info("Загрузка модели эмбеддингов...")
        embeddings = HuggingFaceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        chroma_db = Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings,
            collection_name=collection_name,
        )

        logger.success("Успешное подключение к базе Chroma")
        return chroma_db
    except Exception as e:
        logger.error(f"Ошибка подключения к Chroma: {e}")
        raise


async def initialize_db(chroma_path, collection_name, json_path, links):
    """Инициализирует векторную базу данных: либо загружает существующую, либо создаёт новую.

    Args:
        chroma_path (str): Путь к директории для хранения/загрузки Chroma DB.
        collection_name (str): Имя коллекции в Chroma.
        json_path (str): Путь к JSON-файлу, в котором хранятся предварительно
            извлечённые документы.
        links (List[str]): Список URL-адресов, которые будут распаршены,
            если JSON-файл не найден. Используется для получения исходных данных.

    Returns:
        Chroma: Готовый экземпляр векторной базы данных Chroma.
    """
    if not os.path.exists(chroma_path):
        logger.info('Сбор документов...')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
                documents = chunks_from_md(dicts_to_documents(documents))
        else:
            documents = chunks_from_md(await parse_links(links))
        return generate_vecdb(chroma_path, collection_name, documents)
    return connect_to_vecdb(chroma_path, collection_name)


