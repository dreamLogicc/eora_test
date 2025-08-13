import torch

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from loguru import logger
from langchain_chroma import Chroma
from typing import List

def generate_vecdb(chroma_path: str, collection_name: str, data: List[Document]):
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


def connect_to_vecdb(chroma_path, collection_name):
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


def chunks_from_md(docs):
    headers_to_split_on = [
        ("###", "Header"),
    ]

    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)

    splitted_docs = []
    for doc in docs:
        chunks = text_splitter.split_text(doc.page_content)
        splitted_docs += [
            Document(
                page_content=chunk.page_content,
                metadata={
                    **doc.metadata,
                    **chunk.metadata
                }
            ) for chunk in chunks
        ]
    return splitted_docs


def get_context(db, query):
    relevant_docs = db.similarity_search_with_score(query, k=5)
    context_parts = []
    for doc, score in relevant_docs:
        source = doc.metadata.get('source', 'Источник не указан')
        context_parts.append(
            f"{doc.page_content}\n"
            f"Источник: {source}*\n"
        )

    context = "\n".join(context_parts)
    return context
