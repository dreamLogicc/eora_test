from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter


def chunks_from_md(docs: List[Document]) -> List[Document]:
    """Разбивает список документов с Markdown-содержимым на чанки по заголовкам.

    Args:
        docs (List[Document]): Список объектов Document,
            содержимое которых (`page_content`) представляет собой текст в формате
            Markdown, содержащий заголовки ###.

    Returns:
        List[Document]: Список новых документов — чанков,
            полученных после разделения. Каждый чанк содержит:

            - часть исходного текста разбитого по ###,

            - объединённые метаданные: оригинальные + новые ({'Header': '...'}).
    """
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


def get_context(db: Chroma, query: str) -> str:
    """Извлекает релевантный контекст из векторной базы данных на основе запроса.

     Args:
         db (Chroma): Экземпляр векторной базы данных Chroma.
         query (str): Входной текстовый запрос, по которому ищутся релевантные фрагменты.

     Returns:
         str: Строка, содержащая объединённый контент релевантных документов.
     """
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


def dicts_to_documents(docs: List[dict]) -> List[Document]:
    """Конвертирует список словарей в список объектов Document из LangChain.

    Args:
        docs (List[dict]): Список словарей, каждый из которых должен содержать:

            - 'text' (str): Основной текстовый контент документа.

            - 'source' (str): Источник текста (URL, имя файла и т.п.).

    Returns:
        List[Document]: Список объектов Document, где каждый содержит:

            - page_content: текст из поля 'text',

            - metadata: словарь с ключом 'source'.

    """
    return [Document(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in docs]
