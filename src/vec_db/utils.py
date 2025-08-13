from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter


def chunks_from_md(docs: List[Document]) -> List[Document]:
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


def get_context(db: Chroma, query: str):
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

def dicts_to_documents(docs: List[dict]):
    return [Document(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in docs]