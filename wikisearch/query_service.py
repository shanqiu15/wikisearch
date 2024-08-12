from llama_index.core import load_index_from_storage, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
import os
from Pydantic import BaseModel
from typing import Optional


INDEX_STORE = "/Users/hao/doc_workspace/wikisearch/data/inference/hotpot_val_1k"
FT_MODEL = "/Users/hao/doc_workspace/wikisearch/data/inference/ft_100000_v2"


class Document(BaseModel):
    title: str
    url: str
    text: str


class SearchResponse(BaseModel):
    query: str
    answer: Optional[str] = ""
    results: list[Document]


def perform_search(query):
    # Placeholder for search logic
    return f"Results for '{query}' (simulated data)"


class QueryService:

    def __init__(self, index_location: str = INDEX_STORE, model_location: str = FT_MODEL):
        """_summary_

        Args:
            index_location (str, optional): path to the document index. Defaults to INDEX_STORE.
            model_location (str, optional): path to model snapshot. Defaults to FT_MODEL.
        """
        vector_store = FaissVectorStore.from_persist_dir(index_location)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=index_location
        )
        embed_model = HuggingFaceEmbedding(model_name=model_location, trust_remote_code=True)
        self.index = load_index_from_storage(
            storage_context=storage_context, embed_model=embed_model
        )

        self.retrieve_engine = self.index.as_retriever()

        if "OPENAI_API_KEY" in os.environ:
            query_engine = self.index.as_query_engine()
            # setup base query engine as tool
            query_engine_tools = [
                QueryEngineTool(
                    query_engine=query_engine,
                    metadata=ToolMetadata(
                        name="wiki_search_engine",
                        description="Wikipedia query engin",
                    ),
                ),
            ]
            llm = OpenAI(temperature=0.1, model="gpt-4o")

            self.sub_query_engine = SubQuestionQueryEngine.from_defaults(
                query_engine_tools=query_engine_tools,
                use_async=True,
                llm=llm,
            )

    def search(self, query: str):
        resp = self.retrieve_engine.retrieve(query)

        return SearchResponse(
            query=query,
            results=[
                Document(
                    url=doc.node.metadata["url"], title=doc.node.metadata["title"], text=doc.text
                )
                for doc in resp
            ],
        )

    def answer(self, query: str):
        resp = self.sub_query_engine.query(query)

        return SearchResponse(
            query=query,
            answer=resp.response,
            results=[
                Document(
                    url=doc.node.metadata["url"], title=doc.node.metadata["title"], text=doc.text
                )
                for doc in resp.source_nodes
            ],
        )
