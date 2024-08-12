from llama_index.core import load_index_from_storage, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
import os
from pydantic import BaseModel
from typing import Optional


INDEX_STORE = "data/inference/hotpot_val_1k"
FT_MODEL = "data/inference/ft_100000_v2"


class Document(BaseModel):
    title: str
    url: str
    text: str

    def __str__(self):
        prefix = "    "
        return f"{prefix}Title: {self.title}\n{prefix}URL: {self.url}\n{prefix}Text: {self.text}"


class SearchResponse(BaseModel):
    query: str
    answer: Optional[str] = ""
    contexts: list[Document]

    def __str__(self):
        results_str = "\n\n".join(str(result) for result in self.contexts)
        return f"Query: {self.query}\nAnswer: {self.answer}\nRetrieved Documents:\n{results_str}"


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
        else:
            print(
                "\033[31m"
                + "\nOPENAI_API_KEY not found in environment variables. Will only search for relevant documents. To enable answer generation please set OPENAI_API_KEY in your environment variables.\n"
                + "\033[0m"
            )

    def search(self, query: str):
        resp = self.retrieve_engine.retrieve(query)

        return SearchResponse(
            query=query,
            contexts=[
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
            contexts=[
                Document(
                    url=doc.node.metadata["url"], title=doc.node.metadata["title"], text=doc.text
                )
                for doc in resp.source_nodes
            ],
        )
