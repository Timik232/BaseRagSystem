import json

import requests
import torch
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetriever
from langchain.llms.base import LLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
from transformers import AutoModel, AutoTokenizer


class QdrantRetriever(BaseRetriever):
    """
    Custom retriever class for QdrantVectorStore.
    """

    vector_store: QdrantVectorStore = Field(..., description="store for vectors")

    def _get_relevant_documents(self, query_text: str, k: int = 4):
        # query_embedding = self.vector_store.embeddings.embed_query(query_text)
        results = self.vector_store.similarity_search(query_text, k=k)
        return results

    def retrieve(self, query_text: str, k: int = 4):
        return self._get_relevant_documents(query_text, k)


class CustomAPILLM(LLM):
    """
    Custom LLM class that use LLM API.
    """

    api_url: str = Field(..., description="URL for HTTP requests")
    model_name: str = Field(..., description="Name of the LLM model")

    def get_text_from_response(self, response: dict) -> str:
        """
        Extracts the text from the LLM's response.

        Args:
            response: The LLM's response in JSON format.

        Returns:
            The text from the response.

        Raises:
            ValueError: If the response format is invalid.
        """
        try:
            return response["choices"][0]["message"]["content"]
        except KeyError:
            raise ValueError("Invalid response format")

    def _call(self, prompt: str, *args, **qwargs) -> str:
        """
        Calls the LLM API and returns the response.

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            The LLM's response.

        Raises:
            ValueError: If the API request fails or the response is invalid.
        """

        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        json_data = json.dumps(data)
        headers = {
            "Content-Type": "application/json",
        }
        session = requests.Session()
        response = session.post(self.api_url, headers=headers, data=json_data)
        session.close()
        if response.status_code == 200:
            response_data = response.json()
            return self.get_text_from_response(response_data)
        else:
            raise ValueError(
                f"API request failed with status code {response.status_code}"
            )

    @property
    def _identifying_params(self):
        return {"api_url": self.api_url, "model_name": self.model_name}

    @property
    def _llm_type(self):
        return "custom"


class RAG:
    """
    Class that implements the RAG system.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModel,
        text: str,
        collection_name: str,
        llm: CustomAPILLM,
        embedding_model_name: str,
    ):
        """
        Init Vector DB
        :param tokenizer: tokenizer for vector db
        :param model: model for vector db
        :param text: text to vector db
        :param collection_name: name to vector db
        :param llm: custom LLM model
        """
        self.tokenizer = tokenizer
        self.model = model
        qdrant_client = QdrantClient(host="localhost", port=6333)
        if not qdrant_client.collection_exists(collection_name):
            self.qdrant_create(text, qdrant_client, collection_name)
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vector_store = QdrantVectorStore(
            client=qdrant_client, collection_name=collection_name, embedding=embeddings
        )
        self.llm = llm
        self.qdrant_retriever = QdrantRetriever(vector_store=self.vector_store)
        self.rag_chain = self.create_retriever()

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text
        :param text: text
        :return: tensor
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.squeeze().numpy()

    def qdrant_create(
        self,
        text: str,
        qdrant_client: QdrantClient,
        collection_name: str = "text-collection",
    ):
        """
        Connect to Qdrant
        :param text: text
        :param collection_name: collection name
        :return: Qdrant
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)
        embeddings = [self.encode_text(chunk) for chunk in chunks]
        vectors_config = VectorParams(size=embeddings[0].shape[0], distance="Cosine")
        qdrant_client.create_collection(
            collection_name=collection_name, vectors_config=vectors_config
        )
        points = [
            PointStruct(id=i, vector=embedding.tolist(), payload={"text": chunk})
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        qdrant_client.upsert(collection_name=collection_name, points=points)

    def create_retriever(self) -> RetrievalQA:
        """
        Return retriever
        :return: retriever
        """
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.qdrant_retriever,
            return_source_documents=False,
        )

    def __call__(self, query: str) -> dict:
        """
        Ask question, return answer in dictionary form
        :param query: question
        :return: answer
        """
        response = self.rag_chain.invoke(query)
        return response

    def query(self, query: str) -> str:
        """
        Ask question, return answer in string form
        :param query: question
        :return: answer
        """
        response = self.rag_chain.invoke(query)
        return response["result"]
