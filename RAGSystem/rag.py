import json

import requests
import torch
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from transformers import AutoModel, AutoTokenizer


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

    def _call(self, prompt: str) -> str:
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
            "model": self.LLMModel,
            "messages": [{"role": "user", "content": prompt}],
        }
        json_data = json.dumps(data)
        headers = {
            "Content-Type": "application/json",
        }
        response = requests.post(self.url, headers=headers, data=json_data)
        if response.status_code == 200:
            response_data = response.json()
            return self.get_text_from_response(response_data)
        else:
            raise ValueError(
                f"API request failed with status code {response.status_code}"
            )

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
        self.vector_store = Qdrant(
            client=qdrant_client, collection_name=collection_name
        )
        self.llm = llm
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
        qdrant_client.create_collection(
            collection_name=collection_name,
            vector_size=embeddings[0].shape[0],
            distance="Cosine",
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
            retriever=self.vector_store,
            return_source_documents=False,
        )

    def __call__(self, query: str) -> str:
        """
        Ask question
        :param query: question
        :return: answer
        """
        response = self.rag_chain.run(query)
        return response
