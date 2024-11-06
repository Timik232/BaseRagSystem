from rag import CustomAPILLM, RAG
from transformers import AutoModel, AutoTokenizer

if __name__ == "__main__":
    with open("war_and_peace.txt", "r") as file:
        war_and_peace = file.read()
    print("File was read")
    url = 'http://localhost:1234/v1/chat/completions'
    model_name = "intfloat/multilingual-e5-large-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print("Embedding model was loaded")
    LLMModel= "hermes-3-llama-3.1-8b"
    LLM = CustomAPILLM(api_url=url, model_name=LLMModel)
    print("LLM model was created")
    rag = RAG(tokenizer, model, war_and_peace, "war_and_peace", LLM)
    print("RAG system was created")
    print(rag("Кто главный персонаж книги война и мир?"))

"""
#init Vector DB
embeddings = OpenAIEmbeddings()

doc_store = Qdrant.from_documents(
    war_and_peace,
    embeddings,
    location=":memory:",
    collection_name="docs",
)

llm = OpenAI()
# ask questions

while True:
    question = input('Ваш вопрос: ')
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=doc_store.as_retriever(),
        return_source_documents=False,
    )

    result = qa(question)
    print(f"Answer: {result}")
"""