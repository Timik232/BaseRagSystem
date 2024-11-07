from rag import RAG, CustomAPILLM
from transformers import AutoModel, AutoTokenizer


def simple_rag_init(
    data_path: str = "../war_and_peace.txt",
    embedding_model_name="intfloat/multilingual-e5-large-instruct",
    llm_model_name="hermes-3-llama-3.1-8b",
    logging=True,
) -> RAG:
    with open(data_path, "r") as file:
        war_and_peace = file.read()
    if logging:
        print("File was read")
    url = "http://localhost:1234/v1/chat/completions"
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    model = AutoModel.from_pretrained(embedding_model_name)
    model = model
    if logging:
        print("Embedding model was loaded")
    llm = CustomAPILLM(api_url=url, model_name=llm_model_name)
    if logging:
        print("LLM model was created")
    rag = RAG(
        tokenizer, model, war_and_peace, "war_and_peace", llm, embedding_model_name
    )
    if logging:
        print("RAG system was created")
    return rag


if __name__ == "__main__":
    rag = simple_rag_init("../data/war_and_peace_small.txt")
    print(rag("Кто главный персонаж книги война и мир?"))
    print(rag.query("что делал Долохов?"))
