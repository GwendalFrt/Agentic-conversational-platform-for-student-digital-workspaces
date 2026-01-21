from langchain_mistralai import ChatMistralAI
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
load_dotenv()
import os

MISTRAL_API_KEY = os.environ['MISTRAL_API_KEY']
# modèle de langage
llm = ChatMistralAI(
            model="mistral-small-latest",
            temperature=0,
            max_retries=5
        )
# modèle d'embedding
embedding_function = SentenceTransformerEmbeddingFunction(model_name="intfloat/multilingual-e5-large")
