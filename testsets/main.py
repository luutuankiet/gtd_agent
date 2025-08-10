#%%
from langchain_community.document_loaders import DirectoryLoader
import os
from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI

from langchain_huggingface import HuggingFaceEmbeddings
from ragas.testset import TestsetGenerator


model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
model_name=model_name,
model_kwargs=model_kwargs,
encode_kwargs=encode_kwargs
)

path = "samples"
loader = DirectoryLoader(path, glob="*.md")
docs = loader.load()
#%%

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENROUTER_API_KEY","")
MODEL = os.getenv("OPENROUTER_MODEL","")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

generator_llm = LangchainLLMWrapper(ChatOpenAI(
                                            model=MODEL, 
                                            base_url="https://openrouter.ai/api/v1"
                                            ))
generator_embeddings = LangchainEmbeddingsWrapper(embeddings)


generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)
dataset.to_pandas().to_json("testset.jsonl",
                            indent=4,
                            index=False,
                            orient="records",
                            lines=True
                            )
print("ok")
# %%