import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import asyncio


async def main():
    if not os.listdir('chroma_langchain_rag_db'):
        print('starting load....')
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device': 'cpu'} # Use 'cuda' if you have a GPU
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        # embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
                

        loader = DirectoryLoader("../docs", glob="**/*.md", show_progress=True, use_multithreading=True)
        docs = await loader.aload()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs_split = text_splitter.split_documents(docs)


        vector_store = Chroma.from_documents(
            documents = docs_split,
            embedding=embeddings,
            persist_directory='./chroma_langchain_rag_db'
        )
    else:
        print('docs already loaded.')


if __name__ == "__main__":
    asyncio.run(main())