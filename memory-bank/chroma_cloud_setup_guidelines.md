# ChromaDB Local to Cloud Deployment Guidelines

This document outlines the end-to-end process for setting up a local ChromaDB vectorstore with embeddings and then deploying it to Chroma Cloud using the Chroma CLI. This guide is intended for developers who need to migrate or maintain ChromaDB instances.

## 1. Setting up Local Vectorstore and Embeddings

The local ChromaDB vectorstore is created and populated using the `gtd_agent/tools.py` script. This script handles document loading, text splitting, and embedding generation using HuggingFace embeddings.

### Code (`gtd_agent/tools.py`)

```python
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
```

### Steps to create the local vectorstore:

1.  **Ensure dependencies are installed:** Make sure `langchain-chroma`, `langchain-google-genai` (if using Google embeddings), `langchain-community`, `langchain-text-splitters`, and `langchain-huggingface` are installed.
2.  **Run the script:** Execute the `gtd_agent/tools.py` script. This will load documents from the `../docs` directory, split them, embed them, and persist the ChromaDB to the `./chroma_langchain_rag_db` directory.

    ```bash
    python gtd_agent/tools.py
    ```

## 2. Deploying to Chroma Cloud using Chroma CLI

Once the local ChromaDB vectorstore is created, you can use the Chroma CLI to migrate the data to Chroma Cloud.

### Prerequisites:

*   **Chroma CLI installed:** Ensure you have the Chroma CLI installed.
    ```bash
    pip install chromadb-cli
    ```

 *   **Chroma Cloud Credentials:** Authenticate with Chroma Cloud using the `chroma login` command. This command will guide you through providing your `CHROMA_API_KEY`, `CHROMA_TENANT`, and `CHROMA_DATABASE`, or using OAuth via a browser.
    ```bash
    chroma login
    ```
    Alternatively, if you prefer, you can set these as environment variables or load them from a `.env` file:
    ```
    CHROMA_API_KEY="your_chroma_api_key"
    CHROMA_TENANT="your_chroma_tenant_id"
    CHROMA_DATABASE="main"
    ```

### Migration Steps:

1.  **Start a local ChromaDB server:** The `chroma copy` command requires a running local ChromaDB instance to copy from.
    ```bash
    chroma run --path gtd_agent/chroma_langchain_rag_db
    ```
    *   **Note:** Keep this terminal session active as the local server needs to be running during the copy operation.

2.  **Copy data to Chroma Cloud:** In a new terminal session, execute the `chroma copy` command. Make sure your environment variables for Chroma Cloud are set.
    ```bash
    chroma copy --all --from-local --to-cloud --db main
    ```
    *   `--all`: Copies all collections from the local instance.
    *   `--from-local`: Specifies the source is the local ChromaDB.
    *   `--to-cloud`: Specifies the destination is Chroma Cloud.
    *   `--db main`: Specifies the target database in Chroma Cloud (replace `main` if you are using a different database name).

## 3. Integration and Verification

After the data migration is complete, you can update your MCP server tool's configuration to connect to Chroma Cloud instead of the local ChromaDB.

1.  **Update Configuration:** Modify your application's configuration to use Chroma Cloud connection details (API key, tenant, database).
2.  **Thorough Testing:** Thoroughly test your application to ensure it correctly interacts with the Chroma Cloud instance and that all data is accessible and queryable.

This end-to-end setup ensures a consistent and deployable vectorstore solution.