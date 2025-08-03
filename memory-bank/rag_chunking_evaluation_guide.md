# RAG Chunking Optimization: A Guide for Engineers

## Introduction to RAG and the Importance of Chunking

Retrieval-Augmented Generation (RAG) systems combine the power of large language models (LLMs) with external knowledge bases to generate more accurate, up-to-date, and grounded responses. A critical component of any RAG system is the **chunking strategy**, which involves breaking down large documents into smaller, manageable pieces (chunks) that can be efficiently stored in a vector database and retrieved by the LLM.

The effectiveness of your RAG system heavily relies on how well these documents are chunked. Suboptimal chunking can lead to:
*   **Poor Retrieval:** Chunks that are too small might lack sufficient context, while chunks that are too large might contain irrelevant information, both leading to less accurate retrievals.
*   **Increased Latency & Cost:** Larger chunks mean more tokens processed by the LLM, increasing inference time and computational costs.
*   **Hallucinations:** If the retrieved context is poor, the LLM might generate responses that are not grounded in the provided documents.

Our current RAG setup, as defined in [`gtd_agent/tools.py`](gtd_agent/tools.py), uses `langchain_text_splitters.RecursiveCharacterTextSplitter` with a `chunk_size` of 1000 characters and a `chunk_overlap` of 100 characters. While this is a common starting point, its optimality for our specific use case (a coding assistant for Lightdash semantic models) needs to be rigorously confirmed.

## Why Evaluation is Necessary

As engineers, we need to move beyond anecdotal evidence and confidently assert that our RAG chunking setup is optimal. This requires a systematic evaluation process that quantifies the performance of our RAG system under different chunking strategies. Without this, we risk deploying a system that is inefficient, inaccurate, or both.

## Key Approaches for Evaluating Chunking Optimality

Optimizing chunking is an iterative process that involves defining clear metrics, setting up an evaluation pipeline, experimenting with parameters, and analyzing results.

### 1. Define an Evaluation Dataset

A robust evaluation begins with a high-quality dataset. This dataset should consist of:
*   **Representative Queries:** A diverse set of questions that users are likely to ask the RAG agent. These queries should cover various topics and complexities present in your knowledge base.
*   **Expected Answers (Ground Truth):** For each query, a human-curated "gold standard" answer. This allows us to measure the accuracy and relevance of the LLM's generated responses.
*   **Relevant Document Chunks (Optional but Recommended):** For each query, identify the specific chunks from your knowledge base that are truly relevant to answering the question. This is crucial for evaluating the retrieval component independently of the generation component.

**Rationale:** A well-defined evaluation dataset provides a consistent benchmark against which different chunking strategies can be compared. Including ground truth relevant chunks allows for a more granular analysis of retrieval performance, which is directly impacted by chunking.

### 2. Utilize RAG Evaluation Metrics and Tools

Traditional NLP metrics (like BLEU or ROUGE) are often insufficient for RAG systems as they don't fully capture the nuances of retrieval and generation quality. Instead, specialized RAG evaluation metrics and tools should be employed.

**Recommended Tool:** **RAGAS** (Retrieval Augmented Generation Assessment) is an excellent framework for evaluating RAG pipelines. It provides metrics that assess different aspects of the RAG system without requiring human labels for every generated answer.

**Key RAGAS Metrics:**
*   **Context Relevancy:** Measures how relevant the retrieved context is to the given question. A high score indicates that the retrieved chunks are focused on the query.
*   **Answer Relevancy:** Evaluates how relevant the generated answer is to the original question. This assesses if the LLM's response directly addresses the user's intent.
*   **Faithfulness (or Hallucination):** Determines if the generated answer is factually consistent with the retrieved context. A high score indicates minimal hallucination.
*   **Context Precision:** Assesses whether all truly relevant items within the retrieved contexts are ranked highly. This helps identify if the retriever is bringing back too much irrelevant information alongside the relevant bits.
*   **Processing Time:** While not a RAGAS metric, it's crucial to monitor the time taken for both chunking (if dynamic) and retrieval. This impacts the user experience and operational costs.

**Rationale:** Using specialized RAG metrics provides a holistic view of the system's performance, directly addressing the impact of chunking on retrieval and generation quality. Tools like RAGAS automate much of this process, making iterative evaluation feasible.

### 3. Experiment with Chunking Parameters

This is the core of optimizing your chunking strategy. Systematically vary the parameters and observe their impact on the evaluation metrics.

*   **Vary `chunk_size` and `chunk_overlap`:**
    *   **`chunk_size`:** Experiment with different chunk sizes (e.g., 250, 500, 750, 1000, 1500 characters).
        *   **Smaller chunks:** Can lead to more precise retrieval but might break up important contextual information.
        *   **Larger chunks:** Retain more context but might introduce noise or exceed the LLM's context window, leading to truncation or reduced performance.
    *   **`chunk_overlap`:** Adjust the overlap (e.g., 0, 50, 100, 200 characters). Overlap helps maintain context across chunks, preventing loss of information at chunk boundaries.
*   **Consider Semantic Chunking:** Instead of fixed-size chunks, explore methods that split text at semantically meaningful boundaries. This often involves using an LLM or advanced NLP techniques to identify logical breaks in the document, ensuring that each chunk represents a coherent piece of information.
*   **Explore Agentic (LLM-based) Chunking:** For highly complex or unstructured documents, an LLM can be employed to intelligently identify optimal chunk boundaries. This is a more advanced technique where the LLM itself helps in the preprocessing step.

**Rationale:** Different document types and use cases will have different optimal chunking parameters. Empirical experimentation is necessary to find the sweet spot that balances context preservation, retrieval precision, and LLM efficiency.

### 4. Iterative Testing and Refinement

Chunking optimization is not a one-time task. It's an iterative process:
1.  **Choose a set of chunking parameters.**
2.  **Re-index your vector store** with the new chunking strategy.
3.  **Run the RAG evaluation pipeline** using your defined dataset and metrics.
4.  **Analyze the results.** Identify which metrics improved or degraded.
5.  **Adjust chunking parameters** based on the analysis.
6.  **Repeat** until you achieve satisfactory performance across all key metrics.

**Rationale:** Continuous feedback and refinement are essential for fine-tuning the RAG system. The insights gained from each iteration will guide you toward a more robust and performant chunking strategy.

## Conclusion

By following these detailed steps, engineers can systematically evaluate and optimize the chunking technique for their RAG systems. This rigorous approach ensures that the RAG agent is built on a solid foundation, leading to more accurate, relevant, and efficient responses, ultimately enhancing the user experience. 


## Appendix: Current ChromaDB Setup and Chunking Implementation

This section provides the detailed implementation of our current ChromaDB vector store setup and the document chunking strategy, as found in [`gtd_agent/tools.py`](gtd_agent/tools.py). This information is crucial for reproducibility and for understanding the baseline against which future chunking optimizations will be evaluated.

### 1. Local ChromaDB Setup

Our ChromaDB instance is configured to persist data locally in the `./chroma_langchain_rag_db` directory. This ensures that the vector store is available across sessions without needing to re-embed documents every time.

**Key Components:**
*   **Embeddings Model:** We use `HuggingFaceEmbeddings` with the `sentence-transformers/all-MiniLM-L6-v2` model. This model is chosen for its balance of performance and efficiency on CPU.
*   **Persistence:** The `persist_directory` parameter ensures that the vector store is saved to disk.

**Code Snippet from `gtd_agent/tools.py`:**

```python
import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import asyncio

async def main():
    # Check if the ChromaDB directory is empty to avoid re-loading
    if not os.listdir('chroma_langchain_rag_db'):
        print('Starting document load and embedding...')
        
        # 1. Initialize Embeddings Model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device': 'cpu'} # Using CPU for broader compatibility
        encode_kwargs = {'normalize_embeddings': False} # Normalization depends on downstream use
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
                
        # 2. Load Documents
        # Loads all Markdown files from the '../docs' directory recursively
        loader = DirectoryLoader("../docs", glob="**/*.md", show_progress=True, use_multithreading=True)
        docs = await loader.aload()
        print(f"Loaded {len(docs)} documents.")
        
        # 3. Split Documents into Chunks
        # Uses RecursiveCharacterTextSplitter for robust text splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs_split = text_splitter.split_documents(docs)
        print(f"Split documents into {len(docs_split)} chunks.")

        # 4. Create and Persist ChromaDB Vector Store
        # Embeds the document chunks and stores them in ChromaDB
        vector_store = Chroma.from_documents(
            documents = docs_split,
            embedding=embeddings,
            persist_directory='./chroma_langchain_rag_db'
        )
        print('ChromaDB loaded and persisted successfully.')
    else:
        print('ChromaDB already loaded. Skipping document loading.')

if __name__ == "__main__":
    asyncio.run(main())
```

**To Reproduce the ChromaDB Setup:**
1.  Ensure you have the necessary Python packages installed (e.g., `langchain-chroma`, `langchain-huggingface`, `langchain-community`, `langchain-text-splitters`, `torch`).
2.  Place your Markdown documentation files in the `../docs` directory relative to `gtd_agent/tools.py`.
3.  Run the `tools.py` script: `python gtd_agent/tools.py`
    *   This will create or update the `./chroma_langchain_rag_db` directory with your embedded document chunks.

### 2. Current Chunking Approach

Our current chunking strategy utilizes `RecursiveCharacterTextSplitter`, which is a versatile text splitter that attempts to split text in a hierarchical manner based on a list of characters.

**Configuration:**
*   **`chunk_size=1000`**: Each chunk aims to be approximately 1000 characters long. This size is a common starting point, balancing the need for sufficient context with the constraints of LLM context windows.
*   **`chunk_overlap=100`**: There is an overlap of 100 characters between consecutive chunks. This overlap helps to preserve context that might otherwise be split across two chunks, reducing the risk of losing critical information at chunk boundaries.

**Rationale for `RecursiveCharacterTextSplitter`:**
This splitter is chosen for its ability to handle various document structures by attempting to split on different characters in a specified order (e.g., `\n\n`, `\n`, ` `, `.` etc.). This helps in creating more semantically coherent chunks compared to a simple fixed-size splitter. The overlap ensures continuity of context.

**Code Snippet for Chunking from `gtd_agent/tools.py`:**

```python
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs_split = text_splitter.split_documents(docs)
```

This detailed overview of the ChromaDB setup and chunking approach provides the necessary context for engineers to understand, reproduce, and further optimize the RAG system's performance.