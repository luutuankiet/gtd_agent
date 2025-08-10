### **Action Plan: Baseline Evaluation for the Document Support Agent**

**Objective:** To establish a robust, quantitative performance baseline for our RAG-based **Document Support Agent**. This plan is tailored to the current implementation using Google's Agent Development Kit (ADK), LiteLLM, and ChromaDB. The final scores will serve as the benchmark for all future optimizations.

### **Phase 0: Foundation & Setup**

This phase ensures the project environment is correctly configured based on the existing agent.py and tools.py files.

1. **Environment Configuration:**  
   * Create a dedicated Python virtual environment.  
   * Install all necessary libraries:  
     pip install google-cloud-aiplatform langchain chromadb litellm openrouter \\  
     langchain-huggingface sentence-transformers "unstructured\[md\]" tiktoken \\  
     ragas datasets

   * Set up environment variables for OPENROUTER\_API\_KEY and any other keys required by LiteLLM.  
2. **Data Loading & Ingestion:**  
   * Ensure the tools.py script has been run successfully to populate the chroma\_langchain\_rag\_db directory.  
   * This script loads documents from the ../docs directory, chunks them, and creates the vector store using the sentence-transformers/all-MiniLM-L6-v2 embedding model.

**Mentor's Note:** A clean, isolated environment prevents dependency conflicts. This setup mirrors the existing code, ensuring the evaluation is performed on the exact same foundation.

### **Phase 1: Building the Evaluation Framework**

This is the most critical manual phase. The quality of our evaluation depends entirely on the quality of our test data.

1. **Create the "Golden Dataset":**  
   * **Task:** Manually review the Lightdash documentation and create a diverse set of 20-30 realistic questions. These questions should cover a range of topics and complexities.  
   * **Diversity is Key:** Include:  
     * **Factual Questions:** "What command is used to deploy a Lightdash project?"  
     * **"How-to" Questions:** "How do I connect a new dbt project?"  
     * **Conceptual Questions:** "What is the difference between a metric and a dimension?"  
     * **Negative Questions:** "Can Lightdash connect to a Google Sheet?" (where the answer is likely no).  
   * **For each question, write a concise, accurate ground\_truth answer.** This is the ideal response you expect from a perfect agent.  
2. **Structure the Dataset:**  
   * Store this data in a structured format, like a JSON or CSV file. This file will be the single source of truth for our evaluation.  
   * Example structure for one entry:  
     {  
       "question": "How do I connect a new dbt project?",  
       "ground\_truth": "To connect a new dbt project, you need to configure the connection details in your \`profiles.yml\` file and then reference that profile in your Lightdash project settings under 'dbt Connection'."  
     }

**Mentor's Note:** This dataset is our scientific control. Without a high-quality, human-verified set of questions and answers, any scores from RAGAS are meaningless. Time spent here is a high-leverage activity.

### **Phase 2: Aligning the RAG Pipeline for Evaluation**

This phase confirms the RAG logic to be tested, based on the current agent.py implementation.

1. **Chunking Strategy:**  
   * The current implementation in tools.py uses RecursiveCharacterTextSplitter with chunk\_size=1000 and chunk\_overlap=100. This is a solid semantic chunking strategy and will be the basis of our evaluation.  
2. **Vector Store & Embeddings:**  
   * The vector store is a Chroma instance persisted at ./gtd\_agent/chroma\_langchain\_rag\_db.  
   * The embedding model is HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2"). The evaluation will proceed with this model.  
3. **Retrieval Mechanism (Baseline):**  
   * The current agent.py implements a **direct similarity search retriever**. The retrieve\_documents tool performs a standard vector\_store.similarity\_search(query).  
   * **This initial baseline evaluation will be conducted using this simple retrieval method.** This allows us to measure the performance of the most straightforward RAG approach first.

**Mentor's Note:** The Kapa.ai blog highlights that a more advanced **retrieve-then-rerank** model is a key best practice. We are consciously deferring this. The goal here is to get a score for our *current* simple retriever. The results will then inform whether implementing a re-ranker is the highest-priority next step for optimization.

### **Phase 3: Execution & Analysis**

Now we run the pipeline and gather our baseline metrics.

1. **Generate Agent Responses:**  
   * Write a script that iterates through your "Golden Dataset" from Phase 1\.  
   * For each question:  
     1. Instantiate and run the root\_agent from agent.py.  
     2. The agent will call the retrieve\_documents tool. It is critical to **capture the page\_content of the documents returned by this tool** to use as the contexts for RAGAS. This may require temporarily modifying the retrieve\_documents function or the agent callback to log this information.  
     3. Capture the final generated answer from the agent's response.  
   * Append the captured contexts and answer to your dataset.  
2. **Evaluate with RAGAS:**  
   * Load your fully populated dataset (now with question, ground\_truth, answer, and contexts) into a Hugging Face Dataset object.  
   * Run the ragas.evaluate() function.  
   * **Code Snippet:**  
     from datasets import Dataset  
     from ragas import evaluate  
     from ragas.metrics import (  
         faithfulness, answer\_relevancy, context\_recall, context\_precision  
     )

     \# Assume 'final\_dataset\_list' is your list of dictionaries  
     dataset \= Dataset.from\_list(final\_dataset\_list)

     result \= evaluate(  
         dataset=dataset,  
         metrics=\[  
             faithfulness, answer\_relevancy, context\_precision, context\_recall  
         \],  
     )

### **Phase 4: Documentation & Next Steps**

1. **Document the Baseline:**  
   * Create a new file in the repository named EVALUATION\_RESULTS.md.  
   * Record the final RAGAS scores from the result dictionary in a clear table.  
   * Include the date of the evaluation and the specific model used (openrouter/google/gemini-2.5-flash-lite-preview-06-17).  
2. **Define Next Steps:**  
   * Analyze the scores. Is context\_precision low? This would strongly suggest that implementing a **re-ranking stage** is the most impactful next step. Is faithfulness low? The generation prompt in agent.py might need improvement.  
   * These scores will provide the data-driven direction for the next phase of optimization.