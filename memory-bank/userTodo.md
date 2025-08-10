# User To-Do List

## P1: Baseline Evaluation with Ragas

### 1.1 Phase 0: Foundation & Setup (Completed)

### 1.2 Phase 1: Building the Evaluation Framework

*   **1.2.1 Define the evaluation dataset.**
    *   **Why:** A well-defined dataset of questions and ground-truth answers is crucial for objectively evaluating the RAG pipeline's performance. This dataset will be used by Ragas to calculate various metrics.
*   **1.2.2 Implement a basic RAG pipeline using Langchain and ChromaDB.**
    *   **Why:** This serves as the baseline RAG system that we will evaluate. It needs to be functional to generate responses for the evaluation dataset.
*   **1.2.3 Integrate Ragas into the evaluation workflow.**
    *   **Why:** Ragas provides the tools and metrics necessary to assess the quality of the RAG pipeline's responses, including aspects like faithfulness, answer relevance, and context recall.
*   **1.2.4 Run a preliminary evaluation with Ragas on a small subset of the dataset.**
    *   **Why:** This initial run helps verify that the evaluation framework is correctly set up and provides early insights into the RAG pipeline's performance before a full-scale evaluation.