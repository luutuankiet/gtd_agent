# Summary of Findings: RAG Test Set Generation

**Date:** 2025-08-10

**Context:** This document summarizes the updated strategy for creating RAG evaluation test sets, moving from a manual creation process to a more efficient, scalable workflow using the `ragas` library. This approach was adopted after research documented in `WIP_p1.md`.

---

### 1. Key Insight: Code-First Generation over Manual Creation

The most significant finding is the superiority of a **code-first, automated generation** approach using the `ragas.testset.TestsetGenerator`.

*   **Previous Method:** Manually writing 20-30 questions and `ground_truth` answers. This was time-consuming, difficult to scale, and prone to human bias in question diversity.
*   **New Method:** Leveraging the `TestsetGenerator` to automatically create a diverse range of questions (simple, abstract, multi-hop) directly from the source documents. The primary human task shifts from *creation* to *review and refinement*.

**Benefit:** This dramatically accelerates the creation of a high-quality, "Golden Dataset" and allows for much larger-scale, purely synthetic test sets for stress-testing.

---

### 2. The Phased Evaluation Plan

We have adopted a two-milestone approach to de-risk the process and build momentum.

*   **P1 / Milestone 1: Validate the Default Workflow**
    *   **Objective:** Quickly establish a high-quality, human-validated baseline score.
    *   **Process:**
        1.  Run the default `TestsetGenerator` on a *small subset* of documents (5-10 files).
        2.  An engineer *manually reviews and refines* the output (`.jsonl` file).
        3.  Benchmark the RAG agent against this validated "Golden Dataset".
    *   **Outcome:** A reliable baseline RAGAS score with minimal initial effort.

*   **P2 / Milestone 2: Scale and Stress-Test**
    *   **Objective:** Identify agent weaknesses and guide optimization efforts.
    *   **Process:**
        1.  Run the generator on the *entire document set* to create a large (100+) test set.
        2.  Introduce advanced `ragas` customizations (e.g., using more powerful LLMs for specific question synthesizers).
        3.  Analyze performance drops on more complex questions to identify where to focus optimization (e.g., retrieval, re-ranking, prompting).

---

### 3. Technical Implementation Notes

*   **Core Components:** The generator requires an `llm` (for reasoning) and an `embedding_model` (for topic clustering).
*   **Recommended Models (Baseline):**
    *   **Generator LLM:** A fast, cost-effective model like `google/gemini-2.5-flash`.
    *   **Embedding Model:** A standard, efficient model like `sentence-transformers/all-MiniLM-L6-v2`.
*   **Workflow:** The process is encapsulated in a single script that loads documents, runs the generator, and saves the candidate test set to a `.jsonl` file for review.

This updated strategy provides a more robust, scalable, and data-driven framework for evaluating and improving our RAG agent.