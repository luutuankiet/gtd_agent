### **Action Plan: Baseline Evaluation for the Document Support Agent**

**Objective:** To establish a robust, quantitative performance baseline for our RAG-based **Document Support Agent**. This updated plan leverages the `ragas` TestsetGenerator to rapidly create a high-quality evaluation dataset, moving from a manual process to a more scalable, code-first approach.

### **Phase 1: Automated Test Set Generation (P1 / Milestone 1)**

This phase focuses on quickly generating a high-quality, human-validated test set using the default capabilities of the `ragas` TestsetGenerator. The goal is to get a reliable baseline score with minimal manual effort.

1.  **Environment Configuration:**
    *   Create a dedicated Python virtual environment.
    *   Install all necessary libraries:
        ```bash
        pip install google-cloud-aiplatform langchain chromadb litellm openrouter \
        langchain-huggingface sentence-transformers "unstructured[md]" tiktoken \
        ragas datasets python-dotenv
        ```
    *   Set up a `.env` file with your `OPENROUTER_API_KEY`.

2.  **Generate Candidate Dataset:**
    *   **Action:** Use the provided script (adapted from `WIP_p1.md`) to run the `ragas` `TestsetGenerator` on a small, representative subset of the documentation (e.g., 5-10 key files in a `docs_subset/` directory).
    *   **Rationale:** This generates a diverse set of ~20 questions (simple, complex, multi-hop) automatically, saving significant manual effort compared to writing questions by hand.
    *   **Script:**
        ```python
        import os
        from dotenv import load_dotenv
        from ragas.testset import TestsetGenerator
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.document_loaders import DirectoryLoader
        from datasets import Dataset

        load_dotenv()
        os.environ['OPENAI_API_KEY'] = os.getenv("OPENROUTER_API_KEY", "")
        MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")

        generator_llm = LangchainLLMWrapper(ChatOpenAI(model=MODEL, base_url="https://openrouter.ai/api/v1"))
        embedding_model = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

        generator = TestsetGenerator(llm=generator_llm, embedding_model=embedding_model)

        loader = DirectoryLoader("./docs_subset", glob="**/*.md")
        docs = loader.load()

        testset = generator.generate_with_langchain_docs(documents=docs, test_size=20)
        df = testset.to_pandas()
        output_file = "milestone1_golden_dataset_candidate.jsonl"
        df.to_json(output_file, orient="records", lines=True)
        print(f"Successfully generated candidate test set to {output_file}")
        ```

3.  **Human Review & Refinement:**
    *   **Task:** Manually review the generated `milestone1_golden_dataset_candidate.jsonl` file.
    *   **Goal:** Edit, discard, or approve each generated question, answer, and context to ensure high quality and accuracy. This human-in-the-loop step is critical for creating a trustworthy evaluation set.
    *   **Outcome:** The reviewed file becomes the official **"Golden Dataset"**.

**Mentor's Note:** The `TestsetGenerator` is powerful out-of-the-box. By leveraging it, we replace the tedious manual task of writing questions and ground truths with a more efficient "review and refine" workflow. This gets us to a high-quality baseline faster.

### **Phase 2: Execute Baseline Evaluation**

This phase uses the validated "Golden Dataset" to benchmark the agent's current performance.

1.  **Align RAG Pipeline:**
    *   **Chunking:** The existing `RecursiveCharacterTextSplitter` (chunk_size=1000, chunk_overlap=100) in `tools.py` will be used.
    *   **Embeddings:** The evaluation will use the `sentence-transformers/all-MiniLM-L6-v2` model, consistent with the agent's setup.
    *   **Retrieval:** The baseline test will use the simple **similarity search** retriever currently in `agent.py`.

2.  **Generate Agent Responses:**
    *   Write a script to iterate through the validated "Golden Dataset".
    *   For each question, run the agent and capture two key outputs:
        1.  `contexts`: The `page_content` of the documents returned by the `retrieve_documents` tool.
        2.  `answer`: The final generated answer from the agent.
    *   Store these outputs alongside the original `question` and `ground_truth`.

3.  **Evaluate with RAGAS:**
    *   Load the final, populated dataset into a Hugging Face `Dataset` object.
    *   Run `ragas.evaluate()` to calculate the core metrics.
    *   **Code Snippet:**
        ```python
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

        # Assume 'final_dataset_list' is your list of dicts with
        # ['question', 'ground_truth', 'answer', 'contexts']
        dataset = Dataset.from_list(final_dataset_list)

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        print(result)
        ```

### **Phase 3: Documentation & Next Steps (P2 / Milestone 2)**

1.  **Document the Baseline:**
    *   Create `EVALUATION_RESULTS.md`.
    *   Record the RAGAS scores, evaluation date, and the specific LLM used.

2.  **Define Next Steps:**
    *   **Analyze Scores:** Use the baseline scores to identify the weakest link in the RAG pipeline.
        *   Low `context_precision` or `context_recall` points to a need for a **re-ranker** or improved retrieval strategies.
        *   Low `faithfulness` suggests prompt engineering on the generator agent is needed.
    *   **Scale and Stress-Test:** Run the generator on the **entire document set** to create a larger, more challenging test set (100+ questions). Use this to find knowledge gaps and guide optimization.
    *   **Introduce Advanced Customizations:** Experiment with more powerful models for specific parts of the generation pipeline (e.g., a `pro` model for the multi-hop synthesizer) to create even more difficult tests.

This updated plan provides a data-driven path to systematically improving the agent's performance.