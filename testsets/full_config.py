import os
from dotenv import load_dotenv

# -- Ragas Core Imports --
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import apply_transforms, HeadlineSplitter
from ragas.testset.transforms.extractors.llm_based import HeadlinesExtractor, KeyphraseExtractor
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.multi_hop.base import MultiHopQuerySynthesizer
from ragas.testset.synthesizers import QueryDistribution

# -- Langchain Imports --
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader

# --- Configuration ---
load_dotenv()
# Ensure OPENROUTER_API_KEY is set in your .env file
os.environ['OPENAI_API_KEY'] = os.getenv("OPENROUTER_API_KEY", "")

# --- 1. Embedding Model (for Topic Diversity) ---
# Fast, local model. Does not need to be changed.
embedding_model = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

# --- 2. Specialized LLMs for Each Task (Updated to Gemini 2.5) ---

# For the final 'ground_truth' answer generation. Needs to be reliable.
generator_llm = LangchainLLMWrapper(ChatOpenAI(
    model="google/gemini-2.5-pro",
    base_url="https://openrouter.ai/api/v1"
))

# For generating simple, single-context questions. Needs to be fast and cheap.
single_hop_llm = LangchainLLMWrapper(ChatOpenAI(
    model="google/gemini-2.5-flash",
    base_url="https://openrouter.ai/api/v1"
))

# For generating complex, multi-context questions. Must be a top-tier model.
multi_hop_llm = LangchainLLMWrapper(ChatOpenAI(
    model="google/gemini-2.5-pro",
    base_url="https://openrouter.ai/api/v1"
))

# --- 3. Load Documents ---
print("Loading documents...")
# Make sure your Lightdash docs are in a 'docs/' directory
loader = DirectoryLoader("./docs", glob="**/*.md")
docs = loader.load()

# --- 4. Create and Enrich Knowledge Graph ---
print("Creating and enriching Knowledge Graph...")
kg = KnowledgeGraph()
for doc in docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata,
            },
        )
    )

# Define and apply transforms to enrich the graph
# Using the more capable generator_llm for extraction
headline_extractor = HeadlinesExtractor(llm=generator_llm)
headline_splitter = HeadlineSplitter()
keyphrase_extractor = KeyphraseExtractor(llm=generator_llm)
transforms = [
    headline_extractor,
    headline_splitter,
    keyphrase_extractor,
]
apply_transforms(kg, transforms=transforms)

# --- 5. Define Query Distribution ---
query_distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=single_hop_llm), 0.7),
    (MultiHopQuerySynthesizer(llm=multi_hop_llm), 0.3),
]

# --- 6. Instantiate the Generator with Custom Components ---
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=embedding_model,
    knowledge_graph=kg,
)

# --- 7. Run Generation ---
print("Generating test set... (This may take a while)")
# Generate a small test set to start
testset = generator.generate_with_langchain_docs(
    docs,
    testset_size=10,
    transforms=transforms,
    distributions=query_distribution,
)

# --- 8. Save the Output ---
df = testset.to_pandas()
output_file = "synthetic_testset.jsonl"
df.to_json(output_file, orient="records", lines=True, indent=4)

print(f"Successfully generated and saved test set to {output_file}")
print(df.head())