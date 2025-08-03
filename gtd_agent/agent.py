import os
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import FunctionTool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import date
# from .prompts import return_instructions_root
# from .sub_agents.query_generator.agent import query_generator_agent
# from .sub_agents.data_summarizer.agent import data_summarizer_agent
# from .sub_agents.query_generator.tools import read_business_context
from gtd_agent.patched_lite_llm import LiteLlm
import asyncio
import os

# Initialize ChromaDB and embeddings globally or pass them around
# This assumes your ChromaDB is already populated at './chroma_langchain_rag_db'
# and uses the same embeddings as in gtd_agent/tools.py
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
vector_store = Chroma(persist_directory='./gtd_agent/chroma_langchain_rag_db', embedding_function=embeddings)

async def retrieve_documents(query: str) -> str:
    """
    Retrieves relevant documents from the ChromaDB vector store based on the query.
    """
    docs = vector_store.similarity_search(query)
    return "\n\n".join([doc.page_content for doc in docs])


def update_conversation_history(callback_context: CallbackContext):
    """
    A before_agent_callback to update and format conversation history
    in the session state.
    """
    # Step 1: Get the list of past prompts, or create a new list.
    history_list = callback_context.state.get('conversation_history_list', [])

    # Step 2: Get the new user prompt from the context.
    if callback_context.user_content and callback_context.user_content.parts:
        user_prompt = callback_context.user_content.parts[0].text
        history_list.append(user_prompt)

    # Step 3: Save the updated list back to the state.
    callback_context.state['conversation_history_list'] = history_list
    
    # Step 4: Create a formatted string for the LLM prompt and save it.
    # This matches the format from the implementation guide.
    formatted_history = "\n".join(f'"{prompt}"' for prompt in history_list)
    callback_context.state['conversation_history'] = formatted_history

    return # Return None to allow the agent to proceed normally.

date_today = date.today()

root_agent = LlmAgent(
    model=LiteLlm(
        model=os.getenv("MODEL_ROOT","openrouter/google/gemini-2.5-flash-lite-preview-06-17"),
        max_retries=5,
        initial_delay=2.0,
        max_delay=60.0,
        exponential_base=2.0,
        jitter=True,
    ),
    # model=os.getenv("LOOKER_AGENT_MODEL"),
    name='root_agent',
    description='I am an assistant.',
    instruction="""You are a helpful assistant.
    You have access to a document retrieval tool. Use this tool when the user asks a question that requires information from the knowledge base.
    For example, if the user asks about a specific concept or detail, use the `retrieve_documents` tool with a relevant query to get information from the documents.
    Then, use the retrieved information to answer the user's question.
    """,
    tools=[retrieve_documents],
    # instruction=return_instructions_root(),
    # sub_agents=[
    #     query_generator_agent,
    #     data_summarizer_agent,
    # ],
    # before_agent_callback=[
    #     update_conversation_history,
    #     read_business_context
    #                        ],
)

#TODO: Setup session and runner. Make sure code is being called async.