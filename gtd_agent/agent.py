import os
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
# from .prompts import return_instructions_root
from datetime import date
# from .sub_agents.query_generator.agent import query_generator_agent
# from .sub_agents.data_summarizer.agent import data_summarizer_agent
# from .sub_agents.query_generator.tools import read_business_context
from gtd_agent.patched_lite_llm import LiteLlm

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
        model=os.getenv("MODEL_ROOT"),
        max_retries=5,
        initial_delay=2.0,
        max_delay=60.0,
        exponential_base=2.0,
        jitter=True,
    ),
    # model=os.getenv("LOOKER_AGENT_MODEL"),
    name='root_agent',
    description='I am an assistant.',
    instruction="you are a helpful assistant."
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