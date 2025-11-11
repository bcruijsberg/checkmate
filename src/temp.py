from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Load your model
llm = ChatOllama(
    model="unsloth_llama_q4_k_m:latest",
    temperature=0.1,
    base_url="http://localhost:11434"
)

messages = [
    SystemMessage(content=(
        "You may reason internally but do not show reasoning.\n"
        "Never output <think> or chain-of-thought.\n"
        "Only respond with the final answer."
    )),
    HumanMessage(content="Explain how photosynthesis works in simple terms.")
]

response = llm.invoke(messages)
print(response.content)