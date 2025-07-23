# Retrieval‑Augmented Generation (RAG) Agent
# This agent uses a PDF document to answer questions about stock market performance in 2024.

from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool


load_dotenv()

llm= ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0, #Temperature controls randomness, 0 is deterministic
    max_tokens=1024
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

pdf_path = os.path.join(os.path.dirname(__file__), "Stock_Market_Performance_2024.pdf")

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"Loaded {len(pages)} pages from the PDF.")
    if not pages:
        raise ValueError("No pages found in the PDF.")
except Exception as e:
    raise RuntimeError(f"Failed to load PDF: {e}")

#Chunking process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # Reduced chunk size for more precise chunks
    chunk_overlap=100    # Reduced overlap from 200 to 100
)

pages_split = text_splitter.split_documents(pages)

persist_directory = os.path.join("C:\\Users\\senyi\\Desktop\\LangChain\\ai agents")
collection_name = "stock_market"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    vector_store = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Vector store created and persisted at {persist_directory}.")
except Exception as e:
    raise RuntimeError(f"Failed to create vector store: {e}")

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Reduced from 5 to 3 to get more focused results
)

@tool
def retriever_tool(query: str) -> str:
    """Retrieve relevant information from the document based on the query."""

    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found."
    
    # Remove duplicate content by checking similarity
    unique_docs = []
    seen_content = set()
    
    for doc in docs:
        # Create a simplified version for comparison (remove extra whitespace)
        content_key = " ".join(doc.page_content.split()[:50])  # Use first 50 words as key
        
        if content_key not in seen_content:
            seen_content.add(content_key)
            unique_docs.append(doc)
    
    results = []
    for i, doc in enumerate(unique_docs):
        results.append(f"Document {i+1}:\n{doc.page_content}\n")

    return "\n".join(results)

tools = [retriever_tool]
llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    """Check if the last message tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""


tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of our tools

# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}


# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


running_agent()