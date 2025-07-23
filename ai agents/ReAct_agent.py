# ReAct meaning "Reasoning and Acting"
# This agent uses a combination of reasoning and acting to solve problems.

from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage # The foundation for all message types in LangGraph
from langchain_core.messages import ToolMessage # Represents a message that contains a tool call
from langchain_core.messages import SystemMessage # Represents a system message to the model
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool 
def add(a:int, b:int):
    """Add two integers together and return the result."""
    return a + b

@tool
def substract(a:int, b:int):
    """Subtract two integers and return the result."""
    return a - b

@tool
def multiply(a:int, b:int):
    """Multiply two integers together and return the result."""
    return a * b

@tool
def divide(a:int, b:int):
    """Divide two integers and return the result."""
    return a / b

tools = [add, substract, multiply, divide]

model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.0,
    max_tokens=1000
).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are a helpful assistant that can perform calculations.")
    messages = [system_prompt] + state["messages"]
    response = model.invoke(messages)
    return {
        "messages": state["messages"] + [response]
    }

def should_continue(state: AgentState):
    messages = state["messages"]
    print("mevcut state: " + str(state))
    last_message = messages[-1] if messages else None
    if not last_message or not hasattr(last_message, 'tool_calls') or not last_message.tool_calls: 
        return "end"
    else:
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_edge(START, "our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "our_agent")

app = graph.compile()



def print_stream(stream):
    for s in stream: 
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            print(message.pretty_print())

inputs = {"messages": [("user", "Subtract 3 from 4")]}
print_stream(app.stream(inputs, stream_mode="values"))