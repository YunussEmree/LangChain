#pip install python-dotenv langchain-openai langchain langchain-core

import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0, max_tokens=1000)

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"AI: {response.content}")
    print("Current State:", state)
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while(user_input != "exit"):
    conversation_history.append(HumanMessage(content=user_input))
    
    result = agent.invoke({"messages": conversation_history})

    print(result["messages"])
    conversation_history = result["messages"]

    user_input = input("Enter: ")

