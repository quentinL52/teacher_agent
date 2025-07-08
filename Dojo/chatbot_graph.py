import os
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from tools import get_tools_for_agent
from fastapi import FastAPI
import uvicorn
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    last_course_topic: str


tools = get_tools_for_agent()
model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
model = model.bind_tools(tools)

def agent_node(state):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

def router(state) -> str:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, ToolMessage):
        return "end"
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "call_tool"
    return "end"



workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("action", tool_node)
workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    router,
    {"call_tool": "action", "end": END}
)
workflow.add_edge("action", END)

graph = workflow.compile()


app = FastAPI(
    title="teching_agent",
    description="agent qui aide a creer des cours tech", 
)

class Query(BaseModel):
    question: str

@app.post('/ask')
def ask_question(query: Query):
    """
    Pose une question à l'assistant IA et retourne sa réponse.
    """
    final_state = graph.invoke({"messages": [HumanMessage(content=query.question)]})
    response_message = "Désolé, je n'ai pas pu générer de réponse."
    for msg in reversed(final_state["messages"]):
        if hasattr(msg, "content") and not getattr(msg, "tool_calls", None):
            response_message = msg.content
            break

    return {"answer": response_message}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)