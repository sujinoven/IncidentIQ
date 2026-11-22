
import re
from typing import Literal, TypedDict

from langgraph.graph import StateGraph, START, END

from incidentiq.core.models import (
    AgentRequest,
    AgentResponse,
)
from incidentiq.rag.service import answer_question


Route = Literal[
    "knowledge",
    "incident",
    "ticket",
]


class IncidentIQState(TypedDict):
    request: AgentRequest
    route: Route
    response: AgentResponse | None
#first router node.It examines the question and selects a workflow.
def classify_request(state: IncidentIQState) -> dict:
    """Select the workflow based on the user request."""

    query = state["request"].query.lower().strip()

    if re.search(r"\b(create|open|raise|submit)\b.*\bticket\b", query):
        route: Route = "ticket"

    elif re.search(
        r"\b(incident|inc-\d{4}|investigate|outage)\b",
        query,
    ):
        route = "incident"

    else:
        route = "knowledge"

    return {"route": route}


#This node reuses the RAG service which we already implemented to answer questions using authorized knowledge.
def knowledge_node(state: IncidentIQState) -> dict:
    """Answer using our shared, RBAC-filtered RAG service."""

    request = state["request"]

    response = answer_question(request)

    return {"response": response}


#Safe placeholder for unfinished workflows. These nodes will be replaced with the actual incident and ticket workflows once they are implemented.
def incident_node(state: IncidentIQState) -> dict:
    """Temporary response until MCP investigation is ready."""

    return {
        "response": AgentResponse(
            answer=(
                "Incident investigation is not available yet. "
                "The MCP incident tools will be connected "
                "in a later module."
            ),
            framework="langgraph",
            success=False,
            error="INCIDENT_WORKFLOW_NOT_IMPLEMENTED",
        )
    }


def ticket_node(state: IncidentIQState) -> dict:
    """Temporary response until ticket approval is ready."""

    return {
        "response": AgentResponse(
            answer=(
                "Ticket creation is not available yet. "
                "No ticket has been created."
            ),
            framework="langgraph",
            success=False,
            error="TICKET_WORKFLOW_NOT_IMPLEMENTED",
        )
    }


#ADding the conditional routing logic to the graph. The graph will start at the START node, then classify the request, and based on the classification, it will route to the appropriate node (knowledge_node, incident_node, or ticket_node). Finally, it will end at the END node.
def select_route(state: IncidentIQState) -> Route:
    """Return the route selected by the classifier."""

    return state["route"]

#Building and compiling the graph. The graph is constructed with nodes and edges, and then compiled into a final workflow that can be executed.
def build_graph():
    """Construct the IncidentIQ LangGraph workflow."""

    graph = StateGraph(IncidentIQState) #Creates a graph that uses our state structure.


    graph.add_node("classify", classify_request)
    graph.add_node("knowledge", knowledge_node)
    graph.add_node("incident", incident_node)
    graph.add_node("ticket", ticket_node)

    graph.add_edge(START, "classify")

    graph.add_conditional_edges(
        "classify",
        select_route,
        {
            "knowledge": "knowledge",
            "incident": "incident",
            "ticket": "ticket",
        },
    )

    graph.add_edge("knowledge", END)
    graph.add_edge("incident", END)
    graph.add_edge("ticket", END)

    return graph.compile()   #turns the graph definition into an executable workflow and checks its structure.


incidentiq_graph = build_graph()



class LangGraphAgent:   #Framework ADapter for the shared agent interface. This class implements the run method, which takes an AgentRequest, processes it through the LangGraph workflow, and returns an AgentResponse.
    """LangGraph implementation of the shared agent interface."""

    def run(self, request: AgentRequest) -> AgentResponse:

        if request.framework != "langgraph":
            raise ValueError(
                "LangGraphAgent requires framework='langgraph'"
            )

        initial_state: IncidentIQState = {
            "request": request,
            "route": "knowledge",
            "response": None,
        }

        final_state = incidentiq_graph.invoke(initial_state)

        response = final_state.get("response")

        if response is None:
            raise RuntimeError(
                "LangGraph completed without a response"
            )

        return AgentResponse.model_validate(response)
