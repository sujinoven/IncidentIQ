
import os
import logging

from pathlib import Path
from time import perf_counter  #Latency measurement
from statistics import mean
from datetime import datetime, timezone


# =========================================================
# 1. DEVELOPMENT TELEMETRY CONFIGURATION
# =========================================================


os.environ.setdefault(
    "LANGSMITH_TRACING",
    "false",
)

os.environ.setdefault(
    "AGNO_TELEMETRY",
    "false",
)

os.environ.setdefault(
    "CREWAI_DISABLE_TELEMETRY",
    "true",
)

os.environ.setdefault(
    "OTEL_SDK_DISABLED",
    "true",
)


import streamlit as st

from pydantic import ValidationError

from incidentiq.core.models import (
    AgentRequest,
    AgentResponse,
    UserContext,
    FrameworkName,
)


logger = logging.getLogger(__name__)


# =========================================================
# 2. APPLICATION CONFIGURATION
# =========================================================


PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]


EVALUATION_DIR = (
    PROJECT_ROOT
    / ".local"
    / "evaluation"
)


st.set_page_config(
    page_title="IncidentIQ",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 3. FRAMEWORK CONFIGURATION
# =========================================================


FRAMEWORK_OPTIONS = {
    "LangGraph": "langgraph",
    "CrewAI": "crewai",
    "Agno": "agno",
}


FRAMEWORK_LABELS = {
    value: key
    for key, value in FRAMEWORK_OPTIONS.items()
}


FRAMEWORK_DESCRIPTIONS = {

    "langgraph": (
        "Stateful graph-based orchestration "
        "with deterministic workflow routing."
    ),

    "crewai": (
        "Role-based multi-agent collaboration "
        "using agents, tasks and crews."
    ),

    "agno": (
        "Multi-agent teams with specialist "
        "delegation and coordinated execution."
    ),
}


# =========================================================
# 4. SIMULATED ENTERPRISE USER CONFIGURATION
# =========================================================


ROLE_OPTIONS = {
    "Employee": "employee",
    "IT Support": "it_support",
    "IT Administrator": "it_admin",
}


ROLE_LABELS = {
    value: key
    for key, value in ROLE_OPTIONS.items()
}


DEMO_USERS = {

    "employee": [
        "USR-001",
        "USR-002",
        "USR-003",
        "USR-004",
        "USR-005",
        "USR-006",
        "USR-007",
        "USR-008",
    ],

    "it_support": [
        "USR-009",
        "USR-010",
    ],

    "it_admin": [
        "USR-011",
        "USR-012",
    ],
}


ROLE_DESCRIPTIONS = {

    "employee": (
        "Access employee-public knowledge "
        "and permitted enterprise IT services."
    ),

    "it_support": (
        "Access employee-public and IT-internal "
        "knowledge and operational resources."
    ),

    "it_admin": (
        "Access employee-public, IT-internal "
        "and administrator-restricted resources."
    ),
}


# =========================================================
# 5. APPLICATION NAVIGATION
# =========================================================


NAVIGATION_OPTIONS = [
    "IT Assistant",
    "Ticket Management",
    "Evaluation Dashboard",
    "About IncidentIQ",
]


# =========================================================
# 6. QUICK ACTION CONFIGURATION
# =========================================================


QUICK_ACTIONS = [

    {
        "label": "📚 Knowledge QA",
        "query": "What does VPN Error 809 mean?",
    },

    {
        "label": "🔍 Investigate incident",
        "query": "Investigate incident INC-0012",
    },

    {
        "label": "📡 Service status",
        "query": "Check status of SVC-VPN",
    },

    {
        "label": "🖥️ Search incidents",
        "query": "Search VPN incidents",
    },
]


STAFF_QUICK_ACTIONS = [

    {
        "label": "🎫 Search tickets",
        "query": "Search VPN tickets",
    },

    {
        "label": "💻 Search assets",
        "query": "Find laptop assets",
    },
]


# =========================================================
# 7. ERROR CLASSIFICATION
# =========================================================


SECURITY_ERRORS = {
    "ACCESS_DENIED",
    "RESTRICTED_ACTION",
}


TECHNICAL_ERRORS = {
    "EXECUTION_FAILED",
    "INCIDENT_LOOKUP_FAILED",
    "OPERATIONAL_LOOKUP_FAILED",
    "TICKET_PREPARATION_FAILED",
    "INVALID_TICKET_PREVIEW",
    "INVALID_STRUCTURED_OUTPUT",
    "INVALID_CITATIONS",
    "INVESTIGATION_VALIDATION_FAILED",
}


# =========================================================
# 8. SESSION STATE INITIALIZATION
# =========================================================


def initialize_session_state() -> None:

    defaults = {

        "messages": [],

        "pending_ticket": None,

        "created_ticket": None,

        "execution_history": [],

        "active_context": None,
    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


# =========================================================
# 9. WORKSPACE ISOLATION
# =========================================================


def clear_workspace() -> None:

    st.session_state.messages = []

    st.session_state.pending_ticket = None

    st.session_state.created_ticket = None

    st.session_state.execution_history = []


def synchronize_context(
    framework: FrameworkName,
    user: UserContext,
) -> None:

    current_context = (
        framework,
        user.user_id,
        user.role,
    )

    previous_context = (
        st.session_state.active_context
    )

    if previous_context is None:

        st.session_state.active_context = (
            current_context
        )

        return

    if previous_context != current_context:

        clear_workspace()

        st.session_state.active_context = (
            current_context
        )


# =========================================================
# 10. FRAMEWORK FACTORY
# =========================================================


def build_agent(
    framework: FrameworkName,
):

    if framework == "langgraph":

        from incidentiq.frameworks.langgraph_agent import (
            LangGraphAgent,
        )

        return LangGraphAgent()

    if framework == "crewai":

        from incidentiq.frameworks.crewai_agent import (
            CrewAIAgent,
        )

        return CrewAIAgent()

    if framework == "agno":

        from incidentiq.frameworks.agno_agent import (
            AgnoAgent,
        )

        return AgnoAgent()

    raise ValueError(
        f"Unsupported framework: {framework}"
    )


# =========================================================
# 11. COMMON AGENT EXECUTION INTERFACE
# =========================================================


def execute_agent(
    query: str,
    framework: FrameworkName,
    user: UserContext,
) -> AgentResponse:

    try:

        request = AgentRequest(
            query=query,
            framework=framework,
            user=user,
        )

    except ValidationError:

        return AgentResponse(
            answer=(
                "Please provide a valid request "
                "containing 3 to 2000 characters."
            ),
            framework=framework,
            success=False,
            error="INVALID_INPUT",
        )

    try:

        agent = build_agent(
            framework
        )

        response = agent.run(
            request
        )

        if not isinstance(
            response,
            AgentResponse,
        ):

            return AgentResponse(
                answer=(
                    "The agent returned an "
                    "invalid response."
                ),
                framework=framework,
                success=False,
                error="INVALID_STRUCTURED_OUTPUT",
            )

        if response.framework != framework:

            return AgentResponse(
                answer=(
                    "The response framework "
                    "could not be verified."
                ),
                framework=framework,
                success=False,
                error="INVALID_STRUCTURED_OUTPUT",
            )

        return response

    except PermissionError:

        return AgentResponse(
            answer=(
                "You do not have permission "
                "to perform this operation."
            ),
            framework=framework,
            success=False,
            error="ACCESS_DENIED",
        )

    except Exception as exc:

        logger.error(
            "Agent execution failed: %s",
            type(exc).__name__,
        )

        return AgentResponse(
            answer=(
                "The request could not be completed. "
                "Please try again or contact IT support."
            ),
            framework=framework,
            success=False,
            error="EXECUTION_FAILED",
        )


# =========================================================
# 12. SESSION EXECUTION METRICS
# =========================================================


def record_execution(
    response: AgentResponse,
    latency: float,
) -> None:
    #The assistant message is already stored in the session state, so we only need to record the execution metrics here.
    record = {

        "framework": response.framework,

        "success": response.success,

        "error": response.error,

        "latency_seconds": round(
            latency,
            3,
        ),

        "citation_count": len(
            response.sources
        ),

        "tool_count": len(
            response.tools_used
        ),

        "security_refusal": (
            response.error in SECURITY_ERRORS
        ),

        "technical_failure": (
            response.error in TECHNICAL_ERRORS
        ),

        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }
    #Both assistant and user message are recorded in the session state, but we only want to record the execution metrics for the assistant message. Therefore, we append the record to the execution history here.
    st.session_state.execution_history.append(
        record
    )


def get_session_metrics() -> dict:

    history = (
        st.session_state.execution_history
    )

    total_requests = len(
        history
    )

    successful_requests = sum(
        1
        for record in history
        if record["success"]
    )

    security_refusals = sum(
        1
        for record in history
        if record["security_refusal"]
    )

    technical_failures = sum(
        1
        for record in history
        if record["technical_failure"]
    )

    average_latency = (

        mean([
            record["latency_seconds"]
            for record in history
        ])

        if history

        else 0.0
    )

    return {

        "total_requests": total_requests,

        "successful_requests": (
            successful_requests
        ),

        "security_refusals": security_refusals,

        "technical_failures": technical_failures,

        "average_latency": average_latency,
    }


# =========================================================
# 13. APPLICATION SIDEBAR
# =========================================================


def render_sidebar() -> tuple[
    str,
    FrameworkName,
    UserContext,
]:

    with st.sidebar:

        st.title(
            "🛡️ IncidentIQ"
        )

        st.caption(
            "Enterprise IT Copilot"
        )

        st.divider()

        st.subheader(
            "Workspace"
        )

        page = st.radio(
            "Navigate",
            options=NAVIGATION_OPTIONS,
            label_visibility="collapsed",
            key="workspace_navigation",
        )

        st.divider()

        st.subheader(
            "Agent Configuration"
        )

        framework_label = st.selectbox(
            "Agent framework",
            options=list(
                FRAMEWORK_OPTIONS.keys()
            ),
            key="framework_selector",
        )

        framework = FRAMEWORK_OPTIONS[
            framework_label
        ]

        st.caption(
            FRAMEWORK_DESCRIPTIONS[
                framework
            ]
        )

        st.divider()

        st.subheader(
            "User Context"
        )

        role_label = st.selectbox(
            "Enterprise role",
            options=list(
                ROLE_OPTIONS.keys()
            ),
            key="role_selector",
        )

        role = ROLE_OPTIONS[
            role_label
        ]

        user_id = st.selectbox(
            "User identity",
            options=DEMO_USERS[role],
            key=f"user_selector_{role}",
        )

        user = UserContext(
            user_id=user_id,
            role=role,
        )

        st.caption(
            ROLE_DESCRIPTIONS[role]
        )

        st.warning(
            "Simulated enterprise identity. "
            "Not production authentication."
        )

        # Synchronize before rendering session
        # metrics to prevent stale context display.

        synchronize_context(
            framework=framework,
            user=user,
        )

        st.divider()

        if st.button(
            "Clear workspace",
            use_container_width=True,
        ):

            clear_workspace()

            st.rerun()

        st.caption(
            "NovaTech Solutions — Synthetic Demo"
        )

    return (
        page,
        framework,
        user,
    )


# =========================================================
# 14. APPLICATION HEADER
# =========================================================


def render_header(
    framework: FrameworkName,
    user: UserContext,
) -> None:

    st.title(
        "IncidentIQ"
    )

    st.caption(
        "Enterprise IT Knowledge & "
        "Incident Resolution Copilot"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Active Framework",
            FRAMEWORK_LABELS[framework],
        )

    with col2:

        st.metric(
            "Enterprise Role",
            ROLE_LABELS[user.role],
        )

    with col3:

        st.metric(
            "User Identity",
            user.user_id,
        )

    st.divider()


# =========================================================
# 15. SIDEBAR SESSION METRICS
# =========================================================


def render_session_metrics() -> None:

    metrics = get_session_metrics()

    with st.sidebar:

        st.divider()

        st.subheader(
            "Session Metrics"
        )

        st.metric(
            "Requests",
            metrics["total_requests"],
        )

        st.metric(
            "Successful Responses",
            metrics["successful_requests"],
        )

        st.metric(
            "Average Latency",
            f"{metrics['average_latency']:.2f}s",
        )

        st.metric(
            "Security Refusals",
            metrics["security_refusals"],
        )

        st.metric(
            "Technical Failures",
            metrics["technical_failures"],
        )

        st.caption(
            "Session metrics are operational "
            "statistics, not benchmark scores."
        )


# =========================================================
# 16. QUICK ACTIONS
# =========================================================


def render_quick_actions(
    user: UserContext,
) -> str | None:

    selected_query = None

    st.caption(
        "Quick actions"
    )

    columns = st.columns(2)

    for index, action in enumerate(
        QUICK_ACTIONS
    ):

        with columns[index % 2]:

            if st.button(
                action["label"],
                key=f"quick_action_{index}",
                use_container_width=True,
            ):

                selected_query = (
                    action["query"]
                )

    if user.role in {
        "it_support",
        "it_admin",
    }:

        staff_columns = st.columns(2)

        for index, action in enumerate(
            STAFF_QUICK_ACTIONS
        ):

            with staff_columns[index]:

                if st.button(
                    action["label"],
                    key=f"staff_action_{index}",
                    use_container_width=True,
                ):

                    selected_query = (
                        action["query"]
                    )

    return selected_query


# =========================================================
# 17. KNOWLEDGE CITATION RENDERING
# =========================================================


def render_sources(
    sources: list[dict],
) -> None:

    if not sources:

        return

    with st.expander(
        f"📚 Knowledge Sources ({len(sources)})"
    ):

        for source in sources:

            document_id = source.get(
                "document_id",
                "Unknown",
            )

            title = source.get(
                "title",
                "Untitled document",
            )

            st.text(
                f"{document_id} — {title}"
            )


# =========================================================
# 18. TOOL EXECUTION RENDERING
# =========================================================


def render_tools(
    tools: list[str],
) -> None:

    if not tools:

        return

    with st.expander(
        f"🔧 Tool Execution ({len(tools)})"
    ):

        for index, tool in enumerate(
            tools,
            start=1,
        ):

            tool_type = (

                "RAG Retrieval"

                if tool == "retrieve_knowledge"

                else "MCP Tool"
            )

            st.text(
                f"{index}. {tool} — {tool_type}"
            )

        st.caption(
            "Tools reported by the framework. "
            "A listed tool does not necessarily "
            "indicate successful execution."
        )


# =========================================================
# 19. RESPONSE STATUS RENDERING
# =========================================================


def render_response_status(
    message: dict,
) -> None:

    error = message.get(
        "error"
    )

    if message.get(
        "awaiting_confirmation"
    ):

        st.warning(
            "Ticket prepared. Human confirmation "
            "is required before creation."
        )

    elif error in SECURITY_ERRORS:

        st.warning(
            f"Security policy: {error}"
        )

    elif error in TECHNICAL_ERRORS:

        st.error(
            f"Execution error: {error}"
        )

    elif message.get("success"):

        st.success(
            "Request completed"
        )

    else:

        st.info(
            f"Request not completed: "
            f"{error or 'UNKNOWN_STATUS'}"
        )

    framework = message.get(
        "framework",
        "",
    )

    framework_label = FRAMEWORK_LABELS.get(
        framework,
        framework,
    )

    latency = message.get(
        "latency_seconds",
        0.0,
    )

    st.caption(
        f"Framework: {framework_label} "
        f"| Latency: {latency:.3f}s"
    )


# =========================================================
# 20. CHAT MESSAGE RENDERING
# =========================================================


def render_message(
    message: dict,
) -> None:

    role = message.get(
        "role",
        "assistant",
    )

    with st.chat_message(role):

        if role == "user":

            st.text(
                message["content"]
            )

            return

        # Render model output as plain text.
        # Avoid interpreting untrusted HTML,
        # external images or generated links.

        st.text(
            message["content"]
        )

        render_sources(
            message.get(
                "sources",
                [],
            )
        )

        render_tools(
            message.get(
                "tools_used",
                [],
            )
        )

        render_response_status(
            message
        )


def render_chat_history() -> None:

    for message in (
        st.session_state.messages
    ):

        render_message(
            message
        )


# =========================================================
# 21. PENDING TICKET STATE
# =========================================================


def store_pending_ticket(
    response: AgentResponse,
    framework: FrameworkName,
    user: UserContext,
) -> None:

    token = (
        response.pending_confirmation_token
    )

    if not token:

        return

    if not response.success:

        return

    if "prepare_ticket" not in response.tools_used:

        return

    st.session_state.pending_ticket = {

        "confirmation_token": token,

        "preview": response.answer,

        "framework": framework,

        "user_id": user.user_id,

        "user_role": user.role,

        "prepared_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }


# =========================================================
# 22. NORMALIZE ASSISTANT MESSAGE
# =========================================================


def build_assistant_message(
    response: AgentResponse,
    latency: float,
) -> dict:

    return {

        "role": "assistant",

        "content": response.answer,

        "framework": response.framework,

        "sources": [

            source.model_dump()

            for source in response.sources
        ],

        "tools_used": list(
            response.tools_used
        ),

        "success": response.success,

        "error": response.error,

        "latency_seconds": round(
            latency,
            3,
        ),

        "awaiting_confirmation": bool(
            response.success
            and response.pending_confirmation_token
            and "prepare_ticket" in response.tools_used
        ),
    }


# =========================================================
# 23. PROCESS USER QUERY
# =========================================================


def process_user_query(
    query: str,
    framework: FrameworkName,
    user: UserContext,
) -> None:

    query = query.strip()

    if not query:

        st.warning(
            "Please enter a valid IT request."
        )

        return

    # Prevent an existing pending ticket from
    # being silently replaced by another request.

    if st.session_state.pending_ticket is not None:

        st.warning(
            "A ticket is awaiting confirmation. "
            "Review it in Ticket Management "
            "before submitting another request."
        )

        return
    #Each user interaction produces two conversation records.
    user_message = {

        "role": "user",

        "content": query,
    }

    st.session_state.messages.append(
        user_message
    )

    render_message(
        user_message
    )

    # -----------------------------------------------------
    # Execute selected agent and measure latency
    # -----------------------------------------------------

    start_time = perf_counter()

    with st.spinner(
        f"{FRAMEWORK_LABELS[framework]} "
        "is processing your request..."
    ):

        response = execute_agent(
            query=query,
            framework=framework,
            user=user,
        )

    latency = (
        perf_counter() - start_time
    )

    # -----------------------------------------------------
    # Store pending confirmation securely in
    # server-side session state
    # -----------------------------------------------------

    store_pending_ticket(
        response=response,
        framework=framework,
        user=user,
    )

    # -----------------------------------------------------
    # Normalize response without storing the
    # confirmation token in conversation history
    # -----------------------------------------------------

    assistant_message = build_assistant_message(
        response=response,
        latency=latency,
    )

    st.session_state.messages.append(
        assistant_message
    )

    # -----------------------------------------------------
    # Record operational session metrics
    # -----------------------------------------------------

    record_execution(
        response=response,
        latency=latency,
    )

    # -----------------------------------------------------
    # Render assistant response
    # -----------------------------------------------------

    render_message(
        assistant_message
    )

    # Refresh the application so that pending
    # ticket state and session metrics are current.

    st.rerun()


# =========================================================
# 24. IT ASSISTANT PAGE
# =========================================================


def render_assistant_page(
    framework: FrameworkName,
    user: UserContext,
) -> None:

    st.header(
        "IT Support Assistant"
    )

    st.caption(
        "Ask questions about enterprise knowledge, "
        "incidents, services and operational resources."
    )

    pending_ticket = (
        st.session_state.pending_ticket
    )

    has_pending_ticket = (
        pending_ticket is not None
    )

    if has_pending_ticket:

        st.warning(
            "A ticket is awaiting confirmation. "
            "Open Ticket Management to review it."
        )

    selected_query = None

    if not has_pending_ticket:

        selected_query = render_quick_actions(
            user
        )

    st.divider()

    if not st.session_state.messages:

        with st.chat_message("assistant"):

            st.write(
                "Welcome to IncidentIQ. "
                "I can help you retrieve enterprise "
                "IT knowledge, investigate incidents "
                "and check authorized operational data."
            )

            st.caption(
                "Choose a quick action or enter "
                "your own IT support question."
            )

    render_chat_history()

    chat_query = st.chat_input(
        "Ask an enterprise IT question...",
        max_chars=2000,
        disabled=has_pending_ticket,
    )

    query = (
        chat_query
        or selected_query
    )

    if query:

        process_user_query(
            query=query,
            framework=framework,
            user=user,
        )


# =========================================================
# 25. TICKET MANAGEMENT PAGE
# =========================================================


def render_ticket_page() -> None:

    st.header(
        "Ticket Management"
    )

    st.write(
        "Prepare and review enterprise IT "
        "support tickets."
    )

    pending_ticket = (
        st.session_state.pending_ticket
    )

    if pending_ticket is not None:

        st.warning(
            "A ticket is awaiting confirmation."
        )

        st.subheader(
            "Pending Ticket Preview"
        )

        st.text(
            pending_ticket["preview"]
        )

        st.caption(
            "Requester: "
            f"{pending_ticket['user_id']}"
        )

        st.info(
            "Ticket confirmation will be enabled "
            "in the next implementation step."
        )

        if st.button(
            "Discard pending preview",
            type="secondary",
        ):

            st.session_state.pending_ticket = None

            st.rerun()

        st.caption(
            "Discarding the preview clears the "
            "local session only. The backend "
            "preparation record expires separately."
        )

    else:

        st.info(
            "No ticket is currently awaiting "
            "confirmation."
        )

        st.write(
            "The dedicated ticket preparation "
            "form will be added in Module 11.3."
        )


# =========================================================
# 26. EVALUATION DASHBOARD PAGE
# =========================================================


def render_evaluation_page(
    user: UserContext,
) -> None:

    st.header(
        "Framework Evaluation Dashboard"
    )

    # Evaluation reports can contain privileged
    # enterprise test queries and answers.

    if user.role != "it_admin":

        st.warning(
            "The evaluation dashboard is "
            "restricted to IT administrators."
        )

        return

    st.write(
        "Compare LangGraph, CrewAI and Agno "
        "using shared enterprise evaluation metrics."
    )

    reports = []

    if EVALUATION_DIR.exists():

        reports = sorted(
            EVALUATION_DIR.glob(
                "comparison_*.json"
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Frameworks",
            len(FRAMEWORK_OPTIONS),
        )

    with col2:

        st.metric(
            "Comparison Reports",
            len(reports),
        )

    if reports:

        st.caption(
            f"Latest report: {reports[0].name}"
        )

    else:

        st.info(
            "No comparison reports are available. "
            "Run the evaluation runner first."
        )

    st.info(
        "Detailed benchmark visualizations "
        "will be implemented in Module 11.4."
    )


# =========================================================
# 27. ABOUT PAGE
# =========================================================


def render_about_page() -> None:

    st.header(
        "About IncidentIQ"
    )

    st.markdown(
        """
### Enterprise IT Knowledge & Incident Resolution Copilot

IncidentIQ is an enterprise agentic AI application
designed to demonstrate intelligent IT support
using authorized organizational knowledge and
operational data.

### Supported Agentic Frameworks

- LangGraph
- CrewAI
- Agno

### Core Technologies

- Python 3.11
- LangChain
- OpenAI models
- Chroma vector database
- Model Context Protocol (MCP)
- SQLite
- Pydantic
- Streamlit
- LangSmith

### Enterprise Capabilities

- Retrieval-Augmented Generation
- Multi-agent orchestration
- Incident investigation
- Operational tool execution
- Role-based access control
- Input guardrails
- Structured output validation
- Human-in-the-loop ticket creation
- Comparative framework evaluation

### Enterprise Environment

NovaTech Solutions is a fictional enterprise
created for this demonstration.

All organizational data used in this application
is synthetic.

### Security Notice

The application uses simulated enterprise
identities and is intended for local development
and demonstration, not production deployment.
        """
    )


# =========================================================
# 28. APPLICATION FOOTER
# =========================================================


def render_footer() -> None:

    st.divider()

    st.caption(
        "IncidentIQ | Enterprise Agentic AI "
        "| NovaTech Solutions"
    )


# =========================================================
# 29. MAIN APPLICATION
# =========================================================


def main() -> None:

    initialize_session_state()

    page, framework, user = (
        render_sidebar()
    )

    render_header(
        framework=framework,
        user=user,
    )

    if page == "IT Assistant":

        render_assistant_page(
            framework=framework,
            user=user,
        )

    elif page == "Ticket Management":

        render_ticket_page()

    elif page == "Evaluation Dashboard":

        render_evaluation_page(
            user=user
        )

    elif page == "About IncidentIQ":

        render_about_page()

    render_session_metrics()

    render_footer()


# =========================================================
# 30. APPLICATION ENTRY POINT
# =========================================================


if __name__ == "__main__":

    main()
