
import os

from pathlib import Path

import streamlit as st

from incidentiq.core.models import (
    AgentRequest,
    AgentResponse,
    UserContext,
    FrameworkName,
)


# =========================================================
# 1. APPLICATION CONFIGURATION
# =========================================================


PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]


# Disable optional development telemetry.

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


st.set_page_config(

    page_title="IncidentIQ",

    page_icon="🛡️",

    layout="wide",

    initial_sidebar_state="expanded",
)


# =========================================================
# 2. FRAMEWORK CONFIGURATION
# =========================================================


FRAMEWORK_OPTIONS = {

    "LangGraph": "langgraph",

    "CrewAI": "crewai",

    "Agno": "agno",
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
# 3. SIMULATED ENTERPRISE USERS
# =========================================================


ROLE_OPTIONS = {

    "Employee": "employee",

    "IT Support": "it_support",

    "IT Administrator": "it_admin",
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
        "Access employee-public enterprise "
        "knowledge and permitted IT services."
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
# 4. APPLICATION NAVIGATION
# =========================================================


NAVIGATION_OPTIONS = [

    "IT Assistant",

    "Ticket Management",

    "Evaluation Dashboard",

    "About IncidentIQ",
]


# =========================================================
# 5. SESSION STATE INITIALIZATION
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
# 6. SESSION CONTEXT ISOLATION
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
# 7. FRAMEWORK FACTORY
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
# 8. COMMON AGENT EXECUTION INTERFACE
# =========================================================


def execute_agent(
    query: str,

    framework: FrameworkName,

    user: UserContext,
) -> AgentResponse:

    request = AgentRequest(

        query=query,

        framework=framework,

        user=user,
    )
    #Instead of creating all three frameworks when Streamlit starts, we instantiate only the selected framework when an agent request is executed.
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

        raise TypeError(
            "Invalid agent response."
        )

    return response


# =========================================================
# 9. APPLICATION SIDEBAR
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

        # -------------------------------------------------
        # Navigation
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Framework selection
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Enterprise role selection
        # -------------------------------------------------

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
            "Demo identity selection only. "
            "This is not production authentication."
        )

        st.divider()

        # -------------------------------------------------
        # Workspace controls
        # -------------------------------------------------

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
# 10. APPLICATION HEADER
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

    framework_labels = {

        value: key

        for key, value
        in FRAMEWORK_OPTIONS.items()
    }

    role_labels = {

        value: key

        for key, value
        in ROLE_OPTIONS.items()
    }

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.metric(

            label="Active Framework",

            value=framework_labels[
                framework
            ],
        )

    with col2:

        st.metric(

            label="Enterprise Role",

            value=role_labels[
                user.role
            ],
        )

    with col3:

        st.metric(

            label="User Identity",

            value=user.user_id,
        )

    st.divider()


# =========================================================
# 11. IT ASSISTANT PAGE
# =========================================================


def render_assistant_page(
    framework: FrameworkName,

    user: UserContext,
) -> None:

    st.header(
        "IT Support Assistant"
    )

    st.write(
        "Ask questions about enterprise IT "
        "knowledge, incidents and services."
    )

    col1, col2 = st.columns(
        2
    )

    with col1:

        with st.container(
            border=True
        ):

            st.subheader(
                "📚 Knowledge Assistant"
            )

            st.write(
                "Retrieve authorized enterprise "
                "knowledge, policies and runbooks."
            )

            st.code(
                "What does VPN Error 809 mean?",
                language=None,
            )

        with st.container(
            border=True
        ):

            st.subheader(
                "🔍 Incident Investigation"
            )

            st.write(
                "Investigate incidents using "
                "operational records and knowledge."
            )

            st.code(
                "Investigate incident INC-0012",
                language=None,
            )

    with col2:

        with st.container(
            border=True
        ):

            st.subheader(
                "📡 Service Health"
            )

            st.write(
                "Retrieve current recorded "
                "enterprise service status."
            )

            st.code(
                "Check status of SVC-VPN",
                language=None,
            )

        with st.container(
            border=True
        ):

            st.subheader(
                "🖥️ Operational Search"
            )

            st.write(
                "Search authorized incidents, "
                "support tickets and IT assets."
            )

            st.code(
                "Search VPN incidents",
                language=None,
            )

    st.divider()

    st.info(
        "The conversational assistant will "
        "be enabled in Module 11.2."
    )


# =========================================================
# 12. TICKET MANAGEMENT PAGE
# =========================================================


def render_ticket_page() -> None:

    st.header(
        "Ticket Management"
    )

    st.write(
        "Prepare enterprise IT support tickets "
        "using a human confirmation workflow."
    )

    st.info(
        "Ticket preparation, preview and "
        "confirmation will be integrated "
        "in Module 11.3."
    )


# =========================================================
# 13. EVALUATION DASHBOARD PAGE
# =========================================================


def render_evaluation_page() -> None:

    st.header(
        "Framework Evaluation Dashboard"
    )

    st.write(
        "Compare LangGraph, CrewAI and Agno "
        "using shared enterprise evaluation metrics."
    )

    evaluation_dir = (

        PROJECT_ROOT

        / ".local"

        / "evaluation"
    )

    reports = []

    if evaluation_dir.exists():

        reports = sorted(

            evaluation_dir.glob(
                "comparison_*.json"
            ),

            key=lambda path: (
                path.stat().st_mtime
            ),

            reverse=True,
        )

    col1, col2 = st.columns(
        2
    )

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

    st.info(
        "Detailed benchmark charts and metric "
        "comparisons will be implemented "
        "in Module 11.4."
    )


# =========================================================
# 14. ABOUT PAGE
# =========================================================


def render_about_page() -> None:

    st.header(
        "About IncidentIQ"
    )

    st.markdown(
        """
### Enterprise IT Knowledge & Incident Resolution Copilot

IncidentIQ is an enterprise agentic AI capstone
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

NovaTech Solutions is a fictional organization
created for this capstone project.

All enterprise data used in this demonstration
is synthetic.
        """
    )


# =========================================================
# 15. APPLICATION FOOTER
# =========================================================


def render_footer() -> None:

    st.divider()

    st.caption(
        "IncidentIQ | Enterprise Agentic AI Capstone "
        "| NovaTech Solutions"
    )


# =========================================================
# 16. MAIN APPLICATION
# =========================================================


def main() -> None:

    initialize_session_state()

    page, framework, user = (
        render_sidebar()
    )

    synchronize_context(

        framework=framework,

        user=user,
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

        render_evaluation_page()

    elif page == "About IncidentIQ":

        render_about_page()

    render_footer()


# =========================================================
# 17. APPLICATION ENTRY POINT
# =========================================================


if __name__ == "__main__":

    main()
