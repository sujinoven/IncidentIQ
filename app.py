
"""
IncidentIQ - Ticket Confirmation UI

Demonstrates human-in-the-loop ticket creation.

IMPORTANT:
    The selected demo user is not authenticated.
    This role selector is for local development only.
"""

import streamlit as st

from incidentiq.core.models import (
    AgentRequest,
    UserContext,
)
from incidentiq.frameworks.langgraph_agent import (
    LangGraphAgent,
)
from incidentiq.mcp.client import call_tool


st.set_page_config(
    page_title="IncidentIQ",
    page_icon="🛠️",
    layout="centered",
)

st.title("IncidentIQ")
st.caption("Enterprise IT Incident & Ticket Copilot")

st.subheader("Create an IT Support Ticket")

# Demo identity. Do not use a selectable role in production.
demo_user = UserContext(
    user_id="USR-001",
    role="employee",
)

# Streamlit reruns the script after interactions.
# Session state preserves the pending ticket.
if "pending_ticket" not in st.session_state:
    st.session_state.pending_ticket = None

if "created_ticket" not in st.session_state:
    st.session_state.created_ticket = None


with st.form("ticket_form"):
    service_id = st.selectbox(
        "Service",
        [
            "SVC-VPN",
            "SVC-IDENTITY",
            "SVC-EMAIL",
            "SVC-PAYROLL",
            "SVC-HR",
            "SVC-ENDPOINT",
            "SVC-WIFI",
            "SVC-FILES",
            "SVC-DEVOPS",
            "SVC-MONITOR",
        ],
    )

    title = st.text_input(
        "Ticket title",
        placeholder="VPN connection failure",
    )

    description = st.text_area(
        "Describe your issue",
        placeholder="Explain what is happening...",
    )

    priority = st.selectbox(
        "Priority",
        ["P4", "P3", "P2", "P1"],
        index=1,
    )

    prepare_clicked = st.form_submit_button(
        "Prepare ticket",
        use_container_width=True,
    )


if prepare_clicked:
    # Clear any previous pending confirmation.
    st.session_state.pending_ticket = None
    st.session_state.created_ticket = None

    query = (
        "Create ticket:\n"
        f"Service: {service_id}\n"
        f"Title: {title}\n"
        f"Description: {description}\n"
        f"Priority: {priority}"
    )

    try:
        request = AgentRequest(
            query=query,
            framework="langgraph",
            user=demo_user,
        )

        with st.spinner("Preparing ticket..."):
            response = LangGraphAgent().run(request)

        if response.pending_confirmation_token:
            st.session_state.pending_ticket = {
                "token": response.pending_confirmation_token,
                "service_id": service_id,
                "title": title,
                "description": description,
                "priority": priority,
            }

            st.success("Ticket prepared successfully.")
        else:
            st.warning(response.answer)

    except Exception as exc:
        st.error(f"Ticket preparation failed: {exc}")


pending = st.session_state.pending_ticket

if pending:
    st.divider()
    st.subheader("Review and confirm")

    st.write("**Service:**", pending["service_id"])
    st.write("**Title:**", pending["title"])
    st.write("**Description:**", pending["description"])
    st.write("**Priority:**", pending["priority"])
    st.write("**Requester:**", demo_user.user_id)

    st.info(
        "Your ticket has not been created yet. "
        "Please review the details."
    )

    confirm_col, cancel_col = st.columns(2)

    with confirm_col:
        confirm_clicked = st.button(
            "Confirm ticket",
            type="primary",
            use_container_width=True,
        )

    with cancel_col:
        cancel_clicked = st.button(
            "Cancel",
            use_container_width=True,
        )

    if cancel_clicked:
        st.session_state.pending_ticket = None
        st.rerun()

    if confirm_clicked:
        try:
            # This call is made by the UI confirmation handler,
            # not by the LangGraph agent.
            result = call_tool(
                tool_name="create_ticket",
                arguments={
                    "confirmation_token": pending["token"]
                },
                user=demo_user,
            )

            if result.get("status") == "created":
                st.session_state.created_ticket = result
                st.session_state.pending_ticket = None
                st.rerun()
            else:
                st.error("Ticket creation was not completed.")

        except Exception as exc:
            st.error(f"Ticket creation failed: {exc}")


if st.session_state.created_ticket:
    created = st.session_state.created_ticket

    st.success("Ticket created successfully!")
    st.write("**Ticket ID:**", created["ticket_id"])
    st.write("**Status:** Open")
```

### Why use `st.session_state`?

Streamlit reruns the script whenever a user interacts with a widget.

Without session state, the pending confirmation token could disappear when the user clicks Confirm.

`st.session_state` allows us to retain the pending request between those interactions.

For this local demonstration, the user is fixed as `USR-001`. Later, we'll replace that with authenticated session identity.

---

## Step 4 — Launch Streamlit

From your project root, run:

```powershell
uv run streamlit run src/incidentiq/ui/app.py
```

Streamlit should open the application in your browser.

<box border radius="lg" padding={3} gap={3}>
  <row align="center" gap={2}>
    <icon name="ticket" size="lg"/>
    <box flex="1" gap="2px">
      **IncidentIQ**
      <text color="secondary" size="xs">Preview of the ticket confirmation flow</text>
    </box>
  </row>
  <divider color="subtle"/>
  {@body const [previewTitle,setPreviewTitle]=DIL.useState("VPN connection failure")}
  {@body const [previewDesc,setPreviewDesc]=DIL.useState("Corporate VPN disconnects repeatedly from my laptop.")}
  {@body const [previewState,setPreviewState]=DIL.useState("edit")}
  <box gap={2}>
    <label>Service</label>
    <box background="surface-secondary" radius="md" padding={3}>SVC-VPN</box>
    <label>Ticket title</label>
    <input value={previewTitle} onChange={setPreviewTitle} disabled={previewState!=="edit"}/>
    <label>Description</label>
    <textarea value={previewDesc} onChange={setPreviewDesc} disabled={previewState!=="edit"} rows={2}/>
    <label>Priority</label>
    <box background="surface-secondary" radius="md" padding={3}>P3</box>
  </box>
  {#if previewState==="edit"}
    <button block onClick={()=>setPreviewState("review")} disabled={!previewTitle.trim()||!previewDesc.trim()}>Prepare ticket</button>
  {:else if previewState==="review"}
    <divider color="subtle"/>
    <row align="center" gap={2}>
      <icon name="clock" color="secondary"/>
      **Awaiting confirmation**
    </row>
    <text size="sm">Review your ticket before creating it.</text>
    <row gap={2}>
      <box flex="1">
        <button block variant="outline" color="secondary" onClick={()=>setPreviewState("edit")}>Cancel</button>
      </box>
      <box flex="1">
        <button block onClick={()=>setPreviewState("confirmed")}>Confirm</button>
      </box>
    </row>
  {:else}
    <row align="center" gap={2}>
      <icon name="check-circle" color="success"/>
      **Confirmation demonstrated**
    </row>
    <text color="secondary" size="sm">This interactive preview doesn't call your local MCP server or create a ticket. Use your Streamlit app to test actual persistence.</text>
    <button block variant="outline" color="secondary" onClick={()=>setPreviewState("edit")}>Start again</button>
  {/if}
</box>

## Step 5 — Verify the complete workflow

In your local Streamlit app:

1. Enter a valid ticket title and description.
2. Click **Prepare ticket**.
3. Confirm that the preview appears and no ticket ID exists yet.
4. Click **Confirm ticket**.
5. Verify that a new ticket ID is returned.

You can then look up that ticket using the existing `get_ticket` MCP tool with an IT Support account.

Also test cancellation: preparing a ticket and clicking Cancel must not create a record.

### Important security follow-up

The implementation above demonstrates the correct user experience, but it is **not production-ready authentication or write isolation**. The Streamlit process currently constructs its own user context, and the generic MCP client still has access to `create_ticket`.

Before deploying, we must bind identity to a real authenticated session, separate privileged confirmation calls from agent-accessible calls, and add server-side checks that enforce that trust boundary. We should also add automated tests for expired tokens, unauthorized users, and duplicate confirmations.

## What we've achieved

We now have the pieces of an end-to-end human-in-the-loop workflow:

**User request → LangGraph → MCP prepare_ticket → Streamlit review → explicit confirmation → MCP create_ticket → SQLite.**

Once you run this and verify that the ticket is created and retrievable, our next milestone is **LangSmith tracing and evaluation**, followed by implementing the same shared capabilities in CrewAI and Agno.