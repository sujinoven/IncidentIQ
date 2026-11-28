#The server.py exposes tools, but our LangGraph agent needs a way to invoke them.
"""
IncidentIQ MCP Client

Starts the local MCP server and invokes its tools.

The client passes trusted application user context
through the subprocess environment.
"""

import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from incidentiq.core.models import UserContext
from langsmith import traceable


async def _call_tool_async(
    tool_name: str,
    arguments: dict,
    user: UserContext,
) -> dict:
    """
    Connect to the local MCP server and execute
    one tool using trusted application context.
    """

    environment = os.environ.copy()

    environment["INCIDENTIQ_USER_ID"] = (
        user.user_id
    )

    environment["INCIDENTIQ_USER_ROLE"] = (
        user.role
    )

    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "incidentiq.mcp.server",
        ],
        env=environment,
    )

    # Complete the MCP session before interpreting
    # the returned tool result.

    async with stdio_client(
        server_parameters
    ) as (read, write):

        async with ClientSession(
            read,
            write,
        ) as session:

            await session.initialize()

            result = await session.call_tool(
                tool_name,
                arguments,
            )

    # Handle tool-level errors outside the
    # asynchronous context managers.

    if result.isError:
        raise RuntimeError(
            f"MCP tool failed: {tool_name}"
        )

    if result.structuredContent is not None:

        content = result.structuredContent

        if (
            isinstance(content, dict)
            and set(content) == {"result"}
            and isinstance(
                content["result"],
                dict,
            )
        ):
            content = content["result"]

    else:

        content = None

        for block in result.content:

            if block.type == "text":

                content = json.loads(
                    block.text
                )

                break

    if not isinstance(content, dict):

        raise RuntimeError(
            f"Invalid MCP result: {tool_name}"
        )

    if (
        content.get("success") is False
        and content.get("error")
        == "ACCESS_DENIED"
    ):

        raise PermissionError(
            "Access to this operation is denied."
        )

    return content


def call_tool(
    tool_name: str,
    arguments: dict,
    user: UserContext,
) -> dict:
    """
    Synchronous wrapper around the asynchronous MCP client.

    This lets our existing synchronous LangGraph nodes
    invoke MCP tools without changing the graph to async.
    """

    return asyncio.run(
        _call_tool_async(
            tool_name=tool_name,
            arguments=arguments,
            user=user,
        )
    )


# Tools the agent is allowed to request.
AGENT_ALLOWED_TOOLS = frozenset({
    "get_incident",
    "search_incidents",
    "get_service_status",
    "search_tickets",
    "get_ticket",
    "search_assets",
    "prepare_ticket",
})




@traceable(
    name="IncidentIQ MCP Tool",
    run_type="tool",
    process_inputs=lambda inputs: {
        "tool_name": inputs.get("tool_name")
    },
    process_outputs=lambda outputs: {
        "result_recorded": True
    },
)
def call_agent_tool(
    tool_name: str,
    arguments: dict,
    user: UserContext,
) -> dict:
    """
    Execute an authorized agent-facing MCP tool.

    LangSmith records the execution duration and tool name.

    Raw tool arguments, user context, and returned
    operational records are excluded from this custom trace.
    """

    if tool_name not in AGENT_ALLOWED_TOOLS:
        raise PermissionError(
            f"Agent cannot invoke tool: {tool_name}"
        )

    return call_tool(
        tool_name=tool_name,
        arguments=arguments,
        user=user,
    )
