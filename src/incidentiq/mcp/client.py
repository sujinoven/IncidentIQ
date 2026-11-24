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
    Connect to the MCP server and call one tool.

    A new isolated server subprocess is created for
    each call in this initial implementation.
    """

    # Copy the application's environment, including
    # the configured OpenAI-related settings.
    environment = os.environ.copy()

    # Set the authenticated demo user for this subprocess.
    environment["INCIDENTIQ_USER_ID"] = user.user_id
    environment["INCIDENTIQ_USER_ROLE"] = user.role

    # Launch the server using the same Python interpreter
    # managed by uv.
    #A process is a running instance of a program.A subprocess is a separate process launched by another program.
    server_parameters = StdioServerParameters(
        command=sys.executable,  #It identifies the Python interpreter running the client.
        args=["-m", "incidentiq.mcp.server"], #The -m option tells Python to execute a module by its importable module name.
        env=environment,
    )
    #This starts the server subprocess and establishes the input/output streams.
    async with stdio_client(server_parameters) as (
        read,
        write,
    ):
        async with ClientSession(read, write) as session: #ClientSessionA session manages the MCP protocol interaction.It allows the client to initialize communication and invoke tools.
            await session.initialize() #This performs the MCP initialization handshake.During initialization, the client and server establish protocol compatibility and exchange capability information.After initialization, the client can call registered tools.

            # Call the MCP tool by its registered name.
            result = await session.call_tool(
                tool_name,
                arguments,
            )

            if result.isError:
                raise RuntimeError(
                    f"MCP tool failed: {tool_name}"
                )

            # MCP tools can return structured content.
            if result.structuredContent is not None:
                content = result.structuredContent

                # Some SDK versions wrap structured output.
                if (
                    isinstance(content, dict)
                    and set(content) == {"result"}
                    and isinstance(content["result"], dict)
                ):
                    return content["result"]

                return content

            # Fall back to JSON text content.
            for block in result.content:
                if block.type == "text":
                    return json.loads(block.text)

            raise RuntimeError(
                f"No usable MCP result from {tool_name}"
            )


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
