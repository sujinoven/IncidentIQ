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
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "incidentiq.mcp.server"],
        env=environment,
    )

    async with stdio_client(server_parameters) as (
        read,
        write,
    ):
        async with ClientSession(read, write) as session:
            await session.initialize()

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
