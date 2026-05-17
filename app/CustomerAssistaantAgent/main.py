"""Returns & Refunds Assistant - main entry point.

This agent uses the Strands Agents SDK with AgentCore Memory for
persistent long-term memory across sessions. It connects to the
AgentCore Gateway via MCP for tool access (data_lookup and
policy_retrieval Lambda functions) and uses OAuth for authentication.
"""

import os
import uuid

from strands import Agent
from strands_tools import current_time
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory.integrations.strands.config import (
    AgentCoreMemoryConfig,
    RetrievalConfig,
)
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager,
)
from model.load import load_model
from mcp_client.client import get_gateway_mcp_client

app = BedrockAgentCoreApp()
log = app.logger

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

# Memory configuration
MEMORY_ID: str = os.environ.get(
    "AGENTCORE_MEMORY_ID",
    "AgentCoreProject_CustomerAssistantMemory-1lnC3eFpea",
)
ACTOR_ID: str = "administrator"
REGION: str = "us-west-2"


# ---------------------------------------------------------------------------
# Agent creation with AgentCore Memory and Gateway MCP tools
# ---------------------------------------------------------------------------


def get_or_create_agent(session_id: str, mcp_client):
    """Create the agent with gateway MCP tools, current_time, and AgentCore Memory.

    The gateway exposes these tools from the Lambda functions:
      - data-lookup___order_lookup (customer_id)
      - data-lookup___product_lookup (product_id)
      - data-lookup___user_lookup (customer_id)
      - policy-retrieval___policy_retrieval (query, optional country)

    Returns a tuple of (agent, session_manager).
    """
    # Configure AgentCore Memory with retrieval from all strategy namespaces
    memory_config = AgentCoreMemoryConfig(
        memory_id=MEMORY_ID,
        session_id=session_id,
        actor_id=ACTOR_ID,
        retrieval_config={
            "/facts/{actorId}/": RetrievalConfig(top_k=10, relevance_score=0.0),
            "/summaries/{actorId}/{sessionId}/": RetrievalConfig(top_k=5, relevance_score=0.0),
            "/preferences/{actorId}/": RetrievalConfig(top_k=5, relevance_score=0.0),
        },
    )

    # Create session manager that handles STM and LTM automatically
    log.info(f"Memory config: memory_id={MEMORY_ID}, actor_id={ACTOR_ID}, session_id={session_id}")
    session_manager = AgentCoreMemorySessionManager(
        agentcore_memory_config=memory_config,
        region_name=REGION,
    )

    # Discover tools from the gateway — these are the real Lambda-backed tools
    gateway_tools = mcp_client.list_tools_sync()
    log.info(f"Discovered {len(gateway_tools)} tools from gateway")

    # Combine gateway tools with the built-in current_time tool
    tools = [current_time] + gateway_tools

    agent = Agent(
        model=load_model(),
        system_prompt=(
            "You are the Returns & Refunds Assistant with persistent long-term memory. "
            "You remember information across conversations automatically.\n\n"
            "IMPORTANT: You have a memory system that provides context from previous sessions. "
            "When you see memory context injected into the conversation, ALWAYS use that information "
            "in your response. Memory context takes priority over tool lookups for preferences, "
            "facts, and notes about customers.\n\n"
            "The user is an administrator with access to customer data, orders, and return policies. "
            "Your role is to help administrators check return eligibility, calculate refund amounts, "
            "and answer questions about return policies on behalf of customers.\n\n"
            "Guidelines:\n"
            "- When the user asks about customer preferences or facts, check your memory context FIRST.\n"
            "- Use gateway tools for order details, product info, customer info, and policy lookups.\n"
            "- The data-lookup tools require a customer_id or product_id parameter.\n"
            "- The policy-retrieval tool accepts a natural language query and optional country code.\n"
            "- When the user shares customer preferences or facts, confirm you have noted them.\n"
            "- Be helpful and concise in your responses.\n"
            "- Confirm all relevant details before processing any return or refund.\n"
            "- Use the current_time tool when you need today's date to check return window eligibility.\n"
            "- Clearly state whether a return is eligible or not, and explain why."
        ),
        tools=tools,
        session_manager=session_manager,
    )
    return agent, session_manager


@app.entrypoint
async def invoke(payload, context):
    """AgentCore runtime entrypoint - streams agent responses back to the caller."""
    log.info("Invoking Returns & Refunds Agent...")

    # Use session_id from payload or context for session continuity
    session_id: str = (
        payload.get("sessionId")
        or payload.get("session_id")
        or getattr(context, "session_id", None)
        or str(uuid.uuid4())
    )
    log.info(f"Using session_id: {session_id}")

    # Create the MCP client that connects to the AgentCore Gateway with OAuth
    mcp_client = get_gateway_mcp_client()
    session_manager = None

    try:
        # Open the MCP connection — this establishes the Streamable HTTP session
        with mcp_client:
            agent, session_manager = get_or_create_agent(session_id, mcp_client)

            # Invoke the agent with the user's prompt
            result = agent(payload.get("prompt"))

            # Extract text from the response
            response_text = ""
            if hasattr(result, "message") and result.message:
                for block in result.message.get("content", []):
                    if "text" in block:
                        response_text += block["text"]

            yield response_text
    finally:
        # Flush any buffered memory messages
        if session_manager:
            session_manager.close()


if __name__ == "__main__":
    app.run()
