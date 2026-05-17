"""MCP client for AgentCore Gateway with OAuth token management.

Connects to the AgentCore Gateway via Streamable HTTP and handles
Cognito OAuth client_credentials token acquisition and caching.
"""

import os
import time
import logging

import requests
from mcp.client.streamable_http import streamablehttp_client
from strands.tools.mcp.mcp_client import MCPClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gateway configuration from environment variables
# ---------------------------------------------------------------------------

GATEWAY_URL: str = os.environ.get(
    "GATEWAY_URL",
    "https://agentcoreproject-workshop-gateway-e3o0rwleno.gateway.bedrock-agentcore.us-west-2.amazonaws.com/mcp",
)
GATEWAY_CLIENT_ID: str = os.environ.get(
    "GATEWAY_CLIENT_ID", "46cncg3b4p5kutefgkgqt195gs"
)
GATEWAY_CLIENT_SECRET: str = os.environ.get(
    "GATEWAY_CLIENT_SECRET",
    "18hm7n4f0gemc9iltt2rd73kubvrvda3eo6nsoo5f1gp1v1pso28",
)
GATEWAY_TOKEN_ENDPOINT: str = os.environ.get(
    "GATEWAY_TOKEN_ENDPOINT",
    "https://workshop-gateway-auth-822428608511.auth.us-west-2.amazoncognito.com/oauth2/token",
)
GATEWAY_SCOPE: str = os.environ.get("GATEWAY_SCOPE", "gateway/invoke")

# ---------------------------------------------------------------------------
# OAuth token cache — stores the current access token and its expiry time
# ---------------------------------------------------------------------------

_token_cache: dict = {"access_token": "", "expires_at": 0.0}


def get_access_token() -> str:
    """Obtain a JWT access token from Cognito using client_credentials grant.

    Caches the token and auto-refreshes 60 seconds before expiry to avoid
    using an expired token during an in-flight request.
    """
    # Return cached token if still valid (with 60s buffer)
    if _token_cache["access_token"] and time.time() < _token_cache["expires_at"] - 60:
        return _token_cache["access_token"]

    # Request a new token from the Cognito token endpoint
    response = requests.post(
        GATEWAY_TOKEN_ENDPOINT,
        data={
            "grant_type": "client_credentials",
            "client_id": GATEWAY_CLIENT_ID,
            "client_secret": GATEWAY_CLIENT_SECRET,
            "scope": GATEWAY_SCOPE,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    response.raise_for_status()
    token_data = response.json()

    # Cache the token with its expiry timestamp
    _token_cache["access_token"] = token_data["access_token"]
    _token_cache["expires_at"] = time.time() + token_data.get("expires_in", 3600)
    logger.info("Obtained new gateway access token (expires_in=%s)", token_data.get("expires_in"))
    return _token_cache["access_token"]


def get_gateway_mcp_client() -> MCPClient:
    """Create an MCP client that connects to the AgentCore Gateway with OAuth.

    The transport factory fetches a fresh token on each connection to ensure
    the Authorization header always carries a valid Bearer token.
    """
    def _transport_factory():
        token = get_access_token()
        return streamablehttp_client(
            url=GATEWAY_URL,
            headers={"Authorization": f"Bearer {token}"},
        )

    return MCPClient(_transport_factory)
