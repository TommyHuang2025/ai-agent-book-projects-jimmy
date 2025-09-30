"""Collaboration Tools MCP Server

This MCP server provides tools for:
- Browser automation (using browser-use)
- Human-in-the-loop assistance requests
- IM and email notifications
- Timer/scheduling capabilities
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import tool modules
from browser_tools import (
    browser_navigate,
    browser_get_content,
    browser_execute_task,
    browser_screenshot,
    browser_list_tabs,
    close_browser,
    init_browser
)
from notification_tools import (
    send_email,
    send_telegram_message,
    send_slack_message,
    send_discord_message
)
from hitl_tools import (
    request_admin_approval,
    request_admin_input,
    respond_to_request,
    list_pending_requests
)
from timer_tools import (
    set_timer,
    set_recurring_timer,
    cancel_timer,
    list_timers,
    get_timer_status,
    _load_timers
)
from config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("collaboration-tools")


# ============================================================================
# BROWSER AUTOMATION TOOLS
# ============================================================================

@mcp.tool(description="Navigate to a URL in the virtual browser")
async def mcp_browser_navigate(
    url: str = Field(description="The URL to navigate to"),
    new_tab: bool = Field(default=False, description="Whether to open in a new tab")
) -> str:
    """Navigate to a URL in the browser."""
    result = await browser_navigate(url, new_tab)
    return str(result)


@mcp.tool(description="Get content from the current browser page")
async def mcp_browser_get_content(
    selector: Optional[str] = Field(default=None, description="Optional CSS selector to extract specific content")
) -> str:
    """Get content from the current page."""
    result = await browser_get_content(selector)
    return str(result)


@mcp.tool(description="Execute a high-level browser task using AI agent")
async def mcp_browser_execute_task(
    task: str = Field(description="Natural language description of the task to perform"),
    max_steps: int = Field(default=20, description="Maximum number of steps the agent can take")
) -> str:
    """Execute a browser task using autonomous AI agent."""
    result = await browser_execute_task(task, max_steps)
    return str(result)


@mcp.tool(description="Take a screenshot of the current browser page")
async def mcp_browser_screenshot(
    full_page: bool = Field(default=False, description="Whether to capture the full page or just viewport")
) -> str:
    """Take a screenshot of the current page."""
    result = await browser_screenshot(full_page)
    return str(result)


@mcp.tool(description="List all open browser tabs")
async def mcp_browser_list_tabs() -> str:
    """List all open browser tabs."""
    result = await browser_list_tabs()
    return str(result)


# ============================================================================
# EMAIL NOTIFICATION TOOLS
# ============================================================================

@mcp.tool(description="Send an email notification")
async def mcp_send_email(
    to_email: str = Field(description="Recipient email address"),
    subject: str = Field(description="Email subject"),
    body: str = Field(description="Email body content"),
    html: bool = Field(default=False, description="Whether body is HTML formatted"),
    cc: Optional[List[str]] = Field(default=None, description="Optional list of CC recipients")
) -> str:
    """Send an email notification."""
    result = await send_email(to_email, subject, body, html, cc)
    return str(result)


# ============================================================================
# INSTANT MESSAGING TOOLS
# ============================================================================

@mcp.tool(description="Send a Telegram message")
async def mcp_send_telegram_message(
    message: str = Field(description="Message text to send"),
    chat_id: Optional[str] = Field(default=None, description="Optional Telegram chat ID"),
    parse_mode: str = Field(default="HTML", description="Message parse mode (HTML, Markdown, or None)")
) -> str:
    """Send a Telegram message."""
    result = await send_telegram_message(message, chat_id, parse_mode)
    return str(result)


@mcp.tool(description="Send a Slack message via webhook")
async def mcp_send_slack_message(
    message: str = Field(description="Message text to send"),
    webhook_url: Optional[str] = Field(default=None, description="Optional Slack webhook URL"),
    channel: Optional[str] = Field(default=None, description="Optional channel to post to"),
    username: str = Field(default="Collaboration Agent", description="Bot username to display")
) -> str:
    """Send a Slack message."""
    result = await send_slack_message(message, webhook_url, channel, username)
    return str(result)


@mcp.tool(description="Send a Discord message via webhook")
async def mcp_send_discord_message(
    message: str = Field(description="Message text to send"),
    webhook_url: Optional[str] = Field(default=None, description="Optional Discord webhook URL"),
    username: str = Field(default="Collaboration Agent", description="Bot username to display")
) -> str:
    """Send a Discord message."""
    result = await send_discord_message(message, webhook_url, username)
    return str(result)


# ============================================================================
# HUMAN-IN-THE-LOOP TOOLS
# ============================================================================

@mcp.tool(description="Request approval from a human administrator")
async def mcp_request_admin_approval(
    request_message: str = Field(description="Message describing what needs approval"),
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional context information"),
    timeout_seconds: Optional[int] = Field(default=None, description="How long to wait for response"),
    urgent: bool = Field(default=False, description="Whether this is an urgent request")
) -> str:
    """Request approval from human administrator."""
    result = await request_admin_approval(request_message, context, timeout_seconds, urgent)
    return str(result)


@mcp.tool(description="Request input from a human administrator")
async def mcp_request_admin_input(
    prompt: str = Field(description="Question or prompt for the admin"),
    input_type: str = Field(default="text", description="Type of input expected (text, choice, number)"),
    options: Optional[List[str]] = Field(default=None, description="For choice type, list of available options"),
    timeout_seconds: Optional[int] = Field(default=None, description="How long to wait for response")
) -> str:
    """Request input from human administrator."""
    result = await request_admin_input(prompt, input_type, options, timeout_seconds)
    return str(result)


@mcp.tool(description="Respond to an admin approval request (admin use)")
async def mcp_respond_to_request(
    request_id: str = Field(description="ID of the request to respond to"),
    approved: bool = Field(description="Whether the request is approved"),
    admin_notes: Optional[str] = Field(default=None, description="Optional notes from the admin")
) -> str:
    """Admin response to an approval request."""
    result = await respond_to_request(request_id, approved, admin_notes)
    return str(result)


@mcp.tool(description="List all pending admin approval requests")
async def mcp_list_pending_requests() -> str:
    """List all pending approval requests."""
    result = await list_pending_requests()
    return str(result)


# ============================================================================
# TIMER TOOLS
# ============================================================================

@mcp.tool(description="Set a timer that will notify when completed")
async def mcp_set_timer(
    duration_seconds: int = Field(description="How long to wait before timer expires"),
    timer_name: Optional[str] = Field(default=None, description="Optional name for the timer"),
    callback_message: Optional[str] = Field(default=None, description="Message to return when timer expires"),
    callback_data: Optional[Dict[str, Any]] = Field(default=None, description="Optional data to include")
) -> str:
    """Set a timer that will notify when completed."""
    result = await set_timer(duration_seconds, timer_name, callback_message, callback_data)
    return str(result)


@mcp.tool(description="Set a recurring timer that repeats at intervals")
async def mcp_set_recurring_timer(
    interval_seconds: int = Field(description="Time between occurrences"),
    max_occurrences: Optional[int] = Field(default=None, description="Maximum number of times to repeat"),
    timer_name: Optional[str] = Field(default=None, description="Optional name for the timer"),
    callback_message: Optional[str] = Field(default=None, description="Message for each occurrence")
) -> str:
    """Set a recurring timer."""
    result = await set_recurring_timer(interval_seconds, max_occurrences, timer_name, callback_message)
    return str(result)


@mcp.tool(description="Cancel an active timer")
async def mcp_cancel_timer(
    timer_id: str = Field(description="ID of the timer to cancel")
) -> str:
    """Cancel an active timer."""
    result = await cancel_timer(timer_id)
    return str(result)


@mcp.tool(description="List all timers, optionally filtered by status")
async def mcp_list_timers(
    status: Optional[str] = Field(default=None, description="Optional status filter (active, expired, cancelled)")
) -> str:
    """List all timers."""
    result = await list_timers(status)
    return str(result)


@mcp.tool(description="Get status of a specific timer")
async def mcp_get_timer_status(
    timer_id: str = Field(description="ID of the timer to check")
) -> str:
    """Get timer status."""
    result = await get_timer_status(timer_id)
    return str(result)


# ============================================================================
# SERVER LIFECYCLE
# ============================================================================

# Run the server
if __name__ == "__main__":
    logger.info("Starting Collaboration Tools MCP Server...")
    
    # Load configuration
    config = load_config()
    logger.info(f"Configuration loaded: log_level={config.log_level}")
    
    # Load saved timers
    asyncio.run(_load_timers())
    
    try:
        # Run the MCP server
        mcp.run(transport="stdio")
    finally:
        # Cleanup on exit
        logger.info("Shutting down Collaboration Tools MCP Server...")
        asyncio.run(close_browser())
        logger.info("Server shutdown complete")
