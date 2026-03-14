import os
import sys
from pathlib import Path

# Add project root so "agents" package can be imported from any CWD
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_bolt.authorization import AuthorizeResult
from slack_sdk import WebClient
from dotenv import load_dotenv
import logging
import re
from agents.graph import run_agent


load_dotenv()

# Socket Mode needs two tokens:
# - Bot token (xoxb-...) for the App (posting, API calls)
# - App-level token (xapp-...) for the WebSocket connection only
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN") or os.getenv("SLACK_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not SLACK_BOT_TOKEN:
    raise ValueError("Set SLACK_BOT_TOKEN (or SLACK_TOKEN) in .env to your Bot User OAuth Token (xoxb-...)")
if not SLACK_APP_TOKEN:
    raise ValueError(
        "Set SLACK_APP_TOKEN in .env to your App-Level Token (xapp-...) for Socket Mode. "
        "Create it at https://api.slack.com/apps → your app → Socket Mode → Enable → Generate."
    )

# When SLACK_CLIENT_ID/SECRET are set, Bolt uses OAuth installation store and ignores token=.
# Provide authorize so we always use our bot token (single-workspace Socket Mode).
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
if not SLACK_SIGNING_SECRET:
    raise ValueError("Set SLACK_SIGNING_SECRET in .env (from api.slack.com → your app → Basic Information).")


def _authorize(enterprise_id, team_id, user_id, client: WebClient, logger):
    return AuthorizeResult.from_auth_test_response(
        auth_test_response=client.auth_test(token=SLACK_BOT_TOKEN),
        bot_token=SLACK_BOT_TOKEN,
    )


# Our bot's user ID — skip messages from ourselves to avoid replying in a loop
_bot_user_id = None


def _get_bot_user_id() -> str | None:
    global _bot_user_id
    if _bot_user_id is None:
        try:
            resp = WebClient(token=SLACK_BOT_TOKEN).auth_test()
            _bot_user_id = resp.get("user_id") or resp.get("bot_id")
        except Exception as e:
            logger.warning("Could not get bot user id: %s", e)
    return _bot_user_id


app = App(
    signing_secret=SLACK_SIGNING_SECRET,
    authorize=_authorize,
)

def clean_slack_text(text: str) -> str:
    # Remove Slack mention tokens like <@U123ABC>
    return re.sub(r"<@[\w]+>", "", text).strip()


def _run_agent_and_reply(user_text: str, say):
    """Take the user's question, call the graph agent, and reply with the result."""
    if not user_text or not user_text.strip():
        say("Send me a question, e.g. 'What are the latest developments?' or 'Summarize anomalies in the last 24 hours'.")
        return
    say("Working on it...")
    try:
        response = run_agent(user_text.strip())
        say(response or "Done. (No report generated.)")
    except Exception as e:
        logger.exception("Agent run failed: %s", e)
        say(f"Something went wrong: {e}")


@app.event("app_mention")
def handle_app_mention(body, say, logger):
    event = body.get("event", {})
    raw_text = event.get("text", "")
    user_text = clean_slack_text(raw_text)
    logger.info("App mention: %s", user_text)
    _run_agent_and_reply(user_text, say)


@app.event("message")
def handle_message(body, say, logger):
    """Handle DMs and channel messages: use the message text as user_request for the graph."""
    event = body.get("event", {})
    # Ignore bot messages (including our own) so we don't reply in a loop
    if event.get("bot_id"):
        return
    bot_uid = _get_bot_user_id()
    if bot_uid and event.get("user") == bot_uid:
        return
    # Ignore message subtypes (channel_join, etc.)
    if event.get("subtype") not in (None, ""):
        return

    raw_text = event.get("text", "")
    user_text = clean_slack_text(raw_text).strip()
    logger.info("Message: %s", user_text[:80])

    _run_agent_and_reply(user_text, say)

if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()