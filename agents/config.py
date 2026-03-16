import logging
import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_xai import ChatXAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pathlib import Path
from os import PathLike


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

logger = logging.getLogger(__name__)
#AGENT_NAME = "request_parser_agent"

LLM_MODELS = {
    "openai": ChatOpenAI(model="gpt-5.4-2026-03-05", temperature=0),
    "grok": ChatXAI(model="grok-4-1-fast-reasoning", temperature=0),
    "anthropic": ChatAnthropic(model="claude-opus-4-6", temperature=0),
    "google": ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", temperature=0),
}

MODEL = LLM_MODELS["openai"] # test different models

INTEL_BASE_URL = "http://127.0.0.1:8000"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLANNER_PROMPT_PATH = PROJECT_ROOT / "agents" / "prompts" / "planner_prompt.txt"
SEMANTIC_MODELS_PATH = PROJECT_ROOT / "semantic" / "semantic_models.yml"
FORECAST_PROMPT_PATH = PROJECT_ROOT / "agents" / "prompts" / "forecast_implications.txt"
FUSION_PROMPT_PATH = PROJECT_ROOT / "agents" / "prompts" / "fusion_prompt.txt"