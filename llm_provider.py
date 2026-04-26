"""
Shared LLM factory for all agents.
Provider and model are controlled via .env:
    LLM_PROVIDER = claude | openai | gemini
"""

import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER_MODELS = {
    "claude": "claude-sonnet-4-6",
    "openai": "gpt-4o",
    "gemini": "gemini-2.0-flash",
}


def get_llm(tools: list | None = None):
    """Return a LangChain chat model for the configured provider, optionally with tools bound."""
    provider = os.getenv("LLM_PROVIDER", "claude").lower()

    if provider == "claude":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=PROVIDER_MODELS["claude"],
            temperature=0,
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=PROVIDER_MODELS["openai"],
            temperature=0,
            api_key=os.environ["OPENAI_API_KEY"],
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=PROVIDER_MODELS["gemini"],
            temperature=0,
            google_api_key=os.environ["GEMINI_API_KEY"],
        )

    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Choose: claude | openai | gemini")

    if tools:
        llm = llm.bind_tools(tools)

    return llm


def get_llm_with_schema(schema):
    """Return a structured-output LLM for the configured provider."""
    provider = os.getenv("LLM_PROVIDER", "claude").lower()

    if provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=PROVIDER_MODELS["claude"],
            temperature=0,
            api_key=os.environ["ANTHROPIC_API_KEY"],
        ).with_structured_output(schema)

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=PROVIDER_MODELS["openai"],
            temperature=0,
            api_key=os.environ["OPENAI_API_KEY"],
        ).with_structured_output(schema)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=PROVIDER_MODELS["gemini"],
            temperature=0,
            google_api_key=os.environ["GEMINI_API_KEY"],
        ).with_structured_output(schema)

    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Choose: claude | openai | gemini")


def current_provider() -> str:
    return os.getenv("LLM_PROVIDER", "claude").lower()


def current_model() -> str:
    return PROVIDER_MODELS.get(current_provider(), "unknown")
