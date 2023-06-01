import os
from typing import List

from manager import LLMManager
from sources import BaseLLMSource, H2OGPTLLMSource, MLOpsLLMSource, OpenAILLMSource
from utils import value_or_default

# ENV variables
llm_h2ogpt_gradio_url = os.environ.get("H2OGPT_GRADIO_URL", None)
llm_mlops_url = os.environ.get("MLOPS_LLM_URL", None)
llm_openai_api_key = os.environ.get("OPENAI_API_KEY", None)


def get_llm_sources() -> List[BaseLLMSource]:
    url = value_or_default(llm_h2ogpt_gradio_url, "https://gpt.h2o.ai/")
    h2ogpt_llm = H2OGPTLLMSource(name="h2ogpt-oasst1-512-12b", url=url)

    url = value_or_default(
        llm_h2ogpt_gradio_url,
        "https://model.cloud-qa.h2o.ai/97c39173-0590-4fa9-bee4-77c3af8754dc/model/score",
    )
    mlops_llm = MLOpsLLMSource(
        name="h2ogpt-gm-oasst1-en-2048-open-llama-7b-preview-300bt-v2", url=url
    )

    llm_sources = [h2ogpt_llm, mlops_llm]
    if llm_openai_api_key:
        openai_llm = OpenAILLMSource(name="GPT-4", api_key=llm_openai_api_key)
        llm_sources.append(openai_llm)
    return llm_sources


def test_sources():
    print("\ntest_sources...")
    prompt = "Why 42?"
    llm_sources = get_llm_sources()
    llm_manager = LLMManager(sources=llm_sources)
    llm_manager.set_active_source("h2oGPT")

    print("Available LLM sources...")
    for i, source in enumerate(llm_manager.sources):
        print(f"LLM Source {i}: {source.name}")

    print(f"Active LLM Source: {llm_manager.active_source.name}")
    response = llm_manager.generate(prompt)
    print(f"Prompt: {prompt}")
    print(f"{llm_manager.active_source.name} response: {response}")


def test_add_source():
    print("\ntest_add_source...")
    h2ogpt_llm = H2OGPTLLMSource()
    llm_manager = LLMManager()
    print("Before - Available LLM sources...")
    for i, source in enumerate(llm_manager.sources):
        print(f"LLM Source {i}: {source.name}")
    llm_manager.add_source(h2ogpt_llm)
    print("After - Available LLM sources...")
    for i, source in enumerate(llm_manager.sources):
        print(f"LLM Source {i}: {source.name}")


def test_remove_source():
    print("\ntest_remove_source...")
    h2ogpt_llm = H2OGPTLLMSource()
    llm_manager = LLMManager(sources=[h2ogpt_llm])
    print("Before - Available LLM sources...")
    for i, source in enumerate(llm_manager.sources):
        print(f"LLM Source {i}: {source.name}")
    llm_manager.remove_source(h2ogpt_llm)
    print("After - Available LLM sources...")
    for i, source in enumerate(llm_manager.sources):
        print(f"LLM Source {i}: {source.name}")


if __name__ == "__main__":
    test_add_source()
    test_remove_source()
    test_sources()
