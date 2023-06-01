import os
from sources import H2OGPTLLMSource, MLOpsLLMSource, OpenAILLMSource
from utils import value_or_default

llm_h2ogpt_gradio_url = os.environ.get("H2OGPT_GRADIO_URL", None)
llm_mlops_url = os.environ.get("MLOPS_LLM_URL", None)
llm_openai_api_key = os.environ.get("OPENAI_API_KEY", None)


def test_h2ogpt_llm(prompt: str):
    url = value_or_default(llm_h2ogpt_gradio_url, "https://gpt.h2o.ai/")
    h2ogpt_llm = H2OGPTLLMSource(name="h2ogpt-oasst1-512-12b", url=url)
    response = h2ogpt_llm.generate(prompt=prompt)
    print("*"*15)
    print(f"{h2ogpt_llm.name} response: {response}")

def test_mlops_llm(prompt):
    url = value_or_default(llm_h2ogpt_gradio_url, "https://model.cloud-qa.h2o.ai/97c39173-0590-4fa9-bee4-77c3af8754dc/model/score")
    mlops_llm = MLOpsLLMSource(name="h2ogpt-gm-oasst1-en-2048-open-llama-7b-preview-300bt-v2", url=url)
    response = mlops_llm.generate(prompt=prompt)
    print("*"*15)
    print(f"{mlops_llm.name} response: {response}")

def test_openai_llm(prompt):
    openai_llm = OpenAILLMSource(name="GPT-4", api_key=llm_openai_api_key)
    response = openai_llm.generate(prompt=prompt)
    print("*"*15)
    print(f"{openai_llm.name} response: {response}")

if __name__ == "__main__":
    prompt = "Why 42?"
    print(f"Prompt: {prompt}")
    test_h2ogpt_llm(prompt)
    test_mlops_llm(prompt)
    if llm_openai_api_key:
        test_openai_llm(prompt)

 