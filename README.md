## H2O LLM Manager & LLM Sources
- LLMSources provide a convenient way to interact with various LLM sources and endpoints.
- LLMManager serves as a central hub for orchestrating diverse LLM Sources from one unified platform.

### **LLMSources**
Currently, 3 different LLM sources are supported. You can checkout the test.py for more examples.

#### **H2OGPTLLMSource**
Supports gradio endpoints being created via [h2oGPT](https://github.com/h2oai/h2ogpt)

```python
from sources import H2OGPTLLMSource

url = "https://gpt.h2o.ai/"
h2ogpt_llm = H2OGPTLLMSource(url=url)
response = h2ogpt_llm.generate(prompt="Why 42?")
print(f"{h2ogpt_llm.name} response: {response}")

# Output: h2oGPT response: 42 is the answer to the ultimate question of life, the universe, and everything.
```

#### **MLOpsLLMSource**
Supports endpoints being created via MLOPs

```python
from sources import MLOpsLLMSource

url = "https://model.cloud-qa.h2o.ai/97c39173-0590-4fa9-bee4-77c3af8754dc/model/score"
mlops_llm = MLOpsLLMSource(name="h2ogpt-gm-7b", url=url)
response = mlops_llm.generate(prompt="Why 42?")
print(f"{mlops_llm.name} response: {response}")

# Output: h2ogpt-20b response: 42 is a number that has been used in many different contexts throughout history, including in literature, 
# mathematics, and popular culture. Here are a few reasons why 42 might be considered significant:
# ...
```

#### **OpenAILLMSource**
Supports OpenAI models via OpenAI API

```python
import os

from sources import OpenAILLMSource

llm_openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_llm = OpenAILLMSource(name="GPT-4", api_key=llm_openai_api_key)
response = openai_llm.generate(prompt=Why 42?")
print(f"{openai_llm.name} response: {response}")
# Output: GPT-4 response: 42 is a number that has become famous due to its appearance in the science fiction series "The Hitchhiker's Guide to the
# Galaxy" by Douglas Adams. In the story, a group of hyper-intelligent beings builds a supercomputer named Deep Thought to find the ultimate answer to
# life, the universe, and everything. After much anticipation, Deep Thought reveals that the answer is 42.
```


### **LLMManager**

You can checkout the test_maanger.py for more examples.

```python
import os

from manager import LLMManager
from sources import BaseLLMSource, H2OGPTLLMSource, MLOpsLLMSource

h2ogpt_llm = H2OGPTLLMSource(name="h2ogpt-12b", url="https://gpt.h2o.ai/")
mlops_llm = MLOpsLLMSource(name="h2ogpt-gm-7b", url="https://model.cloud-qa.h2o.ai/97c39173-0590-4fa9-bee4-77c3af8754dc/model/score")

prompt = "Why 42?"
llm_manager = LLMManager(sources=[h2ogpt_llm, mlops_llm])
llm_manager.set_active_source("h2oGPT")

for i, source in enumerate(llm_manager.sources):
    print(f"LLM Source {i}: {source.name}")

response = llm_manager.generate(prompt)
print(f"{llm_manager.active_source.name} response: {response}")

# Output: 
# LLM Source 0: h2ogpt-12b
# LLM Source 1: h2ogpt-gm-7b
# h2ogpt-12b response: 42 is the answer to the ultimate question of life, the universe, and everything.
```
