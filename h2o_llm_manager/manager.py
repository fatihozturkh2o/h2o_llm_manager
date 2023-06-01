from typing import Any, List, Optional, Union
from sources import BaseLLMSource, H2OGPTLLMSource, MLOpsLLMSource, OpenAILLMSource

############################
#         LLMManager       #
############################


class LLMManager(object):
    def __init__(
        self,
        active_source: BaseLLMSource = H2OGPTLLMSource(),
    ) -> None:
        self.sources: List[BaseLLMSource] = [
            H2OGPTLLMSource(),
            MLOpsLLMSource(),
            OpenAILLMSource(),
        ]
        self.active_source: BaseLLMSource = active_source

    def get_source(
        self, source_name: Optional[str] = None, source_type: Optional[str] = None
    ) -> Optional[BaseLLMSource]:
        for source in self.sources:
            if source_type and source.type == source_type:
                return source
            if source_name and source.name == source_name:
                return source
        return None

    def set_active_source(self, value: Union[BaseLLMSource, str]):
        if isinstance(value, BaseLLMSource):
            self.active_source = value
        else:  # str
            self.active_source = self.get_source(source_type=value)

    def generate(
        self,
        prompt: str,
        bullet_text: bool = True,
    ) -> Optional[str]:

        print(f"Generating response with the LLM source: {self.active_source.name}...")
        return self.active_source.generate(prompt=prompt, bullet_text=bullet_text)
