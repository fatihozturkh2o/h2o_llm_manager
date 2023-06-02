from typing import List, Optional, Union

from log import print_log
from sources import BaseLLMSource

############################
#         LLMManager       #
############################


class LLMManager(object):
    def __init__(
        self,
        sources: Optional[List[BaseLLMSource]] = None,
    ) -> None:
        self.sources: Optional[List[BaseLLMSource]] = sources
        self.active_source: Optional[BaseLLMSource] = None
        if self.sources is None:
            self.sources = []

    def add_source(self, source: BaseLLMSource):
        if self.get_source(source_name=source.name):
            """Already exists"""
            return

        self.sources.append(source)

    def remove_source(self, source: Union[BaseLLMSource, str]):
        if not self.get_source(source_name=source.name):
            raise ValueError(f"LLM Source {source.name} doesn't exist")

        if isinstance(source, BaseLLMSource):
            self.sources.remove(source)
        else:  # str
            self.sources.remove(self.get_source(source_name=source))

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
            self.active_source = self.get_source(source_name=value)

    def generate(
        self,
        prompt: str,
        bullet_text: bool = False,
    ) -> Optional[str]:

        if self.active_source is None:
            raise ValueError("Active source needs to be set before generating any response")

        print_log(f"Generating response with the LLM source: {self.active_source.name}...")
        return self.active_source.generate(prompt=prompt, bullet_text=bullet_text)
