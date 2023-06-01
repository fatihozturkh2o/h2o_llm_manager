import json
from enum import Enum
from typing import Any, List, Optional, Union
import os
import openai
import requests
from gradio_client import Client
from requests import Response
from utils import split_sentences, value_or_default, is_dict_string, string_to_dict, LLMGenerateError

# ENV variables
llm_h2ogpt_gradio_url = os.environ.get("H2OGPT_GRADIO_URL", None)
llm_mlops_url = os.environ.get("MLOPS_LLM_URL", None)
llm_openai_api_key = os.environ.get("OPENAI_API_KEY", None)

# logs
enable_logs: bool = False
def print_log(message: str):
    if enable_logs:
        print(message)

class LLMSourceType(str, Enum):
    H2OGPT: str = "h2oGPT"
    MLOPS: str = "MLOps"
    OPENAI: str = "GPT-4"

    def __str__(self) -> str:
        return self.value


class BaseLLMSource:
    def __init__(self, name: Optional[str] = None):
        self.type: Optional[LLMSourceType] = None
        self.name: Optional[str] = name

    @property
    def name(self) -> Optional[str]:
        if self._name:
            return self._name
        else:
            return self.type

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def test_prompt(self) -> str:
        return "Hello!"

    @property
    def failed_connection_message(self) -> str:
        raise NotImplementedError

    @property
    def is_connected(self) -> bool:
        raise NotImplementedError

    def generate(self, prompt: str, bullet_text: bool = False) -> str:
        raise NotImplementedError

    @staticmethod
    def bullet_text(text: str) -> str:
        """Converting the text in a way that each sentence will start with a bullet point at a new line"""
        try:
            return " ".join([f"<li>{t}</li>" for t in split_sentences(text)])
        except Exception as e:
            # Return the input as is in case an error occurs
            print_log(e)
            return text


class H2OGPTLLMSource(BaseLLMSource):
    def __init__(
        self,
        name: Optional[str] = None,
        url: Optional[str] = None,
        prompt_type: Optional[str] = "human_bot",
        chat: bool = False,
        api_name: str = "/submit_nochat",
    ):
        self.type: LLMSourceType = LLMSourceType.H2OGPT
        self.name: Optional[str] = name
        self.url: str = url
        self.prompt_type: Optional[str] = prompt_type
        self.chat: bool = chat
        self.api_name: str = api_name

        # Config override
        self.url = value_or_default(self.url, llm_h2ogpt_gradio_url)

    def get_client(self) -> Client:
        return Client(self.url)

    @property
    def failed_connection_message(self) -> str:
        return f"""For the LLM source '<b>{self.type}</b>', the connection to generate llm response fails with the url: '{self.url}'.
        You can visit [Settings](?view:Settings) to update LLM settings."""

    @property
    def is_connected(self) -> bool:
        try:
            args = self.get_args(prompt=self.test_prompt)
            client = self.get_client()
            client.predict(*tuple(args), api_name=self.api_name)
            return True
        except Exception as e:
            print_log(e)
            return False

    def get_args(self, prompt: str, max_new_tokens: int = 50) -> List[Any]:
        iinput: str = ""
        context: str = ""
        prompt_nochat: str = ""
        iinput_nochat: str = ""
        stream_output: bool = False
        temperature: float = 0.1
        top_p: float = 0.75
        top_k: int = 40
        num_beams: int = 1
        min_new_tokens: int = 0
        early_stopping: bool = False
        max_time: int = 60  # seconds to wait for the response generation
        repetition_penalty: float = 1.0
        num_return_sequences: int = 1
        do_sample: bool = True
        langchain_mode: str = "Disabled"
        top_k_docs: int = 4
        document_choice: List[str] = ["All"]

        if not self.chat:
            # If chat is False, iinput_nochat and instruction_nochat are used during the submit
            prompt_nochat = prompt
            iinput_nochat = iinput

        return [
            prompt,  # used if chat = True
            iinput,  # used if chat = True
            context,  # used if chat = True
            stream_output,
            self.prompt_type,
            temperature,
            top_p,
            top_k,
            num_beams,
            max_new_tokens,
            min_new_tokens,
            early_stopping,
            max_time,
            repetition_penalty,
            num_return_sequences,
            do_sample,
            self.chat,
            prompt_nochat,  # used if chat = False
            iinput_nochat,  # used if chat = False
            langchain_mode,
            top_k_docs,
            document_choice,
        ]

    def generate(
        self,
        prompt: str,
        bullet_text: bool = False,
    ) -> Optional[str]:

        if not self.is_connected:
            raise LLMGenerateError(text=self.failed_connection_message)

        args = self.get_args(prompt=prompt, max_new_tokens=1000)
        client = self.get_client()
        print_log(f"Prompt Type: {self.prompt_type}\nChat: {self.chat}\nPrompt: {prompt}")
        response = client.predict(*tuple(args), api_name=self.api_name)
        if is_dict_string(response):
            response = string_to_dict(response)["response"]
        if bullet_text:
            response = self.bullet_text(response)
        print_log(f"\nResponse: {response}")
        return response


class MLOpsLLMSource(BaseLLMSource):
    def __init__(
        self,
        name: Optional[str] = None,
        url: Optional[str] = None,
    ):
        self.type: LLMSourceType = LLMSourceType.MLOPS
        self.name: Optional[str] = name
        self.url: str = url
        # Config override
        self.url = value_or_default(self.url, llm_mlops_url)

    @property
    def failed_connection_message(self) -> str:
        return f"""For the LLM source '<b>{self.type}</b>', the connection to generate llm response fails with the url: '{self.url}'.
        You can visit [Settings](?view:Settings) to update LLM settings."""

    @staticmethod
    def is_response_successful(response: Response) -> bool:
        """To check if a Response object was succesfully returned or not"""
        successful: bool = response.status_code >= 200 and response.status_code < 300
        if not successful:
            print_log(f"{response.status_code}: {response.reason}")
        return successful

    def parse_response(self, response: Response) -> Optional[str]:
        """Reading the response text based on the request URL"""
        if not self.is_response_successful(response):
            return None

        url_ending: str = self.url.split("/")[-1]
        if url_ending == "score":
            return json.loads(response.json()["score"][0][0])["generated_text"][0]
        else:  # "generate_byte_stream"
            return response.text

    @property
    def is_connected(self) -> bool:
        try:
            r = requests.post(
                url=self.url, json={"fields": ["input"], "rows": [[self.test_prompt]]}
            )
            return self.is_response_successful(r)
        except Exception as e:
            print_log(e)
            return False

    def generate(
        self,
        prompt: str,
        bullet_text: bool = False,
    ) -> Optional[str]:

        if not self.is_connected:
            raise LLMGenerateError(text=self.failed_connection_message)

        print_log(f"Prompt: {prompt}")
        data = {"fields": ["input"], "rows": [[prompt]]}
        r = requests.post(url=self.url, json=data)
        response = self.parse_response(r)
        if bullet_text:
            response = self.bullet_text(response)
        print_log(f"\nResponse: {response}")
        return response


class OpenAILLMSource(BaseLLMSource):
    def __init__(
        self,
        name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.type: LLMSourceType = LLMSourceType.OPENAI
        self.name: Optional[str] = name
        self.api_key: Optional[str] = api_key
        # Config override
        self.api_key = value_or_default(self.api_key, llm_openai_api_key)

    @property
    def failed_connection_message(self) -> str:
        return f"""For the LLM source '<b>{self.type}</b>', the connection to the OpenAI fails. Make sure you are passing the correct API key.
        You can visit [Settings](?view:Settings) to update LLM settings."""

    @property
    def is_connected(self) -> bool:
        openai.api_key = self.api_key
        try:
            openai.Model.list()
            return True
        except openai.error.AuthenticationError as e:
            print_log(e)
            return False

    def generate(
        self,
        prompt: str,
        bullet_text: bool = False,
    ) -> Optional[str]:
        openai.api_key = self.api_key

        if not self.is_connected:
            raise LLMGenerateError(text=self.failed_connection_message)

        max_tokens: int = 500
        temperature: float = 0.1
        frequency_penalty: float = 1.2

        print_log(f"Prompt: {prompt}")

        message = [{"role": "user", "content": prompt}]
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
        )
        response = completion["choices"][0]["message"]["content"].strip()
        if bullet_text:
            response = self.bullet_text(response)
        print_log(f"\nResponse: {response}")
        return response
