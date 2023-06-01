import nltk
import ast

from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class LLMGenerateError(Exception):
    title: str = "LLM Generate Error"
    text: str = ""


def value_or_default(value: Optional[Any], default: Any) -> Any:
    return value if value is not None else default

def split_sentences(text: str) -> List[str]:
    # Use nltk to split the text into sentences
    nltk.download("punkt", quiet=True)
    sentences = nltk.sent_tokenize(text)
    return sentences

def is_dict_string(string: str) -> bool:
    """
    Checks if a string can be converted into a Python dict object.
    """
    try:
        return isinstance(ast.literal_eval(string), dict)
    except Exception:
        return False


def string_to_dict(string: str) -> dict:
    """
    Converts a string into a Python  dict object.
    """
    if is_dict_string(string):
        return ast.literal_eval(string)
    else:
        raise ValueError("String doesn't contain Python dict expression")
