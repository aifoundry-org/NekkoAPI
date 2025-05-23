from __future__ import annotations

from typing import List, Optional, Union, Dict, Any
from typing_extensions import Annotated, TypedDict, Literal, NotRequired

from pydantic import BaseModel, Field

import llama_cpp


model_field = Field(
    description="The model to use for generating completions.", default=None
)

max_tokens_field = Field(
    default=16, ge=1, description="The maximum number of tokens to generate."
)

temperature_field = Field(
    default=1.0,
    ge=0.0,
    le=2.0,
    description="Adjust the randomness of the generated text.\n\n"
    + "Temperature is a hyperparameter that controls the randomness of the generated text. It affects the probability distribution of the model's output tokens. A higher temperature (e.g., 1.5) makes the output more random and creative, while a lower temperature (e.g., 0.5) makes the output more focused, deterministic, and conservative. The default value is 1, which provides a balance between randomness and determinism. At the extreme, a temperature of 0 will always pick the most likely next token, leading to identical outputs in each run. We recommend to alter this or top_p, but not both.",
)

top_p_field = Field(
    default=1.0,
    ge=0.0,
    le=1.0,
    description="Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P.\n\n"
    + "Top-p sampling, also known as nucleus sampling, is another text generation method that selects the next token from a subset of tokens that together have a cumulative probability of at least p. This method provides a balance between diversity and quality by considering both the probabilities of tokens and the number of tokens to sample from. A higher value for top_p (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. We recomment altering this or temperature, but not both.",
)

min_p_field = Field(
    default=0.05,
    ge=0.0,
    le=1.0,
    description="Sets a minimum base probability threshold for token selection.\n\n"
    + "The Min-P sampling method was designed as an alternative to Top-P, and aims to ensure a balance of quality and variety. The parameter min_p represents the minimum probability for a token to be considered, relative to the probability of the most likely token. For example, with min_p=0.05 and the most likely token having a probability of 0.9, logits with a value less than 0.045 are filtered out.",
)

stop_field = Field(
    default=None,
    description="A list of tokens at which to stop generation. If None, no stop tokens are used.",
)

stream_field = Field(
    default=False,
    description="Whether to stream the results as they are generated. Useful for chatbots.",
)

top_k_field = Field(
    default=40,
    ge=0,
    description="Limit the next token selection to the K most probable tokens.\n\n"
    + "Top-k sampling is a text generation method that selects the next token only from the top k most likely tokens predicted by the model. It helps reduce the risk of generating low-probability or nonsensical tokens, but it may also limit the diversity of the output. A higher value for top_k (e.g., 100) will consider more tokens and lead to more diverse text, while a lower value (e.g., 10) will focus on the most probable tokens and generate more conservative text.",
)

repeat_penalty_field = Field(
    default=1.1,
    ge=0.0,
    description="A penalty applied to each token that is already generated. This helps prevent the model from repeating itself.\n\n"
    + "Repeat penalty is a hyperparameter used to penalize the repetition of token sequences during text generation. It helps prevent the model from generating repetitive or monotonous text. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient.",
)

presence_penalty_field = Field(
    default=0.0,
    ge=-2.0,
    le=2.0,
    description="Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.",
)

frequency_penalty_field = Field(
    default=0.0,
    ge=-2.0,
    le=2.0,
    description="Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.",
)

mirostat_mode_field = Field(
    default=0,
    ge=0,
    le=2,
    description="Enable Mirostat constant-perplexity algorithm of the specified version (1 or 2; 0 = disabled)",
)

mirostat_tau_field = Field(
    default=5.0,
    ge=0.0,
    le=10.0,
    description="Mirostat target entropy, i.e. the target perplexity - lower values produce focused and coherent text, larger values produce more diverse and less coherent text",
)

mirostat_eta_field = Field(
    default=0.1, ge=0.001, le=1.0, description="Mirostat learning rate"
)

grammar = Field(
    default=None,
    description="A CBNF grammar (as string) to be used for formatting the model's output.",
)


JsonType = Union[None, int, str, bool, List[Any], Dict[str, Any]]


class FormatJsonSchema(TypedDict):
    description: Annotated[Optional[str], Field(
        default=None,
        description="A description of what the response format is for."
    )]
    name: Annotated[str, Field(description="The name of the response format.")]
    schema: Annotated[Optional[JsonType], Field(
        default=None,
        description="The schema for the response format, described as a JSON Schema object."
    )]
    strict: Annotated[Optional[bool], Field(
        default=False,
        description="Whether to enable strict schema adherence when generating the output. (ignored)"
    )]


class ResponseFormatText(TypedDict):
    type: Annotated[Literal["text"], Field(
        description="TODO"
    )]


class ResponseFormatJsonObject(TypedDict):
    type: Annotated[Literal["json_object"], Field(
        description="TODO"
    )]


# TODO: use the type of JsonSchema to describe JsonSchema (yeah, it
# is self referential...)
class ResponseFormatJsonSchema(TypedDict):
    type: Annotated[Literal["json_schema"], Field(description="TODO")]
    json_schema: Annotated[FormatJsonSchema, Field(description="TODO")]


class StreamOptions(BaseModel):
    include_usage: Optional[bool] = Field(
        default=None,
        description="If true, send additional chunk before [DONE] with usage information."
    )


class CreateCompletionRequest(BaseModel):
    prompt: Union[str, List[str]] = Field(
        default="", description="The prompt to generate completions for."
    )
    suffix: Optional[str] = Field(
        default=None,
        description="A suffix to append to the generated text. If None, no suffix is appended. Useful for chatbots.",
    )
    max_tokens: Optional[int] = Field(
        default=16, ge=0, description="The maximum number of tokens to generate."
    )
    temperature: float = temperature_field
    top_p: float = top_p_field
    min_p: float = min_p_field
    echo: bool = Field(
        default=False,
        description="Whether to echo the prompt in the generated text. Useful for chatbots.",
    )
    stop: Optional[Union[str, List[str]]] = stop_field
    stream: bool = stream_field
    logprobs: Optional[int] = Field(
        default=None,
        ge=0,
        description="The number of logprobs to generate. If None, no logprobs are generated.",
    )
    presence_penalty: Optional[float] = presence_penalty_field
    frequency_penalty: Optional[float] = frequency_penalty_field
    logit_bias: Optional[Dict[str, float]] = Field(None)
    seed: Optional[int] = Field(None)

    # ignored or currently unsupported
    model: Optional[str] = model_field
    n: Optional[int] = 1
    best_of: Optional[int] = 1
    user: Optional[str] = Field(default=None)

    # llama.cpp specific parameters
    top_k: int = top_k_field
    repeat_penalty: float = repeat_penalty_field
    mirostat_mode: int = mirostat_mode_field
    mirostat_tau: float = mirostat_tau_field
    mirostat_eta: float = mirostat_eta_field
    grammar: Optional[str] = None


class CreateEmbeddingRequest(BaseModel):
    model: Optional[str] = model_field
    input: Union[str, List[str]] = Field(description="The input to embed.")
    user: Optional[str] = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "input": "The food was delicious and the waiter...",
                }
            ]
        }
    }


class CreateChatCompletionRequest(BaseModel):
    messages: List[llama_cpp.ChatCompletionRequestMessage] = Field(
        default=[], description="A list of messages to generate completions for."
    )
    functions: Optional[List[llama_cpp.ChatCompletionFunction]] = Field(
        default=None,
        description="A list of functions to apply to the generated completions.",
        deprecated="Deprecated in favor of `tools`.",
    )
    function_call: Optional[llama_cpp.ChatCompletionRequestFunctionCall] = Field(
        default=None,
        description="A function to apply to the generated completions.",
        deprecated="Deprecated in favor of `tool_choice`.",
    )
    tools: Optional[List[llama_cpp.ChatCompletionTool]] = Field(
        default=None,
        description="A list of tools to apply to the generated completions.",
    )
    tool_choice: Optional[llama_cpp.ChatCompletionToolChoiceOption] = Field(
        default=None,
        description="A tool to apply to the generated completions.",
    )  # TODO: verify
    max_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens to generate. Defaults to inf",
        deprecated="Deprecated in favor of max_completion_tokens",
    )
    max_completion_tokens: Optional[int] = Field(
        gt=0,
        default=None,
        description="An upper bound for the number of tokens that can be generated for a completion. Defaults to inf",
    )
    logprobs: Optional[bool] = Field(
        default=False,
        description="Whether to output the logprobs or not. Default is True",
    )
    top_logprobs: Optional[int] = Field(
        default=None,
        ge=0,
        description="The number of logprobs to generate. If None, no logprobs are generated. logprobs need to set to True.",
    )
    temperature: float = temperature_field
    top_p: float = top_p_field
    min_p: float = min_p_field
    stop: Optional[Union[str, List[str]]] = stop_field
    stream: bool = stream_field
    stream_options: Optional[StreamOptions] = Field(None)
    presence_penalty: Optional[float] = presence_penalty_field
    frequency_penalty: Optional[float] = frequency_penalty_field
    logit_bias: Optional[Dict[str, float]] = Field(None)
    seed: Optional[int] = Field(None)
    response_format: Optional[Union[ResponseFormatText, ResponseFormatJsonObject, ResponseFormatJsonSchema]] = Field(
        default=None,
        description="An object specifying the format that the model must output."
    )

    model: str = model_field

    # ignored or currently unsupported
    n: Optional[int] = Field(default=1, description="TODO")

    user: Optional[str] = Field(
        default=None,
        description="A unique identifier for the end-user making the request."
    )

    metadata: Optional[JsonType] = Field(
        default=None,
        description="metadata attached to request."
    )

    store: Optional[Union[bool, str]] = Field(
        default=None,
        description="Store the result of chat completion for later use if true. If value is string, attach that string as a label to the stored completion."
    )

    # llama.cpp specific parameters
    top_k: int = top_k_field
    repeat_penalty: float = repeat_penalty_field


class ModelData(TypedDict):
    id: str
    object: Literal["model"]
    owned_by: str
    permissions: List[str]


class ModelList(TypedDict):
    object: Literal["list"]
    data: List[ModelData]


class TokenizeInputRequest(BaseModel):
    model: Optional[str] = model_field
    input: str = Field(description="The input to tokenize.")

    model_config = {
        "json_schema_extra": {"examples": [{"input": "How many tokens in this query?"}]}
    }


class TokenizeInputResponse(BaseModel):
    tokens: List[int] = Field(description="A list of tokens.")

    model_config = {"json_schema_extra": {
        "example": {"tokens": [123, 321, 222]}}}


class TokenizeInputCountResponse(BaseModel):
    count: int = Field(description="The number of tokens in the input.")

    model_config = {"json_schema_extra": {"example": {"count": 5}}}


class DetokenizeInputRequest(BaseModel):
    model: Optional[str] = model_field
    tokens: List[int] = Field(description="A list of toekns to detokenize.")

    model_config = {"json_schema_extra": {
        "example": [{"tokens": [123, 321, 222]}]}}


class DetokenizeInputResponse(BaseModel):
    text: str = Field(description="The detokenized text.")

    model_config = {
        "json_schema_extra": {"example": {"text": "How many tokens in this query?"}}
    }
