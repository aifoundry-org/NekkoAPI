"""Types and request signatures for OpenAI compatibility

NOTE: These types may change to match the OpenAI OpenAPI specification.

Based on the OpenAI OpenAPI specification:
https://github.com/openai/openai-openapi/blob/master/openapi.yaml

"""

from typing import Any, List, Optional, Dict, Union
from typing_extensions import Annotated, TypedDict, NotRequired, Literal

from pydantic import Field


# NOTE: Defining this correctly using annotations seems to break pydantic validation.
#       This is a workaround until we can figure out how to do this correctly
# JsonType = Union[None, int, str, bool, List["JsonType"], Dict[str, "JsonType"]]
JsonType = Union[None, int, str, bool, List[Any], Dict[str, Any]]


class EmbeddingUsage(TypedDict):
    prompt_tokens: int
    total_tokens: int


class Embedding(TypedDict):
    index: int
    object: str
    embedding: Union[List[float], List[List[float]]]


class CreateEmbeddingResponse(TypedDict):
    object: Literal["list"]
    model: str
    data: List[Embedding]
    usage: EmbeddingUsage


class CompletionLogprobs(TypedDict):
    text_offset: List[int]
    token_logprobs: List[Optional[float]]
    tokens: List[str]
    top_logprobs: List[Optional[Dict[str, float]]]


class CompletionChoice(TypedDict):
    text: str
    index: int
    logprobs: Optional[CompletionLogprobs]
    finish_reason: Optional[Literal["stop", "length"]]


# TODO: annotate
class CompletionUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CreateCompletionResponse(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: NotRequired[CompletionUsage]
    system_fingerprint: str


# TODO: annotate
class ChatCompletionResponseFunctionCall(TypedDict):
    name: str
    arguments: str


# TODO: annotate
class ChatCompletionResponseMessage(TypedDict):
    content: Optional[str]
    tool_calls: NotRequired["ChatCompletionMessageToolCalls"]
    role: Literal["assistant", "function"]  # NOTE: "function" may be incorrect here
    function_call: NotRequired[ChatCompletionResponseFunctionCall]  # DEPRECATED


class ChatCompletionFunction(TypedDict):
    # TODO?: annotate
    name: str
    description: NotRequired[str]
    parameters: Dict[str, JsonType]


# TODO: annotate
class ChatCompletionTopLogprobToken(TypedDict):
    token: str
    logprob: float
    bytes: Optional[List[int]]


# TODO: annotate
class ChatCompletionLogprobToken(ChatCompletionTopLogprobToken):
    token: str
    logprob: float
    bytes: Optional[List[int]]
    top_logprobs: List[ChatCompletionTopLogprobToken]


# TODO: annotate
class ChatCompletionLogprobs(TypedDict):
    content: Optional[List[ChatCompletionLogprobToken]]
    refusal: Optional[List[ChatCompletionLogprobToken]]


# TODO: annotate
class ChatCompletionResponseChoice(TypedDict):
    index: int
    message: "ChatCompletionResponseMessage"
    logprobs: Optional[ChatCompletionLogprobs]
    finish_reason: Optional[str]


# TODO: annotate
class CreateChatCompletionResponse(TypedDict):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: CompletionUsage
    system_fingerprint: str


# TODO: annotate
class ChatCompletionMessageToolCallChunkFunction(TypedDict):
    name: Optional[str]
    arguments: str


# TODO: annotate
class ChatCompletionMessageToolCallChunk(TypedDict):
    index: int
    id: NotRequired[str]
    type: Literal["function"]
    function: ChatCompletionMessageToolCallChunkFunction


class ChatCompletionStreamResponseDeltaEmpty(TypedDict):
    pass


class ChatCompletionStreamResponseDeltaFunctionCall(TypedDict):
    name: str
    arguments: str


class ChatCompletionStreamResponseDelta(TypedDict):
    content: NotRequired[Optional[str]]
    function_call: NotRequired[
        Optional[ChatCompletionStreamResponseDeltaFunctionCall]
    ]  # DEPRECATED
    tool_calls: NotRequired[Optional[List[ChatCompletionMessageToolCallChunk]]]
    role: NotRequired[Optional[Literal["system", "user", "assistant", "tool"]]]


class ChatCompletionStreamResponseChoice(TypedDict):
    index: int
    delta: Union[
        ChatCompletionStreamResponseDelta, ChatCompletionStreamResponseDeltaEmpty
    ]
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "function_call"]]
    logprobs: NotRequired[Optional[ChatCompletionLogprobs]]


class CreateChatCompletionStreamResponse(TypedDict):
    id: str
    model: str
    object: Literal["chat.completion.chunk"]
    created: int
    choices: List[ChatCompletionStreamResponseChoice]


class ChatCompletionFunctions(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: Dict[str, JsonType]  # TODO: make this more specific


class ChatCompletionFunctionCallOption(TypedDict):
    name: str


class ChatCompletionRequestResponseFormat(TypedDict):
    type: Literal["text", "json_object"]
    schema: NotRequired[
        JsonType
    ]  # https://docs.endpoints.anyscale.com/guides/json_mode/


class ChatCompletionRequestMessageContentPartText(TypedDict):
    type: Literal["text"]
    text: str


class ChatCompletionRequestMessageContentPartImageImageUrl(TypedDict):
    url: str
    detail: NotRequired[Literal["auto", "low", "high"]]


class ChatCompletionRequestMessageContentPartImage(TypedDict):
    type: Literal["image_url"]
    image_url: Union[str, ChatCompletionRequestMessageContentPartImageImageUrl]


ChatCompletionRequestMessageContentPart = Union[
    ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestMessageContentPartImage,
]


class ChatCompletionRequestDeveloperMessage(TypedDict):
    role: Literal["developer"]
    content: Optional[str]
    # TODO; add annotation about the field being ignored
    # TODO: move into other message types
    # TODO: add "developer" message type
    name: NotRequired[Optional[str]]


class ChatCompletionRequestSystemMessage(TypedDict):
    role: Literal["system"]
    content: Optional[str]
    # TODO; add annotation about the field being ignored
    # TODO: move into other message types
    # TODO: add "developer" message type
    name: NotRequired[Optional[str]]


class ChatCompletionRequestUserMessage(TypedDict):
    role: Literal["user"]
    content: Optional[Union[str, List[ChatCompletionRequestMessageContentPart]]]
    name: NotRequired[Optional[str]]


class ChatCompletionMessageToolCallFunction(TypedDict):
    name: str
    arguments: str


class ChatCompletionMessageToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: ChatCompletionMessageToolCallFunction


ChatCompletionMessageToolCalls = List[ChatCompletionMessageToolCall]


class ChatCompletionRequestAssistantMessageFunctionCall(TypedDict):
    name: str
    arguments: str


class ChatCompletionRequestAssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: NotRequired[str]
    name: NotRequired[Optional[str]]
    tool_calls: NotRequired[ChatCompletionMessageToolCalls]
    function_call: NotRequired[
        ChatCompletionRequestAssistantMessageFunctionCall
    ]  # DEPRECATED


class ChatCompletionRequestToolMessage(TypedDict):
    role: Literal["tool"]
    content: Optional[str]
    tool_call_id: str


class ChatCompletionRequestFunctionMessage(TypedDict):
    role: Literal["function"]
    content: Optional[str]
    name: str


ChatCompletionRequestMessage = Union[
    ChatCompletionRequestDeveloperMessage,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage,
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestToolMessage,
    ChatCompletionRequestFunctionMessage,
]


class ChatCompletionRequestFunctionCallOption(TypedDict):
    name: str


# TODO?: annotate?
ChatCompletionRequestFunctionCall = Union[
    Literal["none", "auto"], ChatCompletionRequestFunctionCallOption
]

ChatCompletionFunctionParameters = Dict[str, JsonType]  # TODO: make this more specific


# TODO?: annotate?
class ChatCompletionToolFunction(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: ChatCompletionFunctionParameters


class ChatCompletionTool(TypedDict):
    type: Literal["function"]
    function: ChatCompletionToolFunction


class ChatCompletionNamedToolChoiceFunction(TypedDict):
    name: str


class ChatCompletionNamedToolChoice(TypedDict):
    type: Literal["function"]
    function: ChatCompletionNamedToolChoiceFunction


ChatCompletionToolChoiceOption = Union[
    Literal["none", "auto", "required"], ChatCompletionNamedToolChoice
]


# NOTE: The following type names are not part of the OpenAI OpenAPI specification
# and will be removed in a future major release.

EmbeddingData = Embedding
CompletionChunk = CreateCompletionResponse
Completion = CreateCompletionResponse
CreateCompletionStreamResponse = CreateCompletionResponse
ChatCompletionMessage = ChatCompletionResponseMessage
ChatCompletionChoice = ChatCompletionResponseChoice
ChatCompletion = CreateChatCompletionResponse
ChatCompletionChunkDeltaEmpty = ChatCompletionStreamResponseDeltaEmpty
ChatCompletionChunkChoice = ChatCompletionStreamResponseChoice
ChatCompletionChunkDelta = ChatCompletionStreamResponseDelta
ChatCompletionChunk = CreateChatCompletionStreamResponse
ChatCompletionStreamResponse = CreateChatCompletionStreamResponse
ChatCompletionResponseFunction = ChatCompletionFunction
ChatCompletionFunctionCall = ChatCompletionResponseFunctionCall
