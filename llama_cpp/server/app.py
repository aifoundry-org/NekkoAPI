from __future__ import annotations

import os
import json
import typing
import contextlib
import uuid

from threading import Lock
from functools import partial
from typing import Iterator, List, Optional, Union

import llama_cpp

import anyio
from anyio.streams.memory import MemoryObjectSendStream
from starlette.concurrency import run_in_threadpool, iterate_in_threadpool
from fastapi import Depends, FastAPI, APIRouter, Request, HTTPException, status, Body
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html
)
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from starlette_context.plugins import RequestIdPlugin  # type: ignore
from starlette_context.middleware import RawContextMiddleware

from llama_cpp.server.model import (
    LlamaProxy,
)
from llama_cpp.server.settings import (
    ConfigFileSettings,
    Settings,
    ModelSettings,
    ServerSettings,
)
from llama_cpp.server.types import (
    CreateCompletionRequest,
    CreateEmbeddingRequest,
    CreateChatCompletionRequest,
    ModelList,
    TokenizeInputRequest,
    TokenizeInputResponse,
    TokenizeInputCountResponse,
    DetokenizeInputRequest,
    DetokenizeInputResponse,
)
from llama_cpp.server.errors import RouteErrorHandler
from llama_cpp.server.transaction_logs import (
    setup_logger,
    ChatCompletionEvent
)


router = APIRouter(route_class=RouteErrorHandler)

_server_settings: Optional[ServerSettings] = None


def set_server_settings(server_settings: ServerSettings):
    global _server_settings
    _server_settings = server_settings


def get_server_settings():
    yield _server_settings


_llama_proxy: Optional[LlamaProxy] = None

llama_outer_lock = Lock()
llama_inner_lock = Lock()


def set_llama_proxy(model_settings: List[ModelSettings]):
    global _llama_proxy
    _llama_proxy = LlamaProxy(models=model_settings)


def get_llama_proxy():
    # NOTE: This double lock allows the currently streaming llama model to
    # check if any other requests are pending in the same thread and cancel
    # the stream if so.
    llama_outer_lock.acquire()
    release_outer_lock = True
    try:
        llama_inner_lock.acquire()
        try:
            llama_outer_lock.release()
            release_outer_lock = False
            yield _llama_proxy
        finally:
            llama_inner_lock.release()
    finally:
        if release_outer_lock:
            llama_outer_lock.release()


_ping_message_factory: typing.Optional[typing.Callable[[], bytes]] = None


def set_ping_message_factory(factory: typing.Callable[[], bytes]):
    global _ping_message_factory
    _ping_message_factory = factory


def create_app(
    settings: Settings | None = None,
    server_settings: ServerSettings | None = None,
    model_settings: List[ModelSettings] | None = None,
):
    config_file = os.environ.get("CONFIG_FILE", None)
    if config_file is not None:
        if not os.path.exists(config_file):
            raise ValueError(f"Config file {config_file} not found!")
        with open(config_file, "rb") as f:
            # Check if yaml file
            if config_file.endswith(".yaml") or config_file.endswith(".yml"):
                import yaml

                config_file_settings = ConfigFileSettings.model_validate_json(
                    json.dumps(yaml.safe_load(f))
                )
            else:
                config_file_settings = ConfigFileSettings.model_validate_json(f.read())
            server_settings = ServerSettings.model_validate(config_file_settings)
            model_settings = config_file_settings.models

    # TODO: remove settings argument altogether.
    if server_settings is None and model_settings is None:
        server_settings = ServerSettings.model_validate(settings)
        model_settings = [ModelSettings.model_validate(settings)]

    assert (
        server_settings is not None and model_settings is not None
    ), "server_settings and model_settings must be provided together"

    set_server_settings(server_settings)
    middleware = [Middleware(RawContextMiddleware, plugins=(RequestIdPlugin(),))]
    app = FastAPI(
        middleware=middleware,
        title="NekkoAPI",
        version=llama_cpp.__version__,
        root_path=server_settings.root_path,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    app.mount("/static", StaticFiles(directory="static"), name="static")

    assert model_settings is not None
    set_llama_proxy(model_settings=model_settings)

    if server_settings.disable_ping_events:
        set_ping_message_factory(lambda: bytes())

    app.state.transaction_logger = setup_logger()

    configure_openapi(app)

    return app


async def get_event_publisher(
    request: Request,
    inner_send_chan: MemoryObjectSendStream[typing.Any],
    iterator: Iterator[typing.Any],
    on_complete: typing.Optional[typing.Callable[[], None]] = None,
):
    server_settings = next(get_server_settings())
    interrupt_requests = (
        server_settings.interrupt_requests if server_settings else False
    )
    async with inner_send_chan:
        try:
            async for chunk in iterate_in_threadpool(iterator):
                await inner_send_chan.send(dict(data=json.dumps(chunk)))
                if await request.is_disconnected():
                    raise anyio.get_cancelled_exc_class()()
                if interrupt_requests and llama_outer_lock.locked():
                    await inner_send_chan.send(dict(data="[DONE]"))
                    raise anyio.get_cancelled_exc_class()()
            await inner_send_chan.send(dict(data="[DONE]"))
        except anyio.get_cancelled_exc_class() as e:
            print("disconnected")
            with anyio.move_on_after(1, shield=True):
                print(f"Disconnected from client (via refresh/close) {request.client}")
                raise e
        finally:
            if on_complete:
                on_complete()


# Setup Bearer authentication scheme
bearer_scheme = HTTPBearer(auto_error=False)


async def authenticate(
    settings: Settings = Depends(get_server_settings),
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
):
    # Skip API key check if it's not set in settings
    if settings.api_key is None:
        return True

    # check bearer credentials against the api_key
    if authorization and authorization.credentials == settings.api_key:
        # api key is valid
        return authorization.credentials

    # raise http error 401
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
    )


openai_v1_tag = "OpenAI V1"


# @router.post(
#     "/v1/completions",
#     summary="Completion",
#     dependencies=[Depends(authenticate)],
#     response_model=Union[
#         llama_cpp.CreateCompletionResponse,
#         str,
#     ],
#     responses={
#         "200": {
#             "description": "Successful Response",
#             "content": {
#                 "application/json": {
#                     "schema": {
#                         "anyOf": [
#                             {"$ref": "#/components/schemas/CreateCompletionResponse"}
#                         ],
#                         "title": "Completion response, when stream=False",
#                     }
#                 },
#                 "text/event-stream": {
#                     "schema": {
#                         "type": "string",
#                         "title": "Server Side Streaming response, when stream=True. "
#                         + "See SSE format: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format",  # noqa: E501
#                         "example": """data: {... see CreateCompletionResponse ...} \\n\\n data: ... \\n\\n ... data: [DONE]""",
#                     }
#                 },
#             },
#         }
#     },
#     tags=[openai_v1_tag],
# )
# @router.post(
#     "/v1/engines/copilot-codex/completions",
#     include_in_schema=False,
#     dependencies=[Depends(authenticate)],
#     tags=[openai_v1_tag],
# )
async def create_completion(
    request: Request,
    body: CreateCompletionRequest,
) -> llama_cpp.Completion:
    exit_stack = contextlib.ExitStack()
    llama_proxy = await run_in_threadpool(
        lambda: exit_stack.enter_context(contextlib.contextmanager(get_llama_proxy)())
    )
    if llama_proxy is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not available",
        )
    if isinstance(body.prompt, list):
        assert len(body.prompt) <= 1
        body.prompt = body.prompt[0] if len(body.prompt) > 0 else ""

    llama = llama_proxy(
        body.model
        if request.url.path != "/v1/engines/copilot-codex/completions"
        else "copilot-codex"
    )

    exclude = {
        "n",
        "best_of",
        "user",
    }
    kwargs = body.model_dump(exclude=exclude)

    if body.grammar is not None:
        kwargs["grammar"] = llama_cpp.LlamaGrammar.from_string(body.grammar)

    try:
        iterator_or_completion: Union[
            llama_cpp.CreateCompletionResponse,
            Iterator[llama_cpp.CreateCompletionStreamResponse],
        ] = await run_in_threadpool(llama, **kwargs)
    except Exception as err:
        exit_stack.close()
        raise err

    if isinstance(iterator_or_completion, Iterator):
        # EAFP: It's easier to ask for forgiveness than permission
        first_response = await run_in_threadpool(next, iterator_or_completion)

        # If no exception was raised from first_response, we can assume that
        # the iterator is valid and we can use it to stream the response.
        def iterator() -> Iterator[llama_cpp.CreateCompletionStreamResponse]:
            yield first_response
            yield from iterator_or_completion
            exit_stack.close()

        send_chan, recv_chan = anyio.create_memory_object_stream(10)  # type: ignore
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(  # type: ignore
                get_event_publisher,
                request=request,
                inner_send_chan=send_chan,
                iterator=iterator(),
                on_complete=exit_stack.close,
            ),
            sep="\n",
            ping_message_factory=_ping_message_factory,  # type: ignore
        )
    else:
        exit_stack.close()
        return iterator_or_completion


# @router.post(
#     "/v1/embeddings",
#     summary="Embedding",
#     dependencies=[Depends(authenticate)],
#     tags=[openai_v1_tag],
# )
async def create_embedding(
    request: CreateEmbeddingRequest,
    llama_proxy: LlamaProxy = Depends(get_llama_proxy),
):
    return await run_in_threadpool(
        llama_proxy(request.model).create_embedding,
        **request.model_dump(exclude={"user"}),
    )


@router.post(
    "/v1/chat/completions",
    summary="Chat",
    dependencies=[Depends(authenticate)],
    response_model=Union[llama_cpp.ChatCompletion, str],
    responses={
        "200": {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "schema": {
                        "anyOf": [
                            {
                                "$ref": "#/components/schemas/CreateChatCompletionResponse"
                            }
                        ],
                        "title": "Completion response, when stream=False",
                    }
                },
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "title": "Server Side Streaming response, when stream=True"
                        + "See SSE format: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format",  # noqa: E501
                        "example": """data: {... see CreateChatCompletionResponse ...} \\n\\n data: ... \\n\\n ... data: [DONE]""",
                    }
                },
            },
        }
    },
    tags=[openai_v1_tag],
)
async def create_chat_completion(
    request: Request,
    body: CreateChatCompletionRequest = Body(
        openapi_examples={
            "normal": {
                "summary": "Chat Completion",
                "value": {
                    "model": "llama",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "What is the capital of France?"},
                    ],
                },
            },
            "json_mode": {
                "summary": "JSON Mode",
                "value": {
                    "model": "llama",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Who won the world series in 2020"},
                    ],
                    "response_format": {"type": "json_object"},
                },
            },
            "tool_calling": {
                "summary": "Tool Calling",
                "value": {
                    "model": "llama",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Extract Jason is 30 years old."},
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "User",
                                "description": "User record",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "age": {"type": "number"},
                                    },
                                    "required": ["name", "age"],
                                },
                            },
                        }
                    ],
                    "tool_choice": {
                        "type": "function",
                        "function": {
                            "name": "User",
                        },
                    },
                },
            },
            "logprobs": {
                "summary": "Logprobs",
                "value": {
                    "model": "llama",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "What is the capital of France?"},
                    ],
                    "logprobs": True,
                    "top_logprobs": 10,
                },
            },
        }
    ),
) -> llama_cpp.ChatCompletion:
    transaction_id = str(uuid.uuid4())
    completion_event: ChatCompletionEvent = {
        "transaction_id": transaction_id,
        "transaction_type": "chat_completion",
    }
    transaction_logger = request.app.state.transaction_logger
    if body.user:
        completion_event["user"] = body.user
    if body.metadata:
        completion_event["metadata"] = json.dumps(body.metadata)
    if body.store:
        completion_event["store"] = body.store
        completion_event["messages"] = json.dumps(body.messages)
    transaction_logger(completion_event)

    # Will be used to store completion response if required.
    completion_result = []

    def close_transaction():
        if body.store:
            closing_event: ChatCompletionEvent = {
                "transaction_id": transaction_id,
                "transaction_type": "chat_completion",
                "store": body.store,
                "completion": json.dumps(completion_result),
            }
            transaction_logger(closing_event)

    # This is a workaround for an issue in FastAPI dependencies
    # where the dependency is cleaned up before a StreamingResponse
    # is complete.
    # https://github.com/tiangolo/fastapi/issues/11143
    exit_stack = contextlib.ExitStack()
    exit_stack.callback(close_transaction)
    llama_proxy = await run_in_threadpool(
        lambda: exit_stack.enter_context(contextlib.contextmanager(get_llama_proxy)())
    )
    if llama_proxy is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not available",
        )
    exclude = {
        "n",
        "user",
        "metadata",
        "store",
        "max_completion_tokens",
        "stream_options"
    }
    # TODO: use whitelisting and only include permitted fields.
    # TODO: only leave OpenAI API compatible fields.
    kwargs = body.model_dump(exclude=exclude)
    llama = llama_proxy(body.model)

    # LLama.cpp doesn't make distinction between "json_object" and "json_schema"
    # types.
    # Rename "json_schema" into "schema" to avoid touching llama.cpp
    if body.response_format is not None:
        if body.response_format["json_schema"] is not None:
            kwargs["response_format"]["type"] = "json_object"
            kwargs["response_format"]["schema"] = body.response_format["json_schema"].get("schema")
            del kwargs["response_format"]["json_schema"]

    # Override max_tokens with max_completion_tokens.
    if body.max_completion_tokens is not None:
        kwargs["max_tokens"] = body.max_completion_tokens

    if body.stream_options and body.stream_options.include_usage:
        kwargs["usage"] = True

    try:
        iterator_or_completion: Union[
            llama_cpp.ChatCompletion, Iterator[llama_cpp.ChatCompletionChunk]
        ] = await run_in_threadpool(llama.create_chat_completion, **kwargs)
    except Exception as err:
        exit_stack.close()
        raise err

    if isinstance(iterator_or_completion, Iterator):
        # EAFP: It's easier to ask for forgiveness than permission
        first_response = await run_in_threadpool(next, iterator_or_completion)

        # If no exception was raised from first_response, we can assume that
        # the iterator is valid and we can use it to stream the response.
        def iterator() -> Iterator[llama_cpp.ChatCompletionChunk]:
            first_response['system_fingerprint'] = _server_settings.system_fingerprint
            completion_result.append(first_response)
            yield first_response

            for entity in iterator_or_completion:
                entity['system_fingerprint'] = _server_settings.system_fingerprint
                completion_result.append(entity)
                yield entity

            exit_stack.close()

        send_chan, recv_chan = anyio.create_memory_object_stream(10)  # type: ignore
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(  # type: ignore
                get_event_publisher,
                request=request,
                inner_send_chan=send_chan,
                iterator=iterator(),
                on_complete=exit_stack.close,
            ),
            sep="\n",
            ping_message_factory=_ping_message_factory,  # type: ignore
        )
    else:
        completion_result.append(iterator_or_completion)
        exit_stack.close()
        iterator_or_completion['system_fingerprint'] = _server_settings.system_fingerprint
        return iterator_or_completion


@router.get(
    "/v1/models",
    summary="Models",
    dependencies=[Depends(authenticate)],
    tags=[openai_v1_tag],
)
async def get_models(
    llama_proxy: LlamaProxy = Depends(get_llama_proxy),
) -> ModelList:
    return {
        "object": "list",
        "data": [
            {
                "id": model_alias,
                "object": "model",
                "owned_by": "me",
                "permissions": [],
            }
            for model_alias in llama_proxy
        ],
    }


extras_tag = "Extras"


# @router.post(
#     "/extras/tokenize",
#     summary="Tokenize",
#     dependencies=[Depends(authenticate)],
#     tags=[extras_tag],
# )
async def tokenize(
    body: TokenizeInputRequest,
    llama_proxy: LlamaProxy = Depends(get_llama_proxy),
) -> TokenizeInputResponse:
    tokens = llama_proxy(body.model).tokenize(body.input.encode("utf-8"), special=True)

    return TokenizeInputResponse(tokens=tokens)


# @router.post(
#     "/extras/tokenize/count",
#     summary="Tokenize Count",
#     dependencies=[Depends(authenticate)],
#     tags=[extras_tag],
# )
async def count_query_tokens(
    body: TokenizeInputRequest,
    llama_proxy: LlamaProxy = Depends(get_llama_proxy),
) -> TokenizeInputCountResponse:
    tokens = llama_proxy(body.model).tokenize(body.input.encode("utf-8"), special=True)

    return TokenizeInputCountResponse(count=len(tokens))


# @router.post(
#     "/extras/detokenize",
#     summary="Detokenize",
#     dependencies=[Depends(authenticate)],
#     tags=[extras_tag],
# )
async def detokenize(
    body: DetokenizeInputRequest,
    llama_proxy: LlamaProxy = Depends(get_llama_proxy),
) -> DetokenizeInputResponse:
    text = llama_proxy(body.model).detokenize(body.tokens).decode("utf-8")

    return DetokenizeInputResponse(text=text)


def configure_openapi(app):
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title + " - Swagger UI",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url="/static/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui.css",
        )

    @app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
    async def swagger_ui_redirect():
        return get_swagger_ui_oauth2_redirect_html()

    @app.get("/redoc", include_in_schema=False)
    async def redoc_html():
        return get_redoc_html(
            openapi_url=app.openapi_url,
            title=app.title + " - ReDoc",
            redoc_js_url="/static/redoc.standalone.js",
        )
