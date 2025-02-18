from __future__ import annotations

import os
import json
import typing
import contextlib
import uuid

from anyio import Lock
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
    CreateChatCompletionRequest,
    ModelList,
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


async def get_llama_proxy():
    # NOTE: This double lock allows the currently streaming llama model to
    # check if any other requests are pending in the same thread and cancel
    # the stream if so.
    await llama_outer_lock.acquire()
    release_outer_lock = True
    try:
        await llama_inner_lock.acquire()
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


def prepare_request_resources(
    body: CreateChatCompletionRequest,
    llama_proxy: LlamaProxy,
    body_model: str,
    kwargs,
) -> llama_cpp.Llama:
    if llama_proxy is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not available",
        )
    llama = llama_proxy(body_model)
    # TODO: is this required?
    kwargs["logit_bias"] = body.logit_bias

    return llama


async def get_event_publisher(
    request: Request,
    inner_send_chan: MemoryObjectSendStream[typing.Any],
    body: CreateChatCompletionRequest,
    body_model: str | None,
    kwargs,
):
    server_settings = next(get_server_settings())
    async with contextlib.asynccontextmanager(get_llama_proxy)() as llama_proxy:
        llama = prepare_request_resources(body, llama_proxy, body_model, kwargs)
        async with inner_send_chan:
            try:
                iterator = await run_in_threadpool(llama.create_chat_completion, **kwargs)
                async for chunk in iterate_in_threadpool(iterator):
                    chunk["system_fingerprint"] = server_settings.system_fingerprint
                    await inner_send_chan.send(dict(data=json.dumps(chunk)))
                    if await request.is_disconnected():
                        raise anyio.get_cancelled_exc_class()()
                await inner_send_chan.send(dict(data="[DONE]"))
            except anyio.get_cancelled_exc_class() as e:
                # TODO: do we need this?
                with anyio.move_on_after(1, shield=True):
                    raise e



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

    # FIXME: unused? Replace.
    # FIXME: restore logging.
    def close_transaction():
        if body.store:
            closing_event: ChatCompletionEvent = {
                "transaction_id": transaction_id,
                "transaction_type": "chat_completion",
                "store": body.store,
                "completion": json.dumps(completion_result),
            }
            transaction_logger(closing_event)

    body_model = body.model

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

    if kwargs.get("stream", False):
        send_chan, recv_chan = anyio.create_memory_object_stream(10)  # type: ignore
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(  # type: ignore
                get_event_publisher,
                request=request,
                inner_send_chan=send_chan,
                body=body,
                body_model=body_model,
                kwargs=kwargs,
            ),
            sep="\n",
            ping_message_factory=_ping_message_factory,  # type: ignore
        )

    async with contextlib.asynccontextmanager(get_llama_proxy)() as llama_proxy:
        llama = prepare_request_resources(body, llama_proxy, body_model, kwargs)

        if await request.is_disconnected():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Client closed request",
            )

        response = await run_in_threadpool(llama.create_chat_completion, **kwargs)
        response["system_fingerprint"] = _server_settings.system_fingerprint

        return response


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
