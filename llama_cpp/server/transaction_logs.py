from __future__ import annotations

import os
import platform
from time import time_ns

from typing import Annotated, Literal, Union
from typing_extensions import TypedDict, NotRequired

from opentelemetry.sdk._logs import LoggerProvider, LogRecord
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry._logs import NoOpLogger
from opentelemetry._logs.severity import SeverityNumber
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter


def setup_logger():
    instance = platform.node()
    resource = Resource(attributes={
        "service.name": "nekko_api",
        "service.instance.id": instance,
    })

    logger = NoOpLogger('nekko-api')

    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        logger_provider = LoggerProvider(resource=resource)

        otlp_exporter = OTLPLogExporter()

        logger_provider.add_log_record_processor(BatchLogRecordProcessor(otlp_exporter))

        logger = logger_provider.get_logger("nekko-api")

    def log(event):
        timestamp = time_ns()
        attributes = event
        record = LogRecord(
                timestamp=timestamp,
                observed_timestamp=timestamp,
                resource=resource,
                attributes=attributes,
                span_id=0,
                trace_id=0,
                trace_flags=0,
                severity_number=SeverityNumber(0),
            )
        logger.emit(record)

    return log


class ChatCompletionEvent(TypedDict):
    transaction_id: Annotated[str]
    transacton_type: Literal["chat_completion"]
    user: Annotated[NotRequired[str]]
    metadata: Annotated[NotRequired[str]]
    store: Annotated[NotRequired[Union[bool,str]]]
    completion: Annotated[NotRequired[str]]

