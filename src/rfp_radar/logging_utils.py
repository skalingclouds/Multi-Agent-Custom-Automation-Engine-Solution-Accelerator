# logging_utils.py
"""Structured logging utilities for RFP Radar service."""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs JSON-structured log messages."""

    def __init__(
        self,
        service_name: str = "rfp-radar",
        include_timestamp: bool = True,
        include_extra: bool = True,
    ):
        """Initialize the structured formatter.

        Args:
            service_name: Name of the service to include in logs
            include_timestamp: Whether to include timestamp in output
            include_extra: Whether to include extra fields from log record
        """
        super().__init__()
        self.service_name = service_name
        self.include_timestamp = include_timestamp
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string.

        Args:
            record: The log record to format

        Returns:
            JSON-formatted log string
        """
        log_data: Dict[str, Any] = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "service": self.service_name,
        }

        if self.include_timestamp:
            log_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Add source location information
        log_data["source"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from the record
        if self.include_extra:
            extra_fields = {}
            # Standard LogRecord attributes to exclude
            standard_attrs = {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "taskName",
            }
            for key, value in record.__dict__.items():
                if key not in standard_attrs and not key.startswith("_"):
                    try:
                        # Ensure value is JSON serializable
                        json.dumps(value)
                        extra_fields[key] = value
                    except (TypeError, ValueError):
                        extra_fields[key] = str(value)

            if extra_fields:
                log_data["extra"] = extra_fields

        return json.dumps(log_data, default=str)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable formatter for development environments."""

    # ANSI color codes for different log levels
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True):
        """Initialize the human-readable formatter.

        Args:
            use_colors: Whether to use ANSI colors in output
        """
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record in a human-readable format.

        Args:
            record: The log record to format

        Returns:
            Human-readable formatted log string
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname
        message = record.getMessage()

        if self.use_colors:
            color = self.COLORS.get(level, "")
            level_str = f"{color}{level:8}{self.RESET}"
        else:
            level_str = f"{level:8}"

        formatted = f"[{timestamp}] {level_str} [{record.name}] {message}"

        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


def setup_logging(
    level: Optional[str] = None,
    structured: Optional[bool] = None,
    service_name: str = "rfp-radar",
    enable_azure_monitor: bool = True,
) -> logging.Logger:
    """Set up logging configuration for RFP Radar.

    This function configures the root logger and returns a logger for the
    rfp_radar package. It supports both structured (JSON) logging for
    production and human-readable logging for development.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to LOG_LEVEL env var or INFO.
        structured: Whether to use structured JSON logging.
                   Defaults to True in production (APP_ENV != 'dev').
        service_name: Service name to include in structured logs.
        enable_azure_monitor: Whether to attempt Azure Monitor integration.

    Returns:
        Logger instance for rfp_radar

    Example:
        >>> logger = setup_logging(level="DEBUG", structured=False)
        >>> logger.info("Starting RFP Radar")
        >>> logger.info("Processing RFP", extra={"rfp_id": "12345"})
    """
    # Determine log level
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO").upper()

    log_level = getattr(logging, level, logging.INFO)

    # Determine if structured logging should be used
    if structured is None:
        app_env = os.environ.get("APP_ENV", "prod")
        structured = app_env != "dev"

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Set formatter based on mode
    if structured:
        formatter = StructuredFormatter(service_name=service_name)
    else:
        formatter = HumanReadableFormatter(use_colors=True)

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Set up Azure Application Insights if available
    if enable_azure_monitor:
        _setup_azure_monitor()

    # Configure specific loggers to reduce noise
    _configure_third_party_loggers(log_level)

    # Get and return the rfp_radar logger
    logger = logging.getLogger("rfp_radar")
    logger.info(
        "Logging initialized",
        extra={
            "log_level": level,
            "structured": structured,
            "service": service_name,
        }
    )

    return logger


def _setup_azure_monitor() -> None:
    """Set up Azure Application Insights integration if available."""
    connection_string = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")

    if not connection_string:
        return

    try:
        from azure.monitor.opentelemetry import configure_azure_monitor

        configure_azure_monitor(
            connection_string=connection_string,
            logger_name="rfp_radar",
        )
        logging.getLogger("rfp_radar").info(
            "Azure Application Insights configured successfully"
        )
    except ImportError:
        logging.getLogger("rfp_radar").debug(
            "azure-monitor-opentelemetry not installed, "
            "Azure Monitor integration disabled"
        )
    except Exception as e:
        logging.getLogger("rfp_radar").warning(
            f"Failed to configure Azure Application Insights: {e}"
        )


def _configure_third_party_loggers(log_level: int) -> None:
    """Configure third-party loggers to reduce noise.

    Args:
        log_level: The current log level being used
    """
    # Quiet down noisy third-party loggers
    noisy_loggers = [
        "azure.core.pipeline.policies.http_logging_policy",
        "azure.identity",
        "urllib3",
        "requests",
        "httpx",
        "httpcore",
    ]

    # Set these to WARNING unless we're in DEBUG mode
    third_party_level = logging.WARNING if log_level > logging.DEBUG else log_level

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(third_party_level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the rfp_radar namespace.

    Args:
        name: The name of the logger (will be prefixed with 'rfp_radar.')

    Returns:
        A logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    if not name.startswith("rfp_radar"):
        name = f"rfp_radar.{name}"
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding extra fields to log messages.

    This allows adding contextual information that will be included
    in all log messages within the context.

    Example:
        >>> with LogContext(rfp_id="12345", source="govtribe"):
        ...     logger.info("Processing RFP")  # includes rfp_id and source
    """

    _context: Dict[str, Any] = {}

    def __init__(self, **kwargs: Any):
        """Initialize the log context with extra fields.

        Args:
            **kwargs: Key-value pairs to add to log context
        """
        self.new_context = kwargs
        self.old_context: Dict[str, Any] = {}

    def __enter__(self) -> "LogContext":
        """Enter the context and set new fields."""
        self.old_context = LogContext._context.copy()
        LogContext._context.update(self.new_context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context and restore old fields."""
        LogContext._context = self.old_context

    @classmethod
    def get_context(cls) -> Dict[str, Any]:
        """Get the current log context.

        Returns:
            Dictionary of current context fields
        """
        return cls._context.copy()


class ContextAdapter(logging.LoggerAdapter):
    """Logger adapter that automatically includes LogContext fields.

    Example:
        >>> logger = ContextAdapter(logging.getLogger("rfp_radar"))
        >>> with LogContext(rfp_id="12345"):
        ...     logger.info("Processing")  # automatically includes rfp_id
    """

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process the log message to include context fields.

        Args:
            msg: The log message
            kwargs: Additional keyword arguments

        Returns:
            Tuple of (message, kwargs) with context added
        """
        extra = kwargs.get("extra", {})
        extra.update(LogContext.get_context())
        kwargs["extra"] = extra
        return msg, kwargs
