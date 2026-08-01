"""Start the ChemPilot FastAPI inference service."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from chempilot.api.app import (
    create_app,
)
from chempilot.service.registry import (
    DEFAULT_INFERENCE_CONFIG_PATH,
    ModelRegistry,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Start the ChemPilot unified "
            "inference API."
        )
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Listening address. The default "
            "only accepts local connections."
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=(
            DEFAULT_INFERENCE_CONFIG_PATH
        ),
        help=(
            "Path to the inference YAML "
            "configuration."
        ),
    )

    parser.add_argument(
        "--log-level",
        choices=[
            "critical",
            "error",
            "warning",
            "info",
            "debug",
        ],
        default="info",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if not (
        1 <= arguments.port <= 65535
    ):
        raise ValueError(
            "Port must be between "
            "1 and 65535."
        )

    logging.basicConfig(
        level=getattr(
            logging,
            arguments.log_level.upper(),
        ),
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    if arguments.host in {
        "0.0.0.0",
        "::",
    }:
        logging.warning(
            "ChemPilot is being exposed on "
            "all network interfaces. Add "
            "authentication and a reverse "
            "proxy before public deployment."
        )

    registry = ModelRegistry(
        config_path=arguments.config
    )

    application = create_app(
        registry
    )

    uvicorn.run(
        application,
        host=arguments.host,
        port=arguments.port,
        log_level=(
            arguments.log_level
        ),
    )


if __name__ == "__main__":
    main()