"""Uvicorn entry point that loads local Qwen and the LoRA adapter at startup."""
from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from safety_gateway.api import create_app
from safety_gateway.model import LocalQwenAnnotator
from safety_gateway.pipeline import LocalSafetyPipeline, UnavailableSafetyPipeline
from safety_gateway.settings import GatewaySettings
from safety_gateway.vault import InMemoryMappingVault


LOGGER = logging.getLogger("easyteaching.safety_gateway")


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = GatewaySettings()
    try:
        LOGGER.info("Loading local safety model and adapter")
        annotator = LocalQwenAnnotator.load(
            model_dir=settings.model_dir,
            adapter_dir=settings.adapter_dir,
            max_input_tokens=settings.max_input_tokens,
            max_new_tokens=settings.max_new_tokens,
        )
        application.state.pipeline = LocalSafetyPipeline(
            annotator=annotator,
            vault=InMemoryMappingVault(settings.mapping_ttl_seconds),
        )
        LOGGER.info("Local safety gateway is ready")
    except Exception as error:
        application.state.pipeline = UnavailableSafetyPipeline()
        LOGGER.error("Local safety gateway is not ready (%s)", type(error).__name__)
    yield


app = create_app(lifespan=lifespan)
