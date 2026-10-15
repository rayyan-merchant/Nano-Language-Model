"""Strict public contracts; filesystem and checkpoint upload fields are forbidden."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

ModelID = Literal['e02', 'b01', 'f01']


class Settings(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    prompt: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=128, ge=1, le=256)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    seed: int | None = Field(default=42, ge=0, le=2147483647)
    greedy: bool = False

    @field_validator('prompt')
    @classmethod
    def valid_prompt(cls, value):
        if not value.strip() or any(0xD800 <= ord(c) <= 0xDFFF for c in value):
            raise ValueError('Enter a nonempty, valid Unicode prompt')
        return value


class GenerateRequest(Settings):
    model_id: ModelID = 'b01'


class GenerationResult(BaseModel):
    model_id: ModelID
    model_name: str
    checkpoint_step: int
    prompt: str
    continuation: str
    full_output: str
    generated_tokens: int
    output_characters: int
    prompt_tokens: int
    effective_context: int
    context_cropped: bool
    settings: dict
    latency_seconds: float
    replacement_characters: bool
    seed: int
    request_id: str


class CompareSlot(BaseModel):
    model_id: ModelID
    status: Literal['completed', 'unavailable', 'failed']
    result: GenerationResult | None = None
    error: str | None = None


class CompareResult(BaseModel):
    request_id: str
    seed: int
    results: list[CompareSlot]
    total_latency_seconds: float


class ModelInfo(BaseModel):
    id: ModelID
    name: str
    description: str
    parameters: int
    checkpoint_step: int
    validation_bpb: float
    architecture: dict
    tokenizer_hash: str
    tokenizer: str
    context_length: int
    availability: Literal['ready', 'not_loaded', 'unavailable']
    limitations: str


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    loading_policy: Literal['lazy_locked']
    device: Literal['cpu', 'cuda']
