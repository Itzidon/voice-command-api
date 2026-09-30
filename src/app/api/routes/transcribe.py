from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from groq import Groq
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from src.app.api.routes.instruction import route_instruction
from src.app.api.routes.tasks import (
    create_task,
    delete_task,
    get_tasks,
    replace_task,
    update_task,
)
from src.app.core.config import get_settings
from src.app.schemas.voice import (
    InstructionPayload,
    InstructionRequest,
    TaskCreate,
    TaskReplace,
    TaskUpdate,
    TranscribeFlowResponse,
)

router = APIRouter(tags=["transcribe"])


@router.get("/")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


def execute_instruction(instruction: InstructionPayload) -> Any:
    method = instruction.method.upper()
    endpoint = instruction.endpoint.rstrip("/")
    params = instruction.params

    if endpoint == "/tasks":
        if method == "GET":
            return get_tasks()

        if method == "POST":
            return create_task(TaskCreate(**params))

    parts = endpoint.strip("/").split("/")

    if len(parts) == 2 and parts[0] == "tasks":
        try:
            task_id = int(parts[1])
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Groq returned an invalid task ID",
            ) from error

        if method == "PUT":
            return replace_task(
                task_id,
                TaskReplace(**params),
            )

        if method == "PATCH":
            return update_task(
                task_id,
                TaskUpdate(**params),
            )

        if method == "DELETE":
            return delete_task(task_id)

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Groq returned an unsupported route",
    )


def process_transcription(transcription: str) -> TranscribeFlowResponse:
    transcription = transcription.strip()

    if not transcription:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty transcription",
        )

    try:
        instruction = route_instruction(
            InstructionRequest(transcription=transcription)
        )

        result = execute_instruction(instruction)

        return TranscribeFlowResponse(
            transcription=transcription,
            instruction=instruction,
            result=result,
        )

    except ValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Invalid instruction parameters returned by Groq",
        ) from error


@router.post("/transcribe", response_model=TranscribeFlowResponse)
async def transcribe_and_run_flow(
    request: Request,
) -> TranscribeFlowResponse:
    content_type = request.headers.get("content-type", "")

    # Permite también el modo manual incluido en el frontend
    if "application/json" in content_type:
        body = await request.json()

        transcription = str(
            body.get("transcription", "")
        ).strip()

        return process_transcription(transcription)

    # Flujo normal: audio grabado desde el navegador
    if "multipart/form-data" not in content_type:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Expected multipart/form-data or application/json",
        )

    form = await request.form()

    uploaded_file = form.get("file")
    language = form.get("language")

    if not isinstance(uploaded_file, UploadFile):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is required",
        )

    audio_bytes = await uploaded_file.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is empty",
        )

    settings = get_settings()
    client = Groq(api_key=settings.groq_api_key)

    transcription_args: dict[str, Any] = {
        "file": (
            uploaded_file.filename or "recording.webm",
            audio_bytes,
        ),
        "model": settings.groq_transcription_model,
        "response_format": "json",
    }

    if isinstance(language, str) and language.strip():
        transcription_args["language"] = language.strip()

    try:
        transcription_response = client.audio.transcriptions.create(
            **transcription_args
        )
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Audio transcription failed: {str(error)}",
        ) from error

    transcription = transcription_response.text.strip()

    return process_transcription(transcription)