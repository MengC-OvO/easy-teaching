"""Session-scoped file uploads for document and voice tools."""

from typing import Optional, Union

from fastapi import APIRouter, Depends, File, Request, UploadFile, status
from fastapi.responses import JSONResponse

from app.api.auth import CurrentUser, get_current_user, require_session_owner
from app.api.dependencies import get_runtime
from app.schemas import ApiErrorDetail, ApiErrorResponse, UploadResponse


router = APIRouter(prefix="/sessions", tags=["uploads"])


@router.post(
    "/{session_id}/uploads",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ApiErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ApiErrorResponse},
    },
)
async def upload_session_file(
    session_id: str,
    request: Request,
    file: UploadFile = File(...),
    current_user: Optional[CurrentUser] = Depends(get_current_user),
) -> Union[UploadResponse, JSONResponse]:
    runtime = get_runtime(request)
    session = await runtime.store.get_conversation_session(session_id)
    if session is None:
        return _error(status.HTTP_404_NOT_FOUND, "session_not_found", "The requested session does not exist.")
    require_session_owner(session, current_user)
    teacher_id = str(session.get("teacher_id") or "")
    class_id = str(session.get("class_id") or "")
    if not teacher_id or not class_id or runtime.file_store is None:
        return _error(status.HTTP_400_BAD_REQUEST, "upload_scope_required", "Uploads require a session with teacher_id and class_id.")
    try:
        content = await file.read(runtime.file_store.max_bytes + 1)
        record = runtime.file_store.save_bytes(
            filename=file.filename or "upload",
            content_type=file.content_type,
            content=content,
            teacher_id=teacher_id,
            class_id=class_id,
            session_id=session_id,
        )
    except ValueError as error:
        return _error(status.HTTP_400_BAD_REQUEST, "invalid_upload", str(error))
    finally:
        await file.close()
    return UploadResponse(
        file_id=record.file_id,
        filename=record.original_name,
        category=record.category,
        content_type=record.content_type,
        size_bytes=record.size_bytes,
        sha256=record.sha256,
    )


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    payload = ApiErrorResponse(
        error=ApiErrorDetail(code=code, message=message, recoverable=False)
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))
