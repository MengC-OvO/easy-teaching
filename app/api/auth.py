"""Minimal Supabase Auth integration for the FastAPI boundary."""

from dataclasses import dataclass
import inspect
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from app.config import settings


AUTH_COOKIE_NAME = "easyteaching_access_token"


@dataclass(frozen=True)
class CurrentUser:
    """Trusted identity returned after Supabase validates an access token."""

    user_id: str
    email: Optional[str] = None

    @property
    def teacher_id(self) -> str:
        return self.user_id


class AuthConfigResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    supabase_url: str = ""
    supabase_publishable_key: str = ""


class AuthSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    access_token: str = Field(min_length=1)


class AuthUserResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    email: Optional[str] = None


class SupabaseAuthClient:
    """Validate Supabase access tokens through the official Auth user endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        publishable_key: str,
        timeout_seconds: float = 5.0,
        http_client: Optional[Any] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.publishable_key = publishable_key
        self.timeout_seconds = timeout_seconds
        self.http_client = http_client

    async def get_user(self, access_token: str) -> CurrentUser:
        headers = {
            "apikey": self.publishable_key,
            "Authorization": f"Bearer {access_token}",
        }
        try:
            if self.http_client is not None:
                response = self.http_client.get(
                    f"{self.base_url}/auth/v1/user",
                    headers=headers,
                )
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.base_url}/auth/v1/user",
                        headers=headers,
                        timeout=self.timeout_seconds,
                    )
            if inspect.isawaitable(response):
                response = await response
        except httpx.HTTPError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service is unavailable.",
            ) from error

        if response.status_code != status.HTTP_200_OK:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Login expired or invalid.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        payload: dict[str, Any] = response.json()
        user_id = payload.get("id")
        if not isinstance(user_id, str) or not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication response did not contain a user ID.",
            )
        email = payload.get("email")
        return CurrentUser(
            user_id=user_id,
            email=email if isinstance(email, str) else None,
        )


def _auth_client() -> SupabaseAuthClient:
    if not settings.auth_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Authentication is not enabled.",
        )
    if not settings.supabase_url or not settings.supabase_publishable_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase Auth is not configured.",
        )
    return SupabaseAuthClient(
        base_url=settings.supabase_url,
        publishable_key=settings.supabase_publishable_key,
        timeout_seconds=settings.auth_timeout_seconds,
    )


def _request_token(request: Request) -> Optional[str]:
    cookie_token = request.cookies.get(AUTH_COOKIE_NAME)
    if cookie_token:
        return cookie_token
    authorization = request.headers.get("Authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token:
        return token
    return None


async def get_current_user(request: Request) -> Optional[CurrentUser]:
    """Return a trusted user when auth is enabled; preserve local offline mode."""

    if not settings.auth_enabled:
        return None
    token = _request_token(request)
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Please log in to continue.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await _auth_client().get_user(token)


def require_authenticated_user(
    current_user: Optional[CurrentUser] = Depends(get_current_user),
) -> CurrentUser:
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Authentication is not enabled.",
        )
    return current_user


def require_session_owner(
    conversation: dict[str, Any],
    current_user: Optional[CurrentUser],
) -> None:
    """Prevent an authenticated teacher from accessing another teacher's session."""

    if current_user is None:
        return
    if conversation.get("teacher_id") != current_user.teacher_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this session.",
        )


router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/config", response_model=AuthConfigResponse)
def auth_config() -> AuthConfigResponse:
    return AuthConfigResponse(
        enabled=settings.auth_enabled,
        supabase_url=settings.supabase_url if settings.auth_enabled else "",
        supabase_publishable_key=(
            settings.supabase_publishable_key if settings.auth_enabled else ""
        ),
    )


@router.post("/session", response_model=AuthUserResponse)
async def create_auth_session(
    payload: AuthSessionRequest, response: Response
) -> AuthUserResponse:
    user = await _auth_client().get_user(payload.access_token)
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=payload.access_token,
        max_age=settings.auth_cookie_max_age_seconds,
        httponly=True,
        secure=settings.app_env not in {"local", "test"},
        samesite="lax",
        path="/",
    )
    return AuthUserResponse(user_id=user.user_id, email=user.email)


@router.get("/me", response_model=AuthUserResponse)
def get_auth_user(user: CurrentUser = Depends(require_authenticated_user)) -> AuthUserResponse:
    return AuthUserResponse(user_id=user.user_id, email=user.email)


@router.delete("/session", status_code=status.HTTP_204_NO_CONTENT)
def delete_auth_session(response: Response) -> Response:
    response.delete_cookie(
        key=AUTH_COOKIE_NAME,
        path="/",
        secure=settings.app_env not in {"local", "test"},
        samesite="lax",
    )
    response.status_code = status.HTTP_204_NO_CONTENT
    return response
