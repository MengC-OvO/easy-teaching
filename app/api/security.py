"""Streaming-compatible HTTP boundary protections."""

from urllib.parse import urlsplit

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.formparsers import MultiPartException

from app.config import settings


class RequestBoundaryMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = Headers(scope=scope)
        unsafe = scope["method"] not in {"GET", "HEAD", "OPTIONS"}
        if unsafe:
            origin = headers.get("origin")
            expected = settings.public_origin.rstrip("/") or (
                f'{scope["scheme"]}://{headers.get("host", "")}'
            )
            if headers.get("sec-fetch-site") == "cross-site" or (
                origin is not None and origin != expected
            ):
                return await JSONResponse(
                    {"detail": "Cross-site requests are not allowed."}, status_code=403
                )(scope, receive, send)

        # Count actual bytes as well as Content-Length; chunked uploads have no length.
        length = headers.get("content-length")
        if length is not None:
            try:
                size = int(length)
                if size < 0:
                    raise ValueError
            except ValueError:
                return await JSONResponse({"detail": "Invalid Content-Length."}, 400)(scope, receive, send)
            if size > settings.request_max_bytes:
                return await JSONResponse({"detail": "Request body is too large."}, 413)(scope, receive, send)
        received = 0
        too_large = False
        started = False

        async def limited_receive():
            nonlocal received, too_large
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > settings.request_max_bytes:
                    too_large = True
                    if headers.get("content-type", "").startswith("multipart/"):
                        # The multipart parser closes its temporary files on this error.
                        raise MultiPartException("Request body is too large.")
                    return {"type": "http.disconnect"}
            return message

        async def protected_send(message):
            nonlocal started
            if too_large:
                return
            if message["type"] == "http.response.start":
                started = True
                response_headers = MutableHeaders(scope=message)
                response_headers["X-Content-Type-Options"] = "nosniff"
                response_headers["X-Frame-Options"] = "DENY"
                response_headers["Referrer-Policy"] = "no-referrer"
                response_headers["Cache-Control"] = "no-store"
                auth_origin = urlsplit(settings.supabase_url)
                connect = f"{auth_origin.scheme}://{auth_origin.netloc}" if auth_origin.hostname else ""
                response_headers["Content-Security-Policy"] = (
                    "default-src 'self'; script-src 'self'; style-src 'self'; "
                    f"connect-src 'self' {connect}; img-src 'self' data:; "
                    "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
                )
                if settings.public_origin.startswith("https://"):
                    response_headers["Strict-Transport-Security"] = "max-age=31536000"
            await send(message)

        try:
            await self.app(scope, limited_receive, protected_send)
        except Exception:
            if not too_large:
                raise
        if too_large and not started:
            await JSONResponse({"detail": "Request body is too large."}, 413)(scope, receive, send)
