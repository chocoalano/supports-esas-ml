"""Who is allowed to send this service anything at all.

The service performs face recognition on whatever it is handed, so an open port
is not merely a free GPU: it is a way for anybody to test photographs of people
against each other. It holds no accounts and no sessions, so the whole of its
access control is one shared key per caller.

`FSA_CORS_ORIGINS` is not this. CORS is a rule a *browser* applies to its own
requests; curl, a mobile client and a script ignore it entirely, so an origin
list has never kept anybody out of anything.

The default is closed. With no keys configured the service refuses every
request rather than serving them, except while `FSA_DEBUG` is on - which is the
local-development case, and says so loudly at boot.
"""

from __future__ import annotations

import hmac
import logging
from typing import Annotated

from fastapi import Depends, Header

from app.core.config import Settings, get_settings
from app.core.errors import AppError

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"


class UnauthorizedError(AppError):
    code = "unauthorized"
    status_code = 401


class NotConfiguredError(AppError):
    code = "api_keys_not_configured"
    status_code = 503


def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header(alias=API_KEY_HEADER)] = None,
) -> str | None:
    """Admit a request, or refuse it.

    Returns the caller's label so log lines can say who was asking, without the
    key itself ever reaching a log.
    """
    # Asked of the usable keys rather than of the raw list. A service whose only
    # configured entries are malformed - `label:` with nothing after it, a stray
    # comma - has no key anybody can present, and saying "that key is not
    # recognised" to every caller describes the caller's mistake instead of the
    # operator's. `FSA_API_KEYS` is also named in the boot log for that case.
    if not settings.api_key_map:
        if settings.debug:
            return "debug"

        raise NotConfiguredError(
            "This service has no usable API keys configured and will not serve requests. "
            "Set FSA_API_KEYS.",
        )

    if not x_api_key:
        raise UnauthorizedError(f"Missing {API_KEY_HEADER} header.")

    # Compared as bytes. `hmac.compare_digest` refuses two str arguments unless
    # both are ASCII-only, and an ASGI header is decoded as latin-1 — so a single
    # byte above 127 in the header turned an unrecognised key into a TypeError
    # and a 500. A malformed key is a refusal, not a server fault.
    candidate = x_api_key.encode("utf-8", errors="ignore")

    # compare_digest on every candidate, and no early exit on the first
    # mismatch: `==` leaks the length of the shared prefix through timing, which
    # is enough to recover a key one character at a time.
    matched: str | None = None
    for label, key in settings.api_key_map.items():
        if hmac.compare_digest(candidate, key.encode("utf-8")):
            matched = label

    if matched is None:
        raise UnauthorizedError("That API key is not recognised.")

    return matched


ApiKeyDep = Annotated[str | None, Depends(require_api_key)]
