"""Service authentication for the agent-revamp API.

Ported from realbooks-agents/app/api/auth.py — the Authorization header must carry a
Cognito user access token, verified via JWKS; account_id / user_id / role_key /
permissions are derived from the users SAPI by the verified `sub` claim. Identity
fields are never trusted from the request body. Auth is always on (fail-closed).
Same env names (COGNITO_JWKS_URL, USERS_DOMAIN) as the original service, so the same
credentials and servers work unchanged.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx
import jwt
from fastapi import HTTPException, Request, status

from agent_revamp.config import settings

logger = logging.getLogger(__name__)

_jwks_client: jwt.PyJWKClient | None = None
_user_cache: dict[str, tuple[float, dict]] = {}
_USER_CACHE_TTL_SECONDS = 10.0


@dataclass
class UserIdentity:
    """Identity of the caller, derived from a verified token (or body in dev mode)."""

    account_id: int
    user_id: int
    role_key: str
    permissions: list[str]


def _get_jwks_client() -> jwt.PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = jwt.PyJWKClient(settings.cognito_jwks_url, cache_keys=True)
    return _jwks_client


def _extract_bearer(auth_header: str) -> str:
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized: missing bearer token")
    token = auth_header.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized: missing bearer token")
    return token


def verify_cognito_token(auth_header: str) -> str:
    """Verify a Cognito Bearer token via JWKS and return its `sub` claim.

    Raises HTTPException(401) on a missing or invalid token.
    """
    token = _extract_bearer(auth_header)
    try:
        signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
        decoded = jwt.decode(token, signing_key.key, algorithms=["RS256"])
    except Exception as exc:
        logger.warning("Cognito token verification failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc
    sub = decoded.get("sub") or decoded.get("cognito:sub")
    if not sub:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return sub


async def get_user_info(cognito_sub: str) -> dict | None:
    """Fetch user identity (uid, account_id, role_key, permissions) from the users SAPI.

    Mirrors get_user_info() in realbooks-agents. Returns {} when the sub is not
    registered; None when the lookup itself failed (users-api unreachable or non-200)
    — callers should distinguish the two.
    """
    now = time.monotonic()
    cached = _user_cache.get(cognito_sub)
    if cached and now - cached[0] < _USER_CACHE_TTL_SECONDS:
        return cached[1]
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(settings.users_domain, params={"cognito_sub": cognito_sub})
    except Exception as exc:
        logger.warning("get_user_info failed for cognito_sub=%s at %s: %s", cognito_sub, settings.users_domain, exc)
        return None
    if resp.status_code != 200:
        logger.warning(
            "users-api returned %s for cognito_sub=%s at %s",
            resp.status_code,
            cognito_sub,
            settings.users_domain,
        )
        return None
    data = resp.json()
    users = data if isinstance(data, list) else data.get("data", [])
    if not users:
        logger.warning("no user found for cognito_sub=%s at %s", cognito_sub, settings.users_domain)
        return {}
    user = users[0]
    _user_cache[cognito_sub] = (now, user)
    return user


async def get_verified_identity(request: Request) -> UserIdentity:
    """FastAPI dependency: the caller's verified identity.

    The Bearer token must be a Cognito user access token, verified via JWKS;
    account_id / user_id / role_key / permissions are derived from the users
    SAPI by the verified `sub` claim. Body identity fields are never trusted.
    """
    sub = verify_cognito_token(request.headers.get("Authorization", ""))
    user = await get_user_info(sub)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity service unavailable. Please try again.",
        )
    if not user:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User not found.")
    return UserIdentity(
        account_id=int(user.get("account_id") or 0),
        user_id=int(user.get("uid") or 0),
        role_key=str(user.get("role_key") or ""),
        permissions=list(user.get("permissions") or []),
    )
