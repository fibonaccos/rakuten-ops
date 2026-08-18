import bcrypt

from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import User, UserRole
from db.session import get_session
from services.db import UserRepository
from schemas.auth import TokenPayload, APIUser
from _config import get_settings


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    """Hash a plain password with bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check a plain password against its bcrypt hash."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"), hashed_password.encode("utf-8")
    )


async def authenticate_user(username: str, password: str, session: AsyncSession) -> APIUser:
    """Return the user if credentials are valid, otherwise None."""
    repository: UserRepository = UserRepository(session)
    user = await repository.get_by_name(username)
    if user is None or not verify_password(password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return APIUser(username=user.username, role=user.role)


def create_access_token(subject: str) -> str:
    """Create a signed JWT access token for the given subject (username)."""
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=get_settings().jwt_expiration_in_minutes
    )
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(
        payload,
        key=get_settings().jwt_secret.get_secret_value(),
        algorithm=get_settings().jwt_algorithm
    )


async def get_current_user(token: str = Depends(oauth2_scheme), session: AsyncSession = Depends(get_session)) -> User:
    """FastAPI dependency resolving the authenticated user from the Bearer token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        raw_payload = jwt.decode(
            token,
            get_settings().jwt_secret.get_secret_value(),
            algorithms=[get_settings().jwt_algorithm]
        )
        payload = TokenPayload(**raw_payload)
    except JWTError:
        raise credentials_exception

    if payload.sub is None:
        raise credentials_exception

    user_name = payload.sub

    repository: UserRepository = UserRepository(session)
    user = await repository.get_by_name(user_name)

    if user is None:
        raise credentials_exception

    return user


async def ensure_user_is_admin_from_token(token: str = Depends(oauth2_scheme), session: AsyncSession = Depends(get_session)) -> User:
    user: User = await get_current_user(token, session)
    if user.role != UserRole.ADMIN:
        message = f"Access requires admin privileges. Your role : {user.role}."
        raise HTTPException(status_code=401, detail=message)
    return user
