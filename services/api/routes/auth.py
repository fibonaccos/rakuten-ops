from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import User
from db.session import get_session
from schemas.auth import Token, APIUser
from services.auth import (
    authenticate_user,
    create_access_token,
    get_current_user
)


router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    path="/login",
    summary="login",
    description="Get a JWT token by authenticating to the API using username and passowrd.",
    response_model=Token
)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_session)
) -> Token:
    user = await authenticate_user(form_data.username, form_data.password, session)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Token(access_token=create_access_token(subject=user.username))


@router.get(
    path="/me",
    summary="me",
    description="Get the current user (you) informations",
    response_model=APIUser
)
def read_current_user(current_user: User = Depends(get_current_user)) -> APIUser:
    return APIUser(username=current_user.username, role=current_user.role)
