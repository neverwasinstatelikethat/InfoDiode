"""API-роутер для аутентификации и управления профилем пользователя."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.auth_service import (
    authenticate_user,
    change_password,
    create_access_token,
    decode_access_token,
    register_user,
    update_email_settings,
    update_user_profile,
)
from app.models.schemas import (
    ChangePasswordRequest,
    EmailSettingsRequest,
    LoginRequest,
    ProfileUpdateRequest,
    RegisterRequest,
    TokenResponse,
    UserProfileResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

security = HTTPBearer(auto_error=False)


async def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> dict:
    """Извлекает текущего пользователя из JWT-токена.

    Args:
        credentials: HTTP Bearer credentials.

    Returns:
        Данные пользователя.

    Raises:
        HTTPException: 401, если токен невалиден.
    """
    from app.core.auth_service import get_user

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Требуется авторизация",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Недействительный или истёкший токен",
            headers={"WWW-Authenticate": "Bearer"},
        )

    username: str | None = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Недействительный токен",
        )

    user = get_user(username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Пользователь не найден",
        )

    return user


@router.post("/register", response_model=UserProfileResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest) -> UserProfileResponse:
    """Регистрирует нового пользователя.

    Args:
        request: Данные регистрации.

    Returns:
        Профиль созданного пользователя.
    """
    user = register_user(
        username=request.username,
        email=request.email,
        password=request.password,
        full_name=request.full_name,
    )
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Пользователь с таким логином уже существует",
        )

    return UserProfileResponse(
        username=user["username"],
        email=user["email"],
        full_name=user["full_name"],
        email_recipients=user.get("email_recipients", []),
        default_recipient=user.get("default_recipient", ""),
    )


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest) -> TokenResponse:
    """Аутентификация пользователя и получение JWT-токена.

    Args:
        request: Данные входа (логин + пароль).

    Returns:
        JWT-токен доступа.
    """
    user = authenticate_user(request.username, request.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный логин или пароль",
        )

    token = create_access_token(data={"sub": user["username"]})
    return TokenResponse(access_token=token, token_type="bearer")


@router.get("/me", response_model=UserProfileResponse)
async def get_profile(current_user: dict = Depends(get_current_user)) -> UserProfileResponse:
    """Возвращает профиль текущего пользователя.

    Args:
        current_user: Текущий аутентифицированный пользователь.

    Returns:
        Данные профиля.
    """
    return UserProfileResponse(
        username=current_user["username"],
        email=current_user["email"],
        full_name=current_user["full_name"],
        email_recipients=current_user.get("email_recipients", []),
        default_recipient=current_user.get("default_recipient", ""),
    )


@router.patch("/profile", response_model=UserProfileResponse)
async def update_profile(
    request: ProfileUpdateRequest,
    current_user: dict = Depends(get_current_user),
) -> UserProfileResponse:
    """Обновляет профиль текущего пользователя.

    Args:
        request: Данные для обновления.
        current_user: Текущий пользователь.

    Returns:
        Обновлённый профиль.
    """
    user = update_user_profile(
        username=current_user["username"],
        full_name=request.full_name,
        email=request.email,
    )
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Пользователь не найден",
        )

    return UserProfileResponse(
        username=user["username"],
        email=user["email"],
        full_name=user["full_name"],
        email_recipients=user.get("email_recipients", []),
        default_recipient=user.get("default_recipient", ""),
    )


@router.post("/change-password")
async def change_user_password(
    request: ChangePasswordRequest,
    current_user: dict = Depends(get_current_user),
) -> dict[str, str]:
    """Меняет пароль текущего пользователя.

    Args:
        request: Старый и новый пароль.
        current_user: Текущий пользователь.

    Returns:
        Сообщение об успешной смене пароля.
    """
    success = change_password(
        username=current_user["username"],
        old_password=request.old_password,
        new_password=request.new_password,
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Неверный текущий пароль",
        )

    return {"message": "Пароль успешно изменён"}


@router.patch("/email-settings", response_model=UserProfileResponse)
async def update_user_email_settings(
    request: EmailSettingsRequest,
    current_user: dict = Depends(get_current_user),
) -> UserProfileResponse:
    """Обновляет настройки email для отправки XML.

    Args:
        request: Настройки email (список получателей, получатель по умолчанию).
        current_user: Текущий пользователь.

    Returns:
        Обновлённый профиль.
    """
    user = update_email_settings(
        username=current_user["username"],
        email_recipients=request.email_recipients,
        default_recipient=request.default_recipient,
    )
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Пользователь не найден",
        )

    return UserProfileResponse(
        username=user["username"],
        email=user["email"],
        full_name=user["full_name"],
        email_recipients=user.get("email_recipients", []),
        default_recipient=user.get("default_recipient", ""),
    )
