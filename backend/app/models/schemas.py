"""Pydantic-схемы для API InfoDiode."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class VideoType(str, Enum):
    """Тип видеозаписи — определяет pipeline предобработки."""

    DIRECT = "direct"
    HANDHELD = "handheld"
    HANDHELD_ANGLE = "handheld_angle"


class ZoneType(str, Enum):
    """Зоны мнемосхемы SCADA."""

    HEADER = "header"
    LEFT_NAV = "left_nav"
    CENTRAL_SCHEMA = "central_schema"
    RIGHT_PANEL = "right_panel"
    BOTTOM_BAR = "bottom_bar"


class VideoUploadResponse(BaseModel):
    """Ответ при загрузке видео."""

    video_id: str
    filename: str
    video_type: VideoType
    resolution: str  # e.g. "1920x1080"
    fps: float
    duration_s: float
    total_frames: int


class FrameExtractionRequest(BaseModel):
    """Запрос на извлечение кадров."""

    video_id: str
    interval_ms: int = Field(default=500, description="Интервал между кадрами в мс")


class BoundingBox(BaseModel):
    """Ограничивающий прямоугольник в нормализованных координатах [0,1]."""

    x1: float
    y1: float
    x2: float
    y2: float


class ZoneBoundary(BaseModel):
    """Границы зоны на мнемосхеме."""

    zone: ZoneType
    bbox: BoundingBox


class OCRTextResult(BaseModel):
    """Результат распознавания текста."""

    text: str
    confidence: float
    bbox: BoundingBox


class LabelValuePair(BaseModel):
    """Пара метка-значение, извлечённая из кадра."""

    label: str
    value: str
    label_bbox: BoundingBox
    value_bbox: BoundingBox
    confidence: float
    zone: ZoneType = ZoneType.CENTRAL_SCHEMA
    color_state: str = "normal"  # normal | alarm | warning | inactive


class ParameterMapping(BaseModel):
    """Связь распознанной метки с ID параметра из таблицы."""

    param_id: int
    label_text: str
    short_name: str
    full_name: str
    unit: str
    decimal_places: int
    roi_bbox: BoundingBox  # DEPRECATED: use label_bbox + value_bbox instead
    label_bbox: BoundingBox | None = None  # Bbox of the label text (e.g., "Температура масла")
    value_bbox: BoundingBox | None = None  # Bbox of the value text (e.g., "45.2")
    zone: ZoneType


class ParamMetadata(BaseModel):
    """Метаданные параметра для XML и API."""

    short_name: str = ""
    full_name: str = ""
    unit: str = ""


class VLMParameter(BaseModel):
    """Один параметр из ответа VLM (валидированная схема)."""

    label: str = Field(..., min_length=1, description="Название параметра с экрана")
    value: float | str = Field(..., description="Числовое значение или статус")
    unit: str = Field(default="", description="Единица измерения (°C, кПа, мм/с)")
    param_type: str = Field(default="R", description="Тип параметра (T, P, Vb, L, n, Pos, V, f, R)")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Уверенность VLM (0.0-1.0)")
    in_range: bool = Field(default=True, description="Значение в физическом диапазоне")
    zone_id: int = Field(default=-1, description="ID зоны, из которой извлечён параметр")


class VLMResponse(BaseModel):
    """Валидированный ответ VLM для одного кадра."""

    parameters: list[VLMParameter] = Field(..., description="Список найденных параметров")

    @property
    def param_count(self) -> int:
        """Количество найденных параметров."""
        return len(self.parameters)


class SnapshotData(BaseModel):
    """Данные одного снимка (500мс интервал)."""

    timestamp: str = Field(pattern=r"^\d{2}:\d{2}:\d{2}\.\d{3}$")
    params: dict[int, str]  # param_id -> value string
    # Опциональные метаданные параметров (для XML с именами)
    param_metadata: dict[int, ParamMetadata] = Field(default_factory=dict)


class PipelineStatus(BaseModel):
    """Статус конвейера обработки."""

    video_id: str
    status: str  # pending | processing | completed | failed
    progress_pct: float = 0.0
    current_step: str = ""
    frames_processed: int = 0
    total_frames: int = 0


class EvaluationMetrics(BaseModel):
    """Метрики оценки качества."""

    ocr_accuracy: float = 0.0
    parameter_identification_accuracy: float = 0.0
    xml_validity_score: float = 0.0
    avg_frame_latency_ms: float = 0.0
    p95_frame_latency_ms: float = 0.0
    total_snapshots: int = 0


# === Схемы аутентификации ===


class LoginRequest(BaseModel):
    """Запрос на вход в систему."""

    username: str = Field(min_length=3, max_length=50, description="Логин пользователя")
    password: str = Field(min_length=4, max_length=128, description="Пароль")


class RegisterRequest(BaseModel):
    """Запрос на регистрацию."""

    username: str = Field(min_length=3, max_length=50, description="Логин пользователя")
    email: str = Field(pattern=r"^[^@]+@[^@]+\.[^@]+$", description="Email адрес")
    password: str = Field(min_length=4, max_length=128, description="Пароль")
    full_name: str = Field(default="", max_length=200, description="ФИО")


class TokenResponse(BaseModel):
    """Ответ с JWT-токеном."""

    access_token: str
    token_type: str = "bearer"


class UserProfileResponse(BaseModel):
    """Профиль пользователя."""

    username: str
    email: str
    full_name: str
    email_recipients: list[str] = []
    default_recipient: str = ""


class ProfileUpdateRequest(BaseModel):
    """Запрос на обновление профиля."""

    full_name: str | None = None
    email: str | None = None


class ChangePasswordRequest(BaseModel):
    """Запрос на смену пароля."""

    old_password: str = Field(min_length=4, description="Текущий пароль")
    new_password: str = Field(min_length=4, max_length=128, description="Новый пароль")


class EmailSettingsRequest(BaseModel):
    """Запрос на обновление настроек email."""

    email_recipients: list[str] | None = None
    default_recipient: str | None = None
