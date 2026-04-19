"""WebSocket для real-time обновлений фронтенда."""

from __future__ import annotations

import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Менеджер WebSocket-соединений."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Принимает новое соединение."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket подключён. Всего: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        """Отключает соединение."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WebSocket отключён. Всего: %d", len(self.active_connections))

    async def broadcast(self, message: dict) -> None:
        """Рассылает сообщение всем подключённым клиентам.

        Args:
            message: Словарь с данными для отправки.
        """
        data = json.dumps(message, ensure_ascii=False)
        disconnected: list[WebSocket] = []

        for connection in self.active_connections:
            try:
                await connection.send_text(data)
            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)

    async def send_progress(
        self,
        video_id: str,
        progress_pct: float,
        current_step: str,
        frames_processed: int,
        total_frames: int,
    ) -> None:
        """Отправляет обновление прогресса обработки.

        Args:
            video_id: ID видео.
            progress_pct: Процент выполнения (0-100).
            current_step: Текущий этап обработки.
            frames_processed: Обработано кадров.
            total_frames: Всего кадров.
        """
        await self.broadcast({
            "type": "progress",
            "video_id": video_id,
            "progress_pct": progress_pct,
            "current_step": current_step,
            "frames_processed": frames_processed,
            "total_frames": total_frames,
        })

    async def send_ocr_result(
        self,
        video_id: str,
        param_id: int,
        value: str,
        confidence: float,
        timestamp: str,
        label: str = "",
        source: str = "paddle",
        processing_ms: float = 0.0,
    ) -> None:
        """Отправляет результат распознавания параметра.

        Args:
            video_id: ID видео.
            param_id: ID параметра.
            value: Распознанное значение.
            confidence: Уверенность пары (0.0–1.0).
            timestamp: Таймстемп кадра.
            label: Метка параметра (например, "TI-101").
            source: Источник OCR ("paddle" | "florence" | "merged").
            processing_ms: Время обработки кадра в мс.
        """
        await self.broadcast({
            "type": "ocr_result",
            "video_id": video_id,
            "param_id": param_id,
            "label": label,
            "value": value,
            "confidence": confidence,
            "source": source,
            "processing_ms": processing_ms,
            "timestamp": timestamp,
        })


ws_manager = ConnectionManager()
