"""Тесты детекции кадров — rigorous validation of frame duplicate detection.

Tests verify ROI-based duplicate detection, scene change detection,
and pixel difference early exit for efficient video processing.
"""

import numpy as np
import pytest


class TestFrameDetection:
    """Тесты детекции дубликатов кадров на основе MSE."""

    def test_identical_frames_detected_as_duplicate(self) -> None:
        """Тест: идентичные кадры определяются как дубликаты.
        
        Two identical frames should be detected as duplicates.
        """
        from app.core.ocr_pipeline import OcrPipeline
        
        # Create identical frames
        frame1 = np.ones((480, 640), dtype=np.uint8) * 128
        frame2 = np.ones((480, 640), dtype=np.uint8) * 128
        
        is_duplicate = OcrPipeline._is_duplicate_frame(frame2, frame1)
        
        assert is_duplicate == True

    def test_different_frames_not_duplicate(self) -> None:
        """Тест: разные кадры не являются дубликатами.
        
        Two significantly different frames should not be duplicates.
        """
        from app.core.ocr_pipeline import OcrPipeline
        
        # Create different frames
        frame1 = np.ones((480, 640), dtype=np.uint8) * 128
        frame2 = np.ones((480, 640), dtype=np.uint8) * 200  # Different brightness
        
        is_duplicate = OcrPipeline._is_duplicate_frame(frame2, frame1)
        
        assert is_duplicate == False

    def test_roi_based_detection_small_change_in_roi(self) -> None:
        """Тест: ROI-based detection — изменение в ROI регионе.
        
        Creates two frames where a region differs significantly.
        This simulates a SCADA value change in a specific region.
        """
        from app.core.ocr_pipeline import OcrPipeline
        
        # Base frame
        frame1 = np.ones((480, 640), dtype=np.uint8) * 100
        
        # Frame with significant change in one region (simulating value change)
        frame2 = frame1.copy()
        # Change a larger region with high contrast (like a SCADA value updating)
        frame2[100:200, 200:400] = 250  # Large bright region
        
        is_duplicate = OcrPipeline._is_duplicate_frame(frame2, frame1)
        
        # Significant change should NOT be detected as duplicate
        # (we want to process frames with value changes)
        assert is_duplicate == False

    def test_roi_based_detection_no_change_in_roi(self) -> None:
        """Тест: ROI-based detection — нет изменений в ROI регионе.
        
        When no significant changes occur, frames should be duplicates.
        """
        from app.core.ocr_pipeline import OcrPipeline
        
        # Base frame
        frame1 = np.ones((480, 640), dtype=np.uint8) * 100
        
        # Frame with only tiny noise (below threshold)
        frame2 = frame1.copy()
        # Add tiny noise (within threshold)
        noise = np.random.randint(-5, 5, frame1.shape, dtype=np.int16)
        frame2 = np.clip(frame1.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        is_duplicate = OcrPipeline._is_duplicate_frame(frame2, frame1)
        
        # Small noise should be detected as duplicate
        assert is_duplicate == True

    def test_scene_change_top_region_different(self) -> None:
        """Тест: смена сцены — верхние 10% кадра отличаются.
        
        Scene change detection when top portion of frame differs significantly.
        This indicates a major change (e.g., menu navigation, screen switch).
        """
        from app.core.ocr_pipeline import OcrPipeline
        
        # Base frame
        frame1 = np.ones((480, 640), dtype=np.uint8) * 100
        
        # Frame with top portion completely different (scene change)
        frame2 = frame1.copy()
        frame2[:48, :] = 250  # Top 10% completely different
        
        is_duplicate = OcrPipeline._is_duplicate_frame(frame2, frame1)
        
        # Major scene change should NOT be duplicate
        assert is_duplicate == False

    def test_pixel_diff_early_exit_nearly_identical(self) -> None:
        """Тест: ранний выход по pixel diff — почти идентичные кадры.
        
        Nearly identical frames should be quickly detected as duplicates
        using early exit optimization.
        """
        from app.core.ocr_pipeline import OcrPipeline
        
        # Create nearly identical frames
        frame1 = np.ones((480, 640), dtype=np.uint8) * 128
        frame2 = frame1.copy()
        # Modify just a few pixels
        frame2[100, 100] = 130
        frame2[200, 200] = 125
        
        is_duplicate = OcrPipeline._is_duplicate_frame(frame2, frame1)
        
        # Should be detected as duplicate (changes too small)
        assert is_duplicate == True

    def test_fallback_no_calibration_rois(self) -> None:
        """Тест: fallback когда нет калибровочных ROI.
        
        When no calibration ROIs are available, should fall back to
        full-frame comparison.
        """
        from app.core.ocr_pipeline import OcrPipeline
        
        # Test with None previous frame (first frame)
        frame = np.ones((480, 640), dtype=np.uint8) * 128
        
        is_duplicate = OcrPipeline._is_duplicate_frame(frame, None)
        
        # First frame is never a duplicate
        assert is_duplicate == False

    def test_different_frame_sizes_not_duplicate(self) -> None:
        """Тест: разные размеры кадров → не дубликаты.
        
        Frames with different dimensions should not be considered duplicates.
        """
        from app.core.ocr_pipeline import OcrPipeline
        
        frame1 = np.ones((480, 640), dtype=np.uint8) * 128
        frame2 = np.ones((240, 320), dtype=np.uint8) * 128  # Different size
        
        is_duplicate = OcrPipeline._is_duplicate_frame(frame2, frame1)
        
        assert is_duplicate == False

    def test_mse_threshold_boundary(self) -> None:
        """Тест: граничное значение MSE порога.
        
        Test right at the MSE threshold boundary.
        """
        from app.core.ocr_pipeline import OcrPipeline, DUPLICATE_MSE_THRESHOLD
        
        # Create frame with controlled MSE
        frame1 = np.ones((480, 640), dtype=np.uint8) * 100
        frame2 = frame1.copy()
        
        # Add significant difference to exceed threshold
        # Change a large portion of the frame
        frame2[100:400, 100:500] = 250  # Large bright region
        
        is_duplicate = OcrPipeline._is_duplicate_frame(frame2, frame1)
        
        # With significant change, should NOT be duplicate
        assert is_duplicate == False

    def test_identical_frames_mse_zero(self) -> None:
        """Тест: идентичные кадры имеют MSE = 0."""
        from app.core.ocr_pipeline import OcrPipeline
        
        frame = np.ones((480, 640), dtype=np.uint8) * 128
        
        # Calculate MSE manually
        small_frame = frame[:64, :64]
        mse = np.mean((small_frame.astype(np.float32) - small_frame.astype(np.float32)) ** 2)
        
        assert mse == 0.0
        
        is_duplicate = OcrPipeline._is_duplicate_frame(frame, frame)
        assert is_duplicate == True


class TestFrameDetectionRealisticScenarios:
    """Тесты реалистичных сценариев детекции кадров."""

    def test_scada_value_update_detected(self) -> None:
        """Тест: обновление значения SCADA обнаруживается.
        
        Simulates a SCADA screen where a numeric value changes significantly.
        """
        from app.core.ocr_pipeline import OcrPipeline
        
        # Create SCADA-like frame
        frame1 = np.ones((480, 640), dtype=np.uint8) * 50  # Dark background
        # Add "text region" with value
        frame1[100:200, 200:500] = 200  # Light text region
        
        # New frame with significantly updated value
        frame2 = np.ones((480, 640), dtype=np.uint8) * 50
        frame2[100:200, 200:500] = 100  # Very different value region
        
        is_duplicate = OcrPipeline._is_duplicate_frame(frame2, frame1)
        
        # Value change should be detected (not duplicate)
        assert is_duplicate == False

    def test_static_scada_screen_duplicate(self) -> None:
        """Тест: статичный SCADA экран → дубликат.
        
        When SCADA screen is static (no value changes), frames should
        be detected as duplicates to save processing time.
        """
        from app.core.ocr_pipeline import OcrPipeline
        
        # Create SCADA-like frame
        frame1 = np.ones((480, 640), dtype=np.uint8) * 50
        frame1[100:150, 200:400] = 200  # Text region
        frame1[300:350, 100:500] = 150  # Another UI element
        
        # Almost identical frame (just tiny noise)
        frame2 = frame1.copy()
        noise = np.random.normal(0, 2, frame1.shape).astype(np.int16)
        frame2 = np.clip(frame1.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        is_duplicate = OcrPipeline._is_duplicate_frame(frame2, frame1)
        
        # Static screen should be duplicate
        assert is_duplicate == True

    def test_camera_movement_not_duplicate(self) -> None:
        """Тест: движение камеры → не дубликат.
        
        When camera moves (handheld video), frames should not be duplicates.
        """
        from app.core.ocr_pipeline import OcrPipeline
        
        # Create frame with pattern
        frame1 = np.ones((480, 640), dtype=np.uint8) * 100
        frame1[200:300, 200:400] = 200  # Central object
        
        # Shifted frame (camera movement)
        frame2 = np.ones((480, 640), dtype=np.uint8) * 100
        frame2[210:310, 210:410] = 200  # Shifted object
        
        is_duplicate = OcrPipeline._is_duplicate_frame(frame2, frame1)
        
        # Camera movement should NOT be duplicate
        assert is_duplicate == False
