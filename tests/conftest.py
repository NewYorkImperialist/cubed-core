from __future__ import annotations

import pytest


@pytest.fixture
def color_calibration_payload() -> dict[str, object]:
    centroids = {
        "white": [235.0, 128.0, 128.0],
        "green": [140.0, 70.0, 155.0],
        "red": [140.0, 190.0, 165.0],
        "blue": [100.0, 160.0, 70.0],
        "orange": [180.0, 170.0, 200.0],
        "yellow": [220.0, 115.0, 210.0],
    }
    return {
        "schema": "cubed-core/color-calibration-v1",
        "schema_version": 1,
        "color_space": "opencv_bgr_lab_uint8_v1",
        "geometry": {
            "target_square_fraction": 0.56,
            "grid_rows": 3,
            "grid_columns": 3,
            "central_patch_fraction": 0.40,
            "coordinate_space": "rotated_video_frame_pixels",
        },
        "thresholds": {
            "minimum_l": 40.0,
            "uniformity_maximum": 28.0,
            "stability_maximum": 10.0,
            "collision_minimum": 15.0,
            "distance_weights": [0.15, 1.0, 1.0],
        },
        "order": ["white", "green", "red", "blue", "orange", "yellow"],
        "frames_per_color": 5,
        "samples_per_frame": 9,
        "centroids": centroids,
        "samples": {
            color: [list(centroid) for _ in range(45)] for color, centroid in centroids.items()
        },
        "provenance": {
            "captured_unix_ms": 1_800_000_010_000.0,
            "collection_started_unix_ms": 1_800_000_000_000.0,
            "camera_facing": "back",
            "mirrored": False,
            "frame_width": 640,
            "frame_height": 360,
            "camera_device_type": "AVCaptureDeviceTypeBuiltInWideAngleCamera",
            "camera_format_width": 1280,
            "camera_format_height": 720,
            "exposure_duration_seconds": 0.008,
            "iso": 160.0,
            "white_balance_gains": [1.8, 1.0, 1.6],
            "lens_position": 0.4,
            "exposure_locked": True,
            "white_balance_locked": True,
            "focus_locked": True,
            "app": "cubed-capture-ios",
            "sampler": "server_grid_opencv_lab_v1",
        },
    }


@pytest.fixture
def flat_color_centroids_payload() -> dict[str, list[float]]:
    """Shape of the released decode-support asset (e.g. calibration_gan12.json)."""

    return {
        "red": [114.5, 195.682, 170.477],
        "blue": [98.622, 132.489, 83.889],
        "green": [159.6, 75.156, 154.378],
        "white": [205.143, 129.381, 135.595],
        "orange": [143.178, 182.489, 189.244],
        "yellow": [199.333, 118.289, 202.022],
    }


@pytest.fixture
def color_centroids_envelope_payload(
    flat_color_centroids_payload: dict[str, list[float]],
) -> dict[str, object]:
    """A pre-wrapped cubed-core/color-centroids-v1 document."""

    return {
        "schema": "cubed-core/color-centroids-v1",
        "schema_version": 1,
        "color_space": "cielab",
        "centroids": {
            color: list(vector) for color, vector in flat_color_centroids_payload.items()
        },
        "provenance": "imported-centroids sha256:" + "0" * 64,
    }
