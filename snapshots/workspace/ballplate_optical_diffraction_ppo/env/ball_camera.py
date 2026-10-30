"""从真实渲染 RGB 提取球的二值掩膜，不读取球的物理位置。"""
import cv2
import mujoco
import numpy as np


def binary_ball(rgb, red_min=70, contrast_min=60, hue_max=18):
    """针对当前橙红色球筛选颜色，只保留最大的四连通区域。"""
    red, green, blue = rgb.astype(np.float32).transpose(2, 0, 1)
    hue = 60.0 * (green - blue) / np.maximum(red - blue, 1.0)
    mask = ((red > red_min) & (red > green) & (green >= blue)
            & (red - blue > contrast_min) & (hue >= 0) & (hue < hue_max))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=4
    )
    if count == 1:
        return np.zeros(mask.shape, dtype=np.uint8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8) * 255


def detect_ball(rgb, min_pixels=6):
    """保留旧诊断接口；图像策略不使用此质心坐标。"""
    rows, cols = np.nonzero(binary_ball(rgb))
    if len(rows) < min_pixels:
        return np.zeros(2, dtype=np.float32), False
    height, width = rgb.shape[:2]
    return np.array([2 * cols.mean() / (width - 1) - 1,
                     1 - 2 * rows.mean() / (height - 1)], dtype=np.float32), True


class BallCamera:
    def __init__(self, model, width=128, height=128, name="ball_camera"):
        self.camera_id = model.camera(name).id
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.last_rgb = None

    def capture(self, data):
        self.renderer.update_scene(data, camera=self.camera_id)
        self.last_rgb = self.renderer.render().copy()
        return self.last_rgb

    def close(self):
        self.renderer.close()

