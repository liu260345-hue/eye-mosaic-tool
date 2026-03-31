"""
视频眼部马赛克处理工具
使用 MediaPipe Face Landmarker 检测人脸眼部区域，对眼部施加马赛克效果。
通过 FFmpeg pipe 直写保持原始视频画质和音频品质，支持批量处理。
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
import numpy as np
import subprocess
import os
import sys
import json
import shutil
import threading
import time
import ctypes
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from tkinter import dnd
from pathlib import Path
from typing import Optional, Callable

# 模型文件路径（兼容 PyInstaller 打包和源码运行）
if getattr(sys, "frozen", False):
    SCRIPT_DIR = sys._MEIPASS
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _safe_model_path(original_path: str) -> str:
    """MediaPipe C 底层不支持路径含非 ASCII 字符，需复制到临时目录。"""
    try:
        original_path.encode("ascii")
        return original_path
    except UnicodeEncodeError:
        pass
    import tempfile
    safe_dir = os.path.join(tempfile.gettempdir(), "eye_mosaic_models")
    os.makedirs(safe_dir, exist_ok=True)
    safe_path = os.path.join(safe_dir, os.path.basename(original_path))
    if not os.path.isfile(safe_path):
        shutil.copy2(original_path, safe_path)
    return safe_path


FACE_LANDMARKER_MODEL = _safe_model_path(os.path.join(SCRIPT_DIR, "face_landmarker.task"))
YUNET_MODEL = _safe_model_path(os.path.join(SCRIPT_DIR, "face_detection_yunet_2023mar.onnx"))

# 人脸检测的最大分辨率（宽），超过此值会降采样检测以提升速度
DETECTION_MAX_WIDTH = 1280

# 检测丢失后沿用上次位置的最大帧数（防止人已离开画面还在打码）
FALLBACK_MAX_FRAMES = 60

# 左眼+左眉区域关键点索引（合并眼部轮廓 + 眉毛轮廓）
LEFT_EYE_BROW = [
    # 左眼轮廓
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    # 左眉毛
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
]
# 右眼+右眉区域关键点索引
RIGHT_EYE_BROW = [
    # 右眼轮廓
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
    # 右眉毛
    300, 293, 334, 296, 336, 285, 295, 282, 283, 276,
]

# ============================================================
# FFmpeg 工具
# ============================================================

# 运行时目录（exe 所在目录 / 脚本所在目录）
if getattr(sys, "frozen", False):
    _APP_DIR = os.path.dirname(sys.executable)
else:
    _APP_DIR = os.path.dirname(os.path.abspath(__file__))

_FFMPEG_DIR = os.path.join(_APP_DIR, "ffmpeg")


def _auto_download_ffmpeg():
    """首次运行时自动下载 FFmpeg 到 _FFMPEG_DIR。"""
    ffmpeg_exe = os.path.join(_FFMPEG_DIR, "ffmpeg.exe")
    if os.path.isfile(ffmpeg_exe):
        return
    if sys.platform != "win32":
        return  # 非 Windows 不自动下载

    import urllib.request
    import zipfile
    import io

    url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    print("首次运行，正在下载 FFmpeg（约 100MB），请稍候...")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()

        os.makedirs(_FFMPEG_DIR, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                basename = os.path.basename(member)
                if basename in ("ffmpeg.exe", "ffprobe.exe"):
                    target = os.path.join(_FFMPEG_DIR, basename)
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())

        if os.path.isfile(ffmpeg_exe):
            print("FFmpeg 下载完成！")
        else:
            print("FFmpeg 下载失败，请手动安装 FFmpeg。")
    except Exception as e:
        print(f"FFmpeg 下载失败: {e}")
        print("请手动安装 FFmpeg 或将 ffmpeg.exe 放入程序目录的 ffmpeg 文件夹中。")


FFMPEG_PATHS = [
    os.path.join(_FFMPEG_DIR, "ffmpeg.exe"),
    os.path.join(_FFMPEG_DIR, "ffmpeg"),
    shutil.which("ffmpeg"),
    r"C:\Users\lenovo\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe",
]

FFPROBE_PATHS = [
    os.path.join(_FFMPEG_DIR, "ffprobe.exe"),
    os.path.join(_FFMPEG_DIR, "ffprobe"),
    shutil.which("ffprobe"),
    r"C:\Users\lenovo\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffprobe.exe",
]


def _find_executable(paths: list) -> str:
    for p in paths:
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError("找不到 FFmpeg，请确保已安装 FFmpeg 并配置 PATH。")


def get_ffmpeg():
    _auto_download_ffmpeg()
    return _find_executable(FFMPEG_PATHS)


def get_ffprobe():
    _auto_download_ffmpeg()
    return _find_executable(FFPROBE_PATHS)


def get_video_info(video_path: str) -> dict:
    """获取视频元数据。"""
    ffprobe = get_ffprobe()
    cmd = [
        ffprobe, "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        video_path
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    info = json.loads(result.stdout)

    video_stream = None
    audio_stream = None
    for s in info.get("streams", []):
        if s["codec_type"] == "video" and video_stream is None:
            video_stream = s
        elif s["codec_type"] == "audio" and audio_stream is None:
            audio_stream = s

    fmt = info.get("format", {})
    result_info = {
        "duration": float(fmt.get("duration", 0)),
        "bit_rate": fmt.get("bit_rate"),
        "has_audio": audio_stream is not None,
    }

    if video_stream:
        # 优先使用 avg_frame_rate（对 VFR 视频更准确）
        avg_fps_str = video_stream.get("avg_frame_rate", "0/0")
        r_fps_str = video_stream.get("r_frame_rate", "30/1")
        avg_num, avg_den = map(int, avg_fps_str.split("/"))
        r_num, r_den = map(int, r_fps_str.split("/"))
        avg_fps = avg_num / avg_den if avg_den else 0
        r_fps = r_num / r_den if r_den else 30.0
        # avg_frame_rate 合理时优先使用，否则 fallback 到 r_frame_rate
        result_info["fps"] = avg_fps if 1 < avg_fps < 240 else r_fps
        result_info["width"] = int(video_stream.get("width", 0))
        result_info["height"] = int(video_stream.get("height", 0))
        result_info["video_codec"] = video_stream.get("codec_name", "h264")
        result_info["video_bitrate"] = video_stream.get("bit_rate")
        result_info["pix_fmt"] = video_stream.get("pix_fmt", "yuv420p")

    if audio_stream:
        result_info["audio_codec"] = audio_stream.get("codec_name")
        result_info["audio_bitrate"] = audio_stream.get("bit_rate")
        result_info["audio_sample_rate"] = audio_stream.get("sample_rate")

    return result_info


# ============================================================
# 核心处理引擎
# ============================================================


def _region_center(r: tuple[int, int, int, int]) -> tuple[float, float]:
    return (r[0] + r[2] / 2.0, r[1] + r[3] / 2.0)


def _per_face_fallback(
    detected: list[tuple[int, int, int, int]],
    tracked: list[tuple[int, int, int, int, int]],
    max_frames: int,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int, int]]]:
    """
    逐人 fallback 追踪：
    - detected: [(x, y, w, h), ...] 当前帧检出的区域
    - tracked:  [(x, y, w, h, age), ...] age=距上次检出帧数
    返回: (本帧最终遮挡区域, 更新后的 tracked)

    对 detected 中每个区域匹配 tracked 中最近的项并重置 age；
    未被匹配的 tracked 项 age+1，未过期则继续输出遮挡。
    """
    new_tracked: list[tuple[int, int, int, int, int]] = []
    used_tracked = set()
    output_regions: list[tuple[int, int, int, int]] = []

    for dr in detected:
        dcx, dcy = _region_center(dr)
        best_idx, best_dist = -1, float("inf")
        for ti, (tx, ty, tw, th, _) in enumerate(tracked):
            if ti in used_tracked:
                continue
            tcx, tcy = tx + tw / 2.0, ty + th / 2.0
            dist = ((dcx - tcx) ** 2 + (dcy - tcy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best_idx = dist, ti

        if best_idx >= 0 and best_dist < max(dr[2], dr[3]) * 1.5:
            used_tracked.add(best_idx)

        new_tracked.append((dr[0], dr[1], dr[2], dr[3], 0))
        output_regions.append(dr)

    # 未被匹配的 tracked 项：age+1，未过期则保留
    for ti, (tx, ty, tw, th, age) in enumerate(tracked):
        if ti in used_tracked:
            continue
        new_age = age + 1
        if new_age <= max_frames:
            new_tracked.append((tx, ty, tw, th, new_age))
            output_regions.append((tx, ty, tw, th))

    return output_regions, new_tracked

def apply_mosaic(image: np.ndarray, x: int, y: int, w: int, h: int, block_size: int = 10) -> np.ndarray:
    """对图像指定区域施加马赛克效果。"""
    ih, iw = image.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(iw, x + w), min(ih, y + h)

    if x2 <= x1 or y2 <= y1:
        return image

    roi = image[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape[:2]
    if roi_h < 1 or roi_w < 1:
        return image

    small_w = max(1, roi_w // block_size)
    small_h = max(1, roi_h // block_size)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    image[y1:y2, x1:x2] = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    return image


def _ensure_model():
    """确保 MediaPipe 模型文件已下载。"""
    if os.path.isfile(FACE_LANDMARKER_MODEL):
        return
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    urllib.request.urlretrieve(url, FACE_LANDMARKER_MODEL)


def _ensure_yunet_model():
    """确保 YuNet 人脸检测模型已下载。"""
    if os.path.isfile(YUNET_MODEL):
        return
    import urllib.request
    url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    urllib.request.urlretrieve(url, YUNET_MODEL)


def _yunet_eye_regions_for_face(
    face_data: np.ndarray, eye_padding: float, det_scale: float,
) -> list[tuple[int, int, int, int]]:
    """
    从 YuNet 单个人脸检测结果中提取眼部区域。
    利用 YuNet 返回的眼部关键点坐标精确定位，而非估算。
    face_data: YuNet 返回的 15 值数组 [x,y,w,h, right_eye_x,right_eye_y,
               left_eye_x,left_eye_y, nose_x,nose_y, rmouth_x,rmouth_y,
               lmouth_x,lmouth_y, confidence]
    """
    fx, fy, fw, fh = int(face_data[0]), int(face_data[1]), int(face_data[2]), int(face_data[3])

    # 两只眼睛的关键点坐标
    reye_x, reye_y = float(face_data[4]), float(face_data[5])
    leye_x, leye_y = float(face_data[6]), float(face_data[7])

    # 缩放回原始分辨率
    if det_scale < 1.0:
        inv = 1.0 / det_scale
        fx, fy = int(fx * inv), int(fy * inv)
        fw, fh = int(fw * inv), int(fh * inv)
        reye_x, reye_y = reye_x * inv, reye_y * inv
        leye_x, leye_y = leye_x * inv, leye_y * inv

    # 每只眼的遮挡区域尺寸：基于人脸框大小
    region_w = int(fw * 0.35)
    region_h = int(fh * 0.25)

    regions = []
    for ex, ey in [(reye_x, reye_y), (leye_x, leye_y)]:
        # 以眼部关键点为中心，向上偏移以覆盖眉毛
        rx = int(ex - region_w / 2)
        ry = int(ey - region_h * 0.6)
        rw, rh = region_w, region_h

        pad_w = int(rw * eye_padding)
        pad_h = int(rh * eye_padding)
        rx -= pad_w
        ry -= pad_h
        rw += 2 * pad_w
        rh += 2 * pad_h

        regions.append((rx, ry, rw, rh))

    return regions


def _is_face_covered(fx: int, fy: int, fw: int, fh: int,
                     regions: list[tuple[int, int, int, int]]) -> bool:
    """判断人脸是否已被已有的眼部区域覆盖（MediaPipe 已检出该人脸）。"""
    for (rx, ry, rw, rh) in regions:
        rcx = rx + rw / 2.0
        rcy = ry + rh / 2.0
        # 眼部区域中心应在人脸框的上半部分
        if fx <= rcx <= fx + fw and fy <= rcy <= fy + fh * 0.7:
            return True
    return False


def _unique_output_path(path: str) -> str:
    """如果输出文件已存在，自动添加序号避免覆盖。"""
    if not os.path.exists(path):
        return path
    p = Path(path)
    stem, suffix = p.stem, p.suffix
    parent = p.parent
    n = 1
    while True:
        new_path = parent / f"{stem}_{n}{suffix}"
        if not new_path.exists():
            return str(new_path)
        n += 1


def _format_duration(seconds: float) -> str:
    """格式化秒数为 HH:MM:SS 或 MM:SS。"""
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 60}:{s % 60:02d}"


def _format_size(size_bytes: int) -> str:
    """格式化文件大小。"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.1f} MB"
    return f"{size_bytes / 1024 / 1024 / 1024:.2f} GB"


def process_video(
    input_path: str,
    output_path: str,
    mosaic_strength: int = 8,
    eye_padding: float = 0.4,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> bool:
    """
    处理单个视频：检测眼部+眉毛区域并打马赛克。
    - 合并眼睛与眉毛关键点，生成更大的遮挡区域
    - 检测丢失帧自动沿用上一次检测位置，确保每帧都有遮挡
    - FFmpeg pipe 直写，宽高自动修正为偶数
    """
    if progress_callback:
        progress_callback(0, "正在分析视频信息...")

    try:
        video_info = get_video_info(input_path)
    except Exception as e:
        if progress_callback:
            progress_callback(-1, f"无法读取视频信息: {e}")
        return False

    fps = video_info.get("fps", 30.0)
    duration = video_info.get("duration", 0)
    total_frames = int(fps * duration) if duration > 0 else 0

    # 确保输出路径不冲突
    output_path = _unique_output_path(output_path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    import tempfile as _tempfile
    ffmpeg_proc = None
    ffmpeg_stderr_file = None
    cap = None
    landmarker = None
    ffmpeg_completed = False

    try:
        if progress_callback:
            progress_callback(1, "正在加载人脸检测模型...")
        _ensure_model()
        _ensure_yunet_model()

        # MediaPipe: 精确眼部+眉毛关键点定位（近景人脸）
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=20,
            min_face_detection_confidence=0.15,
            min_face_presence_confidence=0.15,
            min_tracking_confidence=0.2,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            if progress_callback:
                progress_callback(-1, "无法打开视频文件")
            return False

        # 用 OpenCV 帧率交叉校验 ffprobe 结果（防止 VFR 视频帧率不准）
        cv_fps = cap.get(cv2.CAP_PROP_FPS)
        if cv_fps and 1 < cv_fps < 240:
            if abs(fps - cv_fps) / max(fps, cv_fps) > 0.2:
                fps = cv_fps
        # 重算总帧数
        total_frames = int(fps * duration) if duration > 0 else 0

        # 读取第一帧获取实际分辨率（避免与 ffprobe 元数据不一致导致输出乱码）
        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            if progress_callback:
                progress_callback(-1, "无法读取视频帧")
            return False

        actual_h, actual_w = first_frame.shape[:2]
        # 确保宽高为偶数（libx264 要求）
        out_w = actual_w if actual_w % 2 == 0 else actual_w - 1
        out_h = actual_h if actual_h % 2 == 0 else actual_h - 1

        # 计算检测用的降采样比例
        if actual_w > DETECTION_MAX_WIDTH:
            det_scale = DETECTION_MAX_WIDTH / actual_w
            det_w = DETECTION_MAX_WIDTH
            det_h = int(actual_h * det_scale)
            det_w = det_w if det_w % 2 == 0 else det_w - 1
            det_h = det_h if det_h % 2 == 0 else det_h - 1
        else:
            det_scale = 1.0
            det_w = actual_w
            det_h = actual_h

        # YuNet 人脸检测器（远景人脸补充，OpenCV DNN 模型）
        yunet_detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL, "", (det_w, det_h),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=20,
        )

        # 构建 FFmpeg pipe 命令（使用实际帧尺寸）
        ffmpeg = get_ffmpeg()
        # 使用分数格式的帧率，避免浮点精度问题
        fps_str = f"{fps:.6f}" if fps != int(fps) else str(int(fps))
        ffmpeg_cmd = [ffmpeg, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{out_w}x{out_h}",
            "-r", fps_str,
            "-i", "pipe:0",
        ]
        if video_info.get("has_audio"):
            ffmpeg_cmd += ["-i", input_path, "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy"]

        # 画质保持：CRF 15 + profile high
        # -vsync cfr 确保输出恒定帧率，-r 输出帧率与输入一致，防止加速/减速
        ffmpeg_cmd += [
            "-c:v", "libx264", "-preset", "medium", "-crf", "15",
            "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-r", fps_str, "-vsync", "cfr",
            "-movflags", "+faststart",
            output_path,
        ]

        ffmpeg_stderr_file = _tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=ffmpeg_stderr_file,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )

        frame_idx = 0
        frame_interval_ms = int(1000 / fps) if fps > 0 else 33
        start_time = time.time()

        # 逐人 fallback 追踪列表: [(x, y, w, h, age), ...]
        face_tracked: list[tuple[int, int, int, int, int]] = []

        if progress_callback:
            progress_callback(2, "正在处理视频帧...")

        # 从已读取的第一帧开始处理，然后继续读取后续帧
        frame = first_frame
        first_frame = None  # 释放引用，减少内存占用
        while frame is not None:
            if cancel_event and cancel_event.is_set():
                if progress_callback:
                    progress_callback(-1, "已取消")
                return False

            # 确保每帧尺寸与 FFmpeg pipe 声明一致（防止乱码）
            fh, fw = frame.shape[:2]
            if fw != out_w or fh != out_h:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

            # 降采样检测
            if det_scale < 1.0:
                det_frame = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_AREA)
            else:
                det_frame = frame

            rgb_frame = cv2.cvtColor(det_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = frame_idx * frame_interval_ms

            try:
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
            except Exception:
                result = None

            # MediaPipe/TFLite 可能在 C 层残留 Python 异常状态，
            # 导致后续 cv2.VideoCapture.read() 抛出 SystemError。
            # 主动调用 PyErr_Clear 清除残留异常。
            ctypes.pythonapi.PyErr_Clear()

            # 释放 MediaPipe 检测中间变量
            del rgb_frame, mp_image

            # MediaPipe 检测结果
            mp_regions: list[tuple[int, int, int, int]] = []

            if result and result.face_landmarks:
                for face_lms in result.face_landmarks:
                    for eye_brow_indices in [LEFT_EYE_BROW, RIGHT_EYE_BROW]:
                        points = np.array([
                            (int(face_lms[i].x * out_w), int(face_lms[i].y * out_h))
                            for i in eye_brow_indices
                        ])
                        ex, ey, ew, eh = cv2.boundingRect(points)

                        pad_w = int(ew * eye_padding)
                        pad_h = int(eh * eye_padding)
                        ex -= pad_w
                        ey -= pad_h
                        ew += 2 * pad_w
                        eh += 2 * pad_h

                        mp_regions.append((ex, ey, ew, eh))

            # YuNet 补充检测：捕获 MediaPipe 漏检的远景人脸
            try:
                _, yunet_faces = yunet_detector.detect(det_frame)
            except Exception:
                yunet_faces = None

            if yunet_faces is not None:
                for face_data in yunet_faces:
                    fx, fy = int(face_data[0]), int(face_data[1])
                    fw, fh = int(face_data[2]), int(face_data[3])
                    # 将检测坐标还原到原始分辨率再判断覆盖
                    if det_scale < 1.0:
                        orig_fx = int(fx / det_scale)
                        orig_fy = int(fy / det_scale)
                        orig_fw = int(fw / det_scale)
                        orig_fh = int(fh / det_scale)
                    else:
                        orig_fx, orig_fy, orig_fw, orig_fh = fx, fy, fw, fh
                    # 仅补充 MediaPipe 未覆盖的人脸
                    if not _is_face_covered(orig_fx, orig_fy, orig_fw, orig_fh, mp_regions):
                        yunet_eye_regions = _yunet_eye_regions_for_face(
                            face_data, eye_padding, det_scale,
                        )
                        mp_regions.extend(yunet_eye_regions)

            # 释放检测用的降采样帧
            if det_scale < 1.0:
                del det_frame

            # 逐人 fallback 追踪
            regions, face_tracked = _per_face_fallback(
                mp_regions, face_tracked, FALLBACK_MAX_FRAMES,
            )

            for (ex, ey, ew, eh) in regions:
                frame = apply_mosaic(frame, ex, ey, ew, eh, mosaic_strength)

            # 确保内存连续后写入 FFmpeg pipe
            try:
                if ffmpeg_proc.poll() is not None:
                    ffmpeg_stderr_file.seek(0)
                    err = ffmpeg_stderr_file.read()
                    if progress_callback:
                        progress_callback(-1, f"FFmpeg 异常退出: {err[-300:]}")
                    return False
                ffmpeg_proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                ffmpeg_proc.stdin.flush()
            except (BrokenPipeError, OSError):
                if progress_callback:
                    ffmpeg_stderr_file.seek(0)
                    err = ffmpeg_stderr_file.read()
                    progress_callback(-1, f"FFmpeg 写入失败: {err[-200:]}")
                return False

            frame_idx += 1

            if progress_callback and total_frames > 0 and frame_idx % 15 == 0:
                pct = min(95, int(frame_idx / total_frames * 95))
                elapsed = time.time() - start_time
                speed = frame_idx / elapsed if elapsed > 0 else 0
                eta = (total_frames - frame_idx) / speed if speed > 0 else 0
                progress_callback(pct, f"帧 {frame_idx}/{total_frames}  {speed:.1f}fps  剩余 {_format_duration(eta)}")

            # 读取下一帧（防御 MediaPipe 残留异常状态导致 SystemError）
            try:
                ret, frame = cap.read()
                if not ret:
                    frame = None
            except SystemError:
                ctypes.pythonapi.PyErr_Clear()
                try:
                    ret, frame = cap.read()
                    if not ret:
                        frame = None
                except Exception:
                    frame = None

        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()

        if ffmpeg_proc.returncode != 0:
            ffmpeg_stderr_file.seek(0)
            stderr_text = ffmpeg_stderr_file.read()
            if progress_callback:
                progress_callback(-1, f"FFmpeg 编码失败: {stderr_text[-300:]}")
            return False

        if progress_callback:
            progress_callback(100, "处理完成")
        ffmpeg_completed = True
        return True

    except Exception as e:
        if progress_callback:
            progress_callback(-1, f"处理出错: {e}")
        return False
    finally:
        # 释放资源，顺序重要：先关闭 landmarker 释放 TFLite 内存
        if landmarker is not None:
            try:
                landmarker.close()
            except Exception:
                pass
            landmarker = None
        if cap is not None:
            cap.release()
            cap = None
        if ffmpeg_proc is not None:
            try:
                if ffmpeg_proc.stdin and not ffmpeg_proc.stdin.closed:
                    ffmpeg_proc.stdin.close()
                # 仅在非正常完成时终止 FFmpeg 进程
                if not ffmpeg_completed and ffmpeg_proc.poll() is None:
                    ffmpeg_proc.kill()
                    ffmpeg_proc.wait()
            except Exception:
                pass
        if ffmpeg_stderr_file is not None:
            try:
                ffmpeg_stderr_file.close()
            except Exception:
                pass
        # 强制回收 MediaPipe/TFLite 释放的内存，防止连续批处理时内存不足
        import gc
        gc.collect()


# ============================================================
# GUI 界面
# ============================================================

class VideoTask:
    """单个视频处理任务。"""
    def __init__(self, input_path: str, file_info: str = ""):
        self.input_path = input_path
        self.output_path = ""
        self.status = "等待中"
        self.progress = 0
        self.filename = os.path.basename(input_path)
        self.file_info = file_info  # 时长/分辨率等摘要


class EyeMosaicApp:
    """视频眼部马赛克工具 GUI 应用。"""

    SUPPORTED_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".ts", ".mts"}

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("视频眼部马赛克工具")
        self.root.geometry("960x700")
        self.root.minsize(800, 580)

        self.tasks: list[VideoTask] = []
        self.output_dir = ""
        self.is_processing = False
        self.cancel_event = threading.Event()

        self._build_ui()
        self._setup_drag_drop()

    def _build_ui(self):
        style = ttk.Style()
        style.configure("Start.TButton", font=("", 10, "bold"))

        # ---- 顶部操作栏 ----
        top_frame = ttk.LabelFrame(self.root, text="文件操作", padding=8)
        top_frame.pack(fill=tk.X, padx=10, pady=(10, 4))

        btn_row = ttk.Frame(top_frame)
        btn_row.pack(fill=tk.X)

        self.btn_add = ttk.Button(btn_row, text="添加视频", command=self._add_files)
        self.btn_add.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_add_folder = ttk.Button(btn_row, text="添加文件夹", command=self._add_folder)
        self.btn_add_folder.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_remove = ttk.Button(btn_row, text="移除选中", command=self._remove_selected)
        self.btn_remove.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_clear = ttk.Button(btn_row, text="清空列表", command=self._clear_list)
        self.btn_clear.pack(side=tk.LEFT, padx=(0, 16))

        ttk.Separator(btn_row, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(btn_row, text="输出目录:").pack(side=tk.LEFT)
        self.output_dir_var = tk.StringVar(value="(与源文件同目录)")
        self.output_dir_label = ttk.Label(btn_row, textvariable=self.output_dir_var, width=32,
                                          relief="sunken", padding=2)
        self.output_dir_label.pack(side=tk.LEFT, padx=4)
        self.btn_output = ttk.Button(btn_row, text="选择...", command=self._select_output_dir)
        self.btn_output.pack(side=tk.LEFT)
        self.btn_open_output = ttk.Button(btn_row, text="打开", command=self._open_output_dir)
        self.btn_open_output.pack(side=tk.LEFT, padx=(4, 0))

        # ---- 参数设置 ----
        param_frame = ttk.LabelFrame(self.root, text="参数设置", padding=8)
        param_frame.pack(fill=tk.X, padx=10, pady=4)

        ttk.Label(param_frame, text="马赛克强度:").pack(side=tk.LEFT)
        self.strength_var = tk.IntVar(value=8)
        ttk.Spinbox(param_frame, from_=2, to=20, textvariable=self.strength_var, width=4).pack(side=tk.LEFT, padx=(2, 4))
        ttk.Label(param_frame, text="(越小=块越大)", foreground="gray").pack(side=tk.LEFT, padx=(0, 16))

        ttk.Label(param_frame, text="眼部扩展:").pack(side=tk.LEFT)
        self.padding_var = tk.DoubleVar(value=0.5)
        ttk.Spinbox(param_frame, from_=0.0, to=2.0, increment=0.1, textvariable=self.padding_var, width=4).pack(side=tk.LEFT, padx=(2, 4))
        ttk.Label(param_frame, text="(遮挡范围倍数)", foreground="gray").pack(side=tk.LEFT)

        # ---- 任务列表 ----
        list_frame = ttk.LabelFrame(self.root, text="任务列表 (支持拖拽文件到此处)", padding=4)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        columns = ("filename", "info", "status", "progress")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode=tk.EXTENDED)
        self.tree.heading("filename", text="文件名")
        self.tree.heading("info", text="详情")
        self.tree.heading("status", text="状态")
        self.tree.heading("progress", text="进度")
        self.tree.column("filename", width=340, minwidth=200)
        self.tree.column("info", width=200, minwidth=120)
        self.tree.column("status", width=250, minwidth=120)
        self.tree.column("progress", width=70, minwidth=50)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # ---- 底部状态/按钮 ----
        bottom_frame = ttk.Frame(self.root, padding=8)
        bottom_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.overall_progress = ttk.Progressbar(bottom_frame, mode="determinate", length=400)
        self.overall_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        self.status_label = ttk.Label(bottom_frame, text="就绪", width=28, anchor="w")
        self.status_label.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_cancel = ttk.Button(bottom_frame, text="取消", command=self._cancel_processing, state=tk.DISABLED)
        self.btn_cancel.pack(side=tk.RIGHT, padx=(4, 0))

        self.btn_start = ttk.Button(bottom_frame, text="开始处理", style="Start.TButton",
                                    command=self._start_processing)
        self.btn_start.pack(side=tk.RIGHT)

    def _setup_drag_drop(self):
        """尝试设置 Windows 拖拽支持 (tkinterdnd2 可选)。"""
        try:
            # 如果安装了 tkinterdnd2 则支持拖拽
            from tkinterdnd2 import DND_FILES
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self._on_drop)
        except ImportError:
            pass  # 没有 tkinterdnd2 则跳过，不影响核心功能

    def _on_drop(self, event):
        """处理拖拽文件。"""
        files = self.root.tk.splitlist(event.data)
        added = 0
        for f in files:
            f = f.strip('{}')
            p = Path(f)
            if p.is_dir():
                for sub in p.rglob("*"):
                    if sub.suffix.lower() in self.SUPPORTED_FORMATS:
                        self._add_single_file(str(sub))
                        added += 1
            elif p.suffix.lower() in self.SUPPORTED_FORMATS:
                self._add_single_file(str(p))
                added += 1
        if added == 0:
            messagebox.showinfo("提示", "未找到支持的视频文件。")

    def _get_file_info(self, path: str) -> str:
        """获取文件的摘要信息。"""
        try:
            info = get_video_info(path)
            parts = []
            w, h = info.get("width", 0), info.get("height", 0)
            if w and h:
                parts.append(f"{w}x{h}")
            dur = info.get("duration", 0)
            if dur > 0:
                parts.append(_format_duration(dur))
            size = os.path.getsize(path)
            parts.append(_format_size(size))
            return "  ".join(parts)
        except Exception:
            try:
                return _format_size(os.path.getsize(path))
            except Exception:
                return ""

    def _add_single_file(self, path: str):
        """添加单个文件到任务列表（避免重复）。"""
        norm = os.path.normpath(path)
        for t in self.tasks:
            if os.path.normpath(t.input_path) == norm:
                return
        file_info = self._get_file_info(path)
        task = VideoTask(path, file_info)
        self.tasks.append(task)
        self.tree.insert("", tk.END, values=(task.filename, task.file_info, task.status, "0%"))

    def _add_files(self):
        filetypes = [
            ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v *.ts *.mts"),
            ("所有文件", "*.*"),
        ]
        files = filedialog.askopenfilenames(title="选择视频文件", filetypes=filetypes)
        for f in files:
            if Path(f).suffix.lower() in self.SUPPORTED_FORMATS:
                self._add_single_file(f)

    def _add_folder(self):
        folder = filedialog.askdirectory(title="选择包含视频的文件夹")
        if not folder:
            return
        count = 0
        for f in Path(folder).rglob("*"):
            if f.suffix.lower() in self.SUPPORTED_FORMATS:
                self._add_single_file(str(f))
                count += 1
        if count == 0:
            messagebox.showinfo("提示", "所选文件夹中未找到支持的视频文件。")

    def _remove_selected(self):
        selected = self.tree.selection()
        if not selected:
            return
        indices = sorted([self.tree.index(item) for item in selected], reverse=True)
        for idx in indices:
            if idx < len(self.tasks) and self.tasks[idx].status in ("等待中", "完成") or \
               self.tasks[idx].status.startswith("失败"):
                self.tasks.pop(idx)
        self._refresh_tree()

    def _clear_list(self):
        if self.is_processing:
            messagebox.showwarning("警告", "处理进行中，无法清空列表。")
            return
        self.tasks.clear()
        self._refresh_tree()

    def _refresh_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for task in self.tasks:
            self.tree.insert("", tk.END, values=(task.filename, task.file_info, task.status, f"{task.progress}%"))

    def _select_output_dir(self):
        d = filedialog.askdirectory(title="选择输出目录")
        if d:
            self.output_dir = d
            self.output_dir_var.set(d)

    def _open_output_dir(self):
        """打开输出目录。"""
        target = self.output_dir
        if not target:
            # 如果有已完成的任务，打开第一个完成任务的所在目录
            for t in self.tasks:
                if t.status == "完成" and t.output_path:
                    target = os.path.dirname(t.output_path)
                    break
        if target and os.path.isdir(target):
            if sys.platform == "win32":
                os.startfile(target)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", target])
            else:
                subprocess.Popen(["xdg-open", target])
        else:
            messagebox.showinfo("提示", "没有可打开的输出目录。")

    def _get_output_path(self, task: VideoTask) -> str:
        src = Path(task.input_path)
        name = src.stem + "_眼部马赛克" + src.suffix
        if self.output_dir:
            return os.path.join(self.output_dir, name)
        return os.path.join(str(src.parent), name)

    def _update_task_in_tree(self, task_index: int, task: VideoTask):
        children = self.tree.get_children()
        if task_index < len(children):
            item = children[task_index]
            self.tree.item(item, values=(task.filename, task.file_info, task.status, f"{task.progress}%"))

    def _start_processing(self):
        pending = [i for i, t in enumerate(self.tasks) if t.status in ("等待中",) or t.status.startswith("失败")]
        if not pending:
            messagebox.showinfo("提示", "没有待处理的任务。")
            return

        self.is_processing = True
        self.cancel_event.clear()
        self.overall_progress.config(value=0)
        self._set_buttons_enabled(False)

        thread = threading.Thread(target=self._process_worker, args=(pending,), daemon=True)
        thread.start()

    def _set_buttons_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.btn_start.config(state=state)
        self.btn_add.config(state=state)
        self.btn_add_folder.config(state=state)
        self.btn_clear.config(state=state)
        self.btn_cancel.config(state=tk.DISABLED if enabled else tk.NORMAL)

    def _process_worker(self, task_indices: list[int]):
        total = len(task_indices)
        completed = 0

        for idx in task_indices:
            if self.cancel_event.is_set():
                break

            task = self.tasks[idx]
            task.status = "处理中..."
            task.progress = 0
            task.output_path = self._get_output_path(task)

            # 用 completed_snapshot 避免闭包引用问题
            completed_snapshot = completed

            self.root.after(0, self._update_task_in_tree, idx, task)
            self.root.after(0, self.status_label.config, {"text": f"处理 {completed_snapshot + 1}/{total}"})

            def progress_cb(pct, msg, _idx=idx, _task=task, _done=completed_snapshot):
                if pct >= 0:
                    _task.progress = int(pct)
                    _task.status = msg
                else:
                    _task.status = msg
                self.root.after(0, self._update_task_in_tree, _idx, _task)
                overall_pct = int((_done * 100 + max(0, pct)) / total)
                self.root.after(0, lambda p=overall_pct: self.overall_progress.config(value=p))

            success = process_video(
                input_path=task.input_path,
                output_path=task.output_path,
                mosaic_strength=self.strength_var.get(),
                eye_padding=self.padding_var.get(),
                progress_callback=progress_cb,
                cancel_event=self.cancel_event,
            )

            if success:
                task.status = "完成"
                task.progress = 100
            elif not self.cancel_event.is_set():
                task.status = "失败: " + task.status.replace("处理中...", "")
                task.progress = 0

            self.root.after(0, self._update_task_in_tree, idx, task)
            completed += 1

        self.root.after(0, self._on_processing_done, completed, total)

    def _on_processing_done(self, completed: int, total: int):
        self.is_processing = False
        self._set_buttons_enabled(True)
        self.overall_progress.config(value=100)

        if self.cancel_event.is_set():
            self.status_label.config(text="已取消")
            messagebox.showinfo("提示", f"已取消，完成 {completed}/{total} 个。")
        else:
            self.status_label.config(text=f"全部完成 ({completed}/{total})")
            messagebox.showinfo("完成", f"已处理 {completed} 个视频文件。")

    def _cancel_processing(self):
        self.cancel_event.set()
        self.status_label.config(text="正在取消...")

    def run(self):
        self.root.mainloop()


# ============================================================
# CLI 模式
# ============================================================

def cli_mode():
    import argparse
    parser = argparse.ArgumentParser(description="视频眼部马赛克工具")
    parser.add_argument("inputs", nargs="+", help="输入视频文件路径")
    parser.add_argument("-o", "--output-dir", default="", help="输出目录（默认与源文件同目录）")
    parser.add_argument("-s", "--strength", type=int, default=8, help="马赛克强度（2-20, 默认8）")
    parser.add_argument("-p", "--padding", type=float, default=0.5, help="眼部扩展比例（默认0.5）")
    args = parser.parse_args()

    for input_path in args.inputs:
        if not os.path.isfile(input_path):
            print(f"[跳过] 文件不存在: {input_path}")
            continue

        src = Path(input_path)
        name = src.stem + "_眼部马赛克" + src.suffix
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, name)
        else:
            output_path = os.path.join(str(src.parent), name)

        print(f"[处理] {input_path}")

        def cli_progress(pct, msg):
            if pct >= 0:
                filled = int(pct // 2)
                bar = "\u2588" * filled + "\u2591" * (50 - filled)
                print(f"\r  [{bar}] {pct:.0f}% {msg}", end="", flush=True)
            else:
                print(f"\n  [错误] {msg}")

        success = process_video(
            input_path=input_path,
            output_path=output_path,
            mosaic_strength=args.strength,
            eye_padding=args.padding,
            progress_callback=cli_progress,
        )

        if success:
            print(f"\n  [完成] -> {output_path}")
        else:
            print(f"\n  [失败] {input_path}")

    print("\n全部处理完毕。")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--gui"):
        cli_mode()
    else:
        app = EyeMosaicApp()
        app.run()
