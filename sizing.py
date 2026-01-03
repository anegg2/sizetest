import cv2
import mediapipe as mp
from mediapipe import tasks
from pathlib import Path
from dataclasses import dataclass

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "pose_landmarker_full.task"

BaseOptions = tasks.BaseOptions
VisionRunningMode = tasks.vision.RunningMode
PoseLandmarker = tasks.vision.PoseLandmarker
PoseLandmarkerOptions = tasks.vision.PoseLandmarkerOptions
mp_image = mp.Image


@dataclass
class Measurements:
    waist_girth: float      # D: обхват пояса (линия ремня), см
    hip_girth: float        # E: обхват бёдер (самая широкая часть), см
    pants_length: float     # F: длина брюк (внутренняя поверхность ноги), см
    debug_path: Path | None = None


# ---------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----------

def get_body_height_pixels(landmarks, image_height: int) -> float:
    nose = landmarks[0]
    left_ankle = landmarks[27]
    right_ankle = landmarks[28]
    left_foot = landmarks[31]
    right_foot = landmarks[32]

    foot_y = max(left_ankle.y, right_ankle.y, left_foot.y, right_foot.y)
    head_y = nose.y

    return abs((foot_y - head_y) * image_height)


def get_leg_length_pixels(landmarks, image_height: int) -> float:
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    left_ankle = landmarks[27]
    right_ankle = landmarks[28]

    waist_y = (left_hip.y + right_hip.y) / 2 * image_height
    ankle_y = (left_ankle.y + right_ankle.y) / 2 * image_height

    return abs(ankle_y - waist_y)


def estimate_waist_and_hip(landmarks, image_width: int, image_height: int) -> tuple[float, float]:
    """
    Возвращает (waist_width_px, hip_width_px).

    D (пояс):
      - чуть выше линии бёдер, ближе к талии / ремню;
      - делаем как 0.9 от ширины бёдер (пояс почти всегда уже бёдер).

    E (бёдра):
      - ширина на уровне лэндмарков бедра (left/right hip);
      - это самая широкая часть таза. [file:178]
    """
    left_hip = landmarks[23]
    right_hip = landmarks[24]

    x_left = left_hip.x * image_width
    x_right = right_hip.x * image_width
    hip_width_px = abs(x_right - x_left)

    # Пояс чуть уже бёдер — эмпирически 0.85–0.95, берём 0.9 для старта.
    waist_width_px = hip_width_px * 0.9

    return waist_width_px, hip_width_px


# ---------- ОСНОВНАЯ ФУНКЦИЯ ИЗМЕРЕНИЙ ----------

def estimate_measurements(image_path: Path, height_cm: int) -> Measurements:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Не смог открыть изображение: {image_path}")

    h, w, _ = image_bgr.shape
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = mp_image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(image)

    if not result.pose_landmarks:
        raise RuntimeError("Поза не найдена, попробуй другое фото (полный рост).")

    landmarks = result.pose_landmarks[0]

    body_height_px = get_body_height_pixels(landmarks, h)
    if body_height_px < 10:
        raise RuntimeError("Слишком маленькая высота тела, проверь фото.")

    scale = height_cm / body_height_px  # см на пиксель по вертикали

    leg_px = get_leg_length_pixels(landmarks, h)
    waist_width_px, hip_width_px = estimate_waist_and_hip(landmarks, w, h)

    pants_length_cm = leg_px * scale
    waist_girth_cm = waist_width_px * scale * 2   # D
    hip_girth_cm = hip_width_px * scale * 2       # E

    # debug-картинка
    debug_image = image_bgr.copy()
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(debug_image, (x, y), 3, (0, 255, 0), -1)

    debug_path = image_path.with_name(image_path.stem + "_debug.jpg")
    cv2.imwrite(str(debug_path), debug_image)

    return Measurements(
        waist_girth=waist_girth_cm,
        hip_girth=hip_girth_cm,
        pants_length=pants_length_cm,
        debug_path=debug_path,
    )


# ---------- РАЗМЕРНАЯ МАТРИЦА 7×3 (D × A) ----------

SIZE_MATRIX = {
    85: {"low": 92,  "mid": 46,  "high": 146},
    89: {"low": 96,  "mid": 48,  "high": 148},
    93: {"low": 100, "mid": 50,  "high": 150},
    97: {"low": 104, "mid": 52,  "high": 152},
    101: {"low": 108, "mid": 54, "high": 154},
    105: {"low": 112, "mid": 56, "high": 156},
    109: {"low": 116, "mid": 58, "high": 158},
}
D_VALUES = sorted(SIZE_MATRIX.keys())


def pick_nearest_d(d_cm: float) -> int:
    return min(D_VALUES, key=lambda base: abs(base - d_cm))


def get_height_band(height_cm: int) -> str:
    if 163 <= height_cm <= 173:
        return "low"
    if 174 <= height_cm <= 184:
        return "mid"
    if 185 <= height_cm <= 195:
        return "high"
    if height_cm < 163:
        return "low"
    return "high"


def recommend_yasneg(meas: Measurements, height_cm: int) -> int:
    """
    Выбор размера по:
    - D (пояс) -> строка матрицы,
    - A (рост) -> колонка low/mid/high. [file:179]
    E (бёдра) можно позже использовать для тонкой коррекции.
    """
    waist_estimate = meas.waist_girth
    d = pick_nearest_d(waist_estimate)
    band = get_height_band(height_cm)
    return SIZE_MATRIX[d][band]


# ---------- ОТОБРАЖЕНИЕ РАЗМЕРА ----------

SIZE_LABELS = {
    92: "S46",
    96: "S48",
    100: "S50",
    104: "S52",
    108: "S54",
    112: "S56",
    116: "S58",
    46: "M46",
    48: "M48",
    50: "M50",
    52: "M52",
    54: "M54",
    56: "M56",
    58: "M58",
    146: "L46",
    148: "L48",
    150: "L50",
    152: "L52",
    154: "L54",
    156: "L56",
    158: "L58",
}


def format_size_display(raw_size: int) -> str:
    label = SIZE_LABELS.get(raw_size)
    if label:
        return f"{raw_size} ({label})"
    return str(raw_size)
