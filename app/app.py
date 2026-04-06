"""
FastAPI-приложение для предсказания конверсии на сайте «СберАвтоподписка».

Принимает данные о визите пользователя и возвращает вероятность конверсии.

Запуск:
    uvicorn app.app:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

# Путь к сохранённой модели
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')

app = FastAPI(
    title="СберАвтоподписка — Предсказание конверсии",
    description="API для предсказания вероятности целевого действия пользователя на сайте",
    version="1.0.0"
)

# Загрузка модели при старте приложения
model_artifacts = None


def load_model():
    """Загрузка модели и артефактов из файла."""
    global model_artifacts
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}. Сначала запустите: python app/model.py")
    model_artifacts = joblib.load(MODEL_PATH)
    print(f"Модель загружена. ROC-AUC на тесте: {model_artifacts['roc_auc']:.4f}")


@app.on_event("startup")
def startup_event():
    load_model()


# Группировка utm_medium (дублируем логику из model.py для автономности)
def group_utm_medium(medium: str) -> str:
    mapping = {
        'organic': 'organic', 'referral': 'referral', '(none)': 'direct',
        'unknown_none': 'direct', 'cpc': 'cpc', 'cpm': 'cpm',
        'banner': 'banner', 'email': 'email', 'push': 'push',
        'stories': 'social', 'cpv': 'cpv', 'smm': 'social',
        'blogger_channel': 'social', 'blogger_stories': 'social',
        'cpa': 'cpa', 'tg': 'social', 'app': 'app',
        'post': 'social', 'smartbanner': 'banner'
    }
    return mapping.get(medium, 'other')


class VisitInput(BaseModel):
    """Входные данные визита для предсказания."""
    visit_number: int = Field(1, description="Порядковый номер визита клиента")
    visit_hour: int = Field(12, description="Час визита (0-23)")
    day_of_week: int = Field(0, description="День недели (0=Пн, 6=Вс)")
    is_weekend: int = Field(0, description="Выходной день (0/1)")
    utm_medium: str = Field("cpc", description="Тип привлечения (cpc, organic, banner, ...)")
    utm_source: str = Field("unknown", description="Канал привлечения")
    device_category: str = Field("desktop", description="Тип устройства (desktop, mobile, tablet)")
    device_os: str = Field("Windows", description="ОС устройства")
    device_browser: str = Field("Chrome", description="Браузер")
    geo_city: str = Field("Moscow", description="Город")
    hit_count: int = Field(1, description="Количество хитов в сессии")
    unique_pages: int = Field(1, description="Количество уникальных страниц")
    event_count: int = Field(0, description="Количество событий типа 'event'")
    has_quiz: int = Field(0, description="Был ли quiz (0/1)")
    has_view_card: int = Field(0, description="Был ли просмотр карточки авто (0/1)")
    has_car_claim_click: int = Field(0, description="Был ли клик на заявку (0/1)")

    class Config:
        json_schema_extra = {
            "example": {
                "visit_number": 1,
                "visit_hour": 14,
                "day_of_week": 2,
                "is_weekend": 0,
                "utm_medium": "cpc",
                "utm_source": "yandex",
                "device_category": "desktop",
                "device_os": "Windows",
                "device_browser": "Chrome",
                "geo_city": "Moscow",
                "hit_count": 15,
                "unique_pages": 5,
                "event_count": 10,
                "has_quiz": 1,
                "has_view_card": 1,
                "has_car_claim_click": 0
            }
        }


class PredictionOutput(BaseModel):
    """Результат предсказания."""
    prediction: int = Field(description="Предсказание (0 — нет конверсии, 1 — конверсия)")
    probability: float = Field(description="Вероятность конверсии (0.0 — 1.0)")


def safe_label_encode(encoder, value: str) -> int:
    """Безопасное кодирование: если значение неизвестно, возвращает код для 'other'."""
    try:
        return encoder.transform([value])[0]
    except ValueError:
        if 'other' in encoder.classes_:
            return encoder.transform(['other'])[0]
        return 0


@app.post("/predict", response_model=PredictionOutput)
def predict(visit: VisitInput):
    """
    Предсказание конверсии для визита.

    Принимает характеристики визита пользователя и возвращает
    бинарное предсказание (0/1) и вероятность конверсии.
    """
    if model_artifacts is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    model = model_artifacts['model']
    label_encoders = model_artifacts['label_encoders']
    feature_cols = model_artifacts['feature_cols']

    # Подготовка признаков
    utm_medium_grouped = group_utm_medium(visit.utm_medium)

    # Кодирование категориальных признаков
    features = {
        'visit_number': visit.visit_number,
        'visit_hour': visit.visit_hour,
        'day_of_week': visit.day_of_week,
        'is_weekend': visit.is_weekend,
        'utm_medium_grouped': safe_label_encode(label_encoders['utm_medium_grouped'], utm_medium_grouped),
        'utm_source_top': safe_label_encode(label_encoders['utm_source_top'], visit.utm_source),
        'device_category': safe_label_encode(label_encoders['device_category'], visit.device_category),
        'device_os_top': safe_label_encode(label_encoders['device_os_top'], visit.device_os),
        'device_browser_top': safe_label_encode(label_encoders['device_browser_top'], visit.device_browser),
        'geo_city_top': safe_label_encode(label_encoders['geo_city_top'], visit.geo_city),
        'hit_count': visit.hit_count,
        'unique_pages': visit.unique_pages,
        'event_count': visit.event_count,
        'has_quiz': visit.has_quiz,
        'has_view_card': visit.has_view_card,
        'has_car_claim_click': visit.has_car_claim_click,
    }

    # Формируем вектор признаков в правильном порядке
    X = np.array([[features[col] for col in feature_cols]])

    # Предсказание
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    return PredictionOutput(prediction=prediction, probability=round(probability, 4))


@app.get("/health")
def health():
    """Проверка состояния сервиса."""
    return {
        "status": "ok",
        "model_loaded": model_artifacts is not None,
        "roc_auc": model_artifacts['roc_auc'] if model_artifacts else None
    }
