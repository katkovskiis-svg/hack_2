"""
Скрипт обучения модели для предсказания конверсии на сайте «СберАвтоподписка».

Загружает данные из ga_sessions.pkl и ga_hits.pkl, выполняет feature engineering,
обучает CatBoost-классификатор и сохраняет модель в models/model.pkl.

Запуск:
    python app/model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
import joblib
import os

# Пути к данным и модели
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'Проектный практикум (хакатон)')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')

# Целевые действия — события, считающиеся конверсией
TARGET_ACTIONS = [
    'sub_submit_success',
    'sub_car_claim_submit_click',
    'sub_callback_submit_click',
    'sub_car_request_submit_click',
    'sub_custom_question_submit_click',
    'greenday_sub_callback_submit_click',
    'greenday_sub_submit_success',
]


def group_utm_medium(medium):
    """Группировка utm_medium в основные категории трафика."""
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


def encode_top_categories(series, top_n=20):
    """Оставляет top_n самых частых категорий, остальные заменяет на 'other'."""
    top_cats = series.value_counts().head(top_n).index
    return series.where(series.isin(top_cats), 'other')


def train_model():
    """Обучает модель и сохраняет артефакты."""
    print("Загрузка данных...")
    sessions = pd.read_pickle(os.path.join(DATA_PATH, 'ga_sessions.pkl'))
    hits = pd.read_pickle(os.path.join(DATA_PATH, 'ga_hits.pkl'))

    print(f"Sessions: {len(sessions):,} строк, Hits: {len(hits):,} строк")

    # Очистка
    sessions = sessions.drop_duplicates(subset='session_id')
    sessions['visit_date'] = pd.to_datetime(sessions['visit_date'])

    # Заполнение пропусков
    cat_cols = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent',
                'utm_keyword', 'device_category', 'device_os', 'device_brand',
                'device_model', 'device_screen_resolution', 'device_browser',
                'geo_country', 'geo_city']
    for col in cat_cols:
        if col in sessions.columns:
            sessions[col] = sessions[col].fillna('unknown')
            sessions[col] = sessions[col].replace({'': 'unknown', '(not set)': 'unknown', '(none)': 'unknown_none'})

    # Временные признаки
    sessions['visit_hour'] = pd.to_datetime(sessions['visit_time'], format='%H:%M:%S', errors='coerce').dt.hour
    sessions['day_of_week'] = sessions['visit_date'].dt.dayofweek
    sessions['is_weekend'] = sessions['day_of_week'].isin([5, 6]).astype(int)

    # Целевая переменная
    target_sessions = hits[hits['event_action'].isin(TARGET_ACTIONS)]['session_id'].unique()
    sessions['target'] = sessions['session_id'].isin(target_sessions).astype(int)
    print(f"Конверсия: {sessions['target'].mean():.4f}")

    # Агрегация хитов
    print("Агрегация событий из ga_hits...")
    hit_basic = hits.groupby('session_id').agg(
        hit_count=('hit_number', 'max'),
        unique_pages=('hit_page_path', 'nunique'),
    ).reset_index()

    event_counts = hits[hits['hit_type'] == 'event'].groupby('session_id').size().reset_index(name='event_count')

    quiz_sessions = hits[hits['event_action'].isin(['quiz_show', 'quiz_start'])]['session_id'].unique()
    view_card_sessions = hits[hits['event_action'].isin(['view_card', 'view_new_card'])]['session_id'].unique()
    car_claim_sessions = hits[hits['event_action'] == 'sub_car_claim_click']['session_id'].unique()

    hit_features = hit_basic.copy()
    hit_features = hit_features.merge(event_counts, on='session_id', how='left')
    hit_features['event_count'] = hit_features['event_count'].fillna(0).astype(int)
    hit_features['has_quiz'] = hit_features['session_id'].isin(quiz_sessions).astype(int)
    hit_features['has_view_card'] = hit_features['session_id'].isin(view_card_sessions).astype(int)
    hit_features['has_car_claim_click'] = hit_features['session_id'].isin(car_claim_sessions).astype(int)

    # Объединение
    df = sessions.merge(hit_features, on='session_id', how='left')
    for col in ['hit_count', 'unique_pages', 'event_count', 'has_quiz', 'has_view_card', 'has_car_claim_click']:
        df[col] = df[col].fillna(0).astype(int)

    # Feature engineering
    df['utm_medium_grouped'] = df['utm_medium'].apply(group_utm_medium)
    df['utm_source_top'] = encode_top_categories(df['utm_source'], top_n=15)
    df['geo_city_top'] = encode_top_categories(df['geo_city'], top_n=20)
    df['device_os_top'] = encode_top_categories(df['device_os'], top_n=10)
    df['device_browser_top'] = encode_top_categories(df['device_browser'], top_n=10)

    feature_cols = [
        'visit_number', 'visit_hour', 'day_of_week', 'is_weekend',
        'utm_medium_grouped', 'utm_source_top',
        'device_category', 'device_os_top', 'device_browser_top',
        'geo_city_top',
        'hit_count', 'unique_pages', 'event_count',
        'has_quiz', 'has_view_card', 'has_car_claim_click'
    ]

    cat_features = ['utm_medium_grouped', 'utm_source_top', 'device_category',
                    'device_os_top', 'device_browser_top', 'geo_city_top']

    # LabelEncoding
    label_encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Train/test split
    X = df[feature_cols]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Обучение CatBoost
    print("Обучение CatBoost...")
    model = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        eval_metric='AUC', random_seed=42, verbose=100,
        auto_class_weights='Balanced', early_stopping_rounds=50
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100)

    # Оценка
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(classification_report(y_test, model.predict(X_test), target_names=['Нет конверсии', 'Конверсия']))

    # Сохранение
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model_artifacts = {
        'model': model,
        'label_encoders': label_encoders,
        'feature_cols': feature_cols,
        'cat_features': cat_features,
        'roc_auc': roc_auc
    }
    joblib.dump(model_artifacts, MODEL_PATH)
    print(f"Модель сохранена: {MODEL_PATH}")

    return model_artifacts


if __name__ == '__main__':
    train_model()
