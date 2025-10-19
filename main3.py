import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os

# ============================
# 1️⃣ Пути к файлам
# ============================
MODEL_PATH = "price_optimizer_nn_model.h5"
INPUT_CSV = "test.csv"
OUTPUT_CSV = "predictions.csv"

# ============================
# 2️⃣ Загрузка модели
# ============================
print("🔹 Загружаем модель...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Модель не найдена: {MODEL_PATH}")

model = load_model(MODEL_PATH, compile=False)
expected_features = model.input_shape[1]
print(f"✅ Модель загружена. Ожидает {expected_features} признаков.")

# ============================
# 3️⃣ Чтение CSV
# ============================
print("🔹 Загружаем исходные данные...")
df = pd.read_csv(INPUT_CSV)
print(f"✅ Загружено {len(df)} строк.")

# ============================
# 4️⃣ Подготовка данных
# ============================
processed_df = df.copy()

# Временные признаки
processed_df['order_timestamp'] = pd.to_datetime(processed_df['order_timestamp'])
processed_df['hour'] = processed_df['order_timestamp'].dt.hour
processed_df['day_of_week'] = processed_df['order_timestamp'].dt.dayofweek
processed_df['month'] = processed_df['order_timestamp'].dt.month
processed_df['is_weekend'] = processed_df['day_of_week'].isin([5, 6]).astype(int)
processed_df['is_night'] = ((processed_df['hour'] >= 23) | (processed_df['hour'] <= 6)).astype(int)

# Признаки поездки
processed_df['distance_km'] = processed_df['distance_in_meters'] / 1000
processed_df['duration_min'] = processed_df['duration_in_seconds'] / 60
processed_df['pickup_km'] = processed_df['pickup_in_meters'] / 1000
processed_df['pickup_time_min'] = processed_df['pickup_in_seconds'] / 60

# Расчет скорости
processed_df['speed_kmh'] = processed_df['distance_km'] / (processed_df['duration_min'] / 60 + 1e-5)

# Ценовые признаки
processed_df['price_increase'] = processed_df['price_bid_local'] - processed_df['price_start_local']
processed_df['price_increase_pct'] = (processed_df['price_bid_local'] / processed_df['price_start_local'] - 1) * 100
processed_df['price_per_km'] = processed_df['price_start_local'] / (processed_df['distance_km'] + 1e-5)

# Признаки водителя
processed_df['driver_experience_days'] = (pd.to_datetime(processed_df['order_timestamp']) - 
                                         pd.to_datetime(processed_df['driver_reg_date'])).dt.days
processed_df['driver_rating'] = processed_df['driver_rating'].fillna(4.5)

# ТОЧНО 16 ПРИЗНАКОВ
feature_columns = [
    'hour', 'day_of_week', 'month', 'is_weekend', 'is_night',
    'distance_km', 'duration_min', 'speed_kmh', 'pickup_km', 'pickup_time_min',
    'price_start_local', 'price_bid_local', 'price_increase', 'price_increase_pct',
    'price_per_km', 'driver_rating'
]

for col in feature_columns:
    if col not in processed_df.columns:
        processed_df[col] = 0

X = processed_df[feature_columns].values.astype('float32')
print(f"📊 Форма данных: {X.shape}")

# ============================
# 5️⃣ Предсказание
# ============================
print("🔹 Масштабируем и предсказываем...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
predictions = model.predict(X_scaled, verbose=0).flatten()

# ============================
# 6️⃣ Добавляем ТОЛЬКО финальные предсказания
# ============================
print("🔹 Добавляем предсказания к исходным данным...")

# ТОЛЬКО 3 финальные колонки с предсказаниями
df['predicted_acceptance_probability'] = np.round(predictions, 4)
df['predicted_will_accept'] = (predictions > 0.5).astype(int)
df['expected_revenue'] = np.round(df['price_bid_local'] * predictions, 2)

# ============================
# 7️⃣ Сохраняем ТОЛЬКО исходные данные + предсказания
# ============================
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Готово! Файл сохранён: {OUTPUT_CSV}")
print(f"📊 Статистика предсказаний:")
print(f"   Средняя вероятность принятия: {df['predicted_acceptance_probability'].mean():.2%}")
print(f"   Прогнозируемых принятий: {df['predicted_will_accept'].sum()} записей")
print(f"   Средний ожидаемый доход: {df['expected_revenue'].mean():.0f} ₽")

print("\n📋 Добавленные колонки:")
print("   - predicted_acceptance_probability: вероятность принятия цены (0-1)")
print("   - predicted_will_accept: примет ли цену (1 = да, 0 = нет)")
print("   - expected_revenue: ожидаемый доход (цена × вероятность)")

print("\n🔍 Пример данных:")
sample = df[['order_id', 'price_start_local', 'price_bid_local', 'predicted_acceptance_probability', 'expected_revenue']].head()
print(sample.to_string(index=False))