import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

def load_model_and_scaler():
    """Загрузка модели и скейлера"""
    try:
        model = load_model('price_optimizer_nn_model.h5')
        print("✅ Модель загружена")
        
        # Пробуем загрузить скейлер
        try:
            import pickle
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            print("✅ Скейлер загружен")
        except:
            print("⚠️  Скейлер не найден, создаем новый")
            scaler = StandardScaler()
            
        return model, scaler
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return None, None

def prepare_features(df):
    """Подготовка признаков для дообучения"""
    processed_df = df.copy()
    
    # Проверяем наличие целевой переменной
    if 'is_done' not in processed_df.columns:
        raise ValueError("❌ Для дообучения нужна колонка 'is_done' с реальными результатами")
    
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
    
    # Целевая переменная - принял ли пассажир повышенную цену
    processed_df['target'] = ((processed_df['is_done'] == 1) & 
                             (processed_df['price_bid_local'] > processed_df['price_start_local'])).astype(int)
    
    # ТОЧНО 16 ПРИЗНАКОВ
    feature_columns = [
        'hour', 'day_of_week', 'month', 'is_weekend', 'is_night',
        'distance_km', 'duration_min', 'speed_kmh', 'pickup_km', 'pickup_time_min',
        'price_start_local', 'price_bid_local', 'price_increase', 'price_increase_pct',
        'price_per_km', 'driver_rating'
    ]
    
    # Создаем недостающие колонки
    for col in feature_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0
    
    X = processed_df[feature_columns].values.astype('float32')
    y = processed_df['target'].values
    
    return X, y, processed_df

def fine_tune_model(model, X, y, scaler, epochs=10, batch_size=32):
    """Дообучение модели"""
    print("🎯 Начинаем дообучение модели...")
    
    # Масштабируем данные
    X_scaled = scaler.transform(X)
    
    # Разделяем на train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Данные для дообучения:")
    print(f"   Обучающая выборка: {X_train.shape[0]} записей")
    print(f"   Валидационная выборка: {X_val.shape[0]} записей")
    print(f"   Положительных примеров: {y.sum()} ({y.mean():.1%})")
    
    # Компилируем модель с меньшим learning rate для тонкой настройки
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Меньше LR для дообучения
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Дообучение
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Оценка качества
    train_loss, train_acc, train_precision, train_recall = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc, val_precision, val_recall = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"\n📈 Результаты дообучения:")
    print(f"   Train Accuracy: {train_acc:.3f}, Precision: {train_precision:.3f}, Recall: {train_recall:.3f}")
    print(f"   Val Accuracy: {val_acc:.3f}, Precision: {val_precision:.3f}, Recall: {val_recall:.3f}")
    
    return model, history

def save_updated_model(model, scaler):
    """Сохранение обновленной модели и скейлера"""
    # Сохраняем модель
    model.save('price_optimizer_nn_model_updated.h5')
    print("✅ Обновленная модель сохранена как 'price_optimizer_nn_model_updated.h5'")
    
    # Сохраняем скейлер
    import pickle
    with open('scaler_updated.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Обновленный скейлер сохранен как 'scaler_updated.pkl'")
    
    # Также сохраняем как основные файлы (перезаписываем)
    model.save('price_optimizer_nn_model.h5')
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Модель и скейлер также сохранены как основные файлы")

def main():
    """Основная функция дообучения"""
    print("🔄 ЗАПУСК ПРОЦЕССА ДООБУЧЕНИЯ МОДЕЛИ")
    print("=" * 50)
    
    # Загрузка модели и скейлера
    model, scaler = load_model_and_scaler()
    if model is None:
        print("❌ Не удалось загрузить модель для дообучения")
        return
    
    # Загрузка новых данных для дообучения
    try:
        new_data_file = 'train.csv'  # Файл с новыми данными
        df = pd.read_csv(new_data_file)
        print(f"✅ Новые данные загружены: {len(df)} записей")
    except FileNotFoundError:
        print(f"❌ Файл с новыми данными '{new_data_file}' не найден")
        print("💡 Создайте файл 'train.csv' с колонкой 'is_done'")
        return
    
    # Подготовка данных
    try:
        X, y, processed_df = prepare_features(df)
        print(f"✅ Признаки подготовлены: {X.shape}")
        
        # Обучение скейлера на новых данных (или обновление)
        if not hasattr(scaler, 'n_features_in_'):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            print("✅ Скейлер обучен на новых данных")
        else:
            # Частичное обновление скейлера
            scaler.partial_fit(X)
            print("✅ Скейлер обновлен на новых данных")
            
    except Exception as e:
        print(f"❌ Ошибка подготовки данных: {e}")
        return
    
    # Дообучение модели
    try:
        updated_model, history = fine_tune_model(model, X, y, scaler, epochs=15, batch_size=32)
        
        # Сохранение обновленной модели
        save_updated_model(updated_model, scaler)
        
        print(f"\n🎉 ДООБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("💡 Используйте 'price_optimizer_nn_model_updated.h5' для предсказаний")
        
    except Exception as e:
        print(f"❌ Ошибка дообучения: {e}")

if __name__ == "__main__":
    main()