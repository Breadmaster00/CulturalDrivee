import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import CalculatorConfig  # Импорт конфига

class DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.price_stats = {}
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False
        os.makedirs(CalculatorConfig.VISUALIZATION_DIR, exist_ok=True)
    
    def prepare_features(self, df):
        df = df.copy()
        
        print("🔍 Анализ данных из CSV...")
        print(f"Исходный размер данных: {df.shape}")
        
        missing_columns = [col for col in CalculatorConfig.REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            print(f"❌ Отсутствуют колонки: {missing_columns}")
            return None
        
        try:
            df['order_timestamp'] = pd.to_datetime(df['order_timestamp'])
            df['hour'] = df['order_timestamp'].dt.hour
            df['day_of_week'] = df['order_timestamp'].dt.dayofweek
            df['month'] = df['order_timestamp'].dt.month
        except Exception as e:
            print(f"❌ Ошибка преобразования дат: {e}")
            return None
        
        # Дополнительный feature engineering
        df['price_ratio'] = df['price_bid_local'] / df['price_start_local'].clip(lower=1)
        df['distance_per_minute'] = df['distance_in_meters'] / (df['duration_in_seconds'] / 60).clip(lower=1)
        df['pickup_time_ratio'] = df['pickup_in_seconds'] / df['duration_in_seconds'].clip(lower=1)
        
        for col in CalculatorConfig.CATEGORICAL_COLUMNS:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = df[col].fillna('unknown')
                self.label_encoders[col].fit(df[col])
            
            df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        df['is_done'] = df['is_done'].map({'done': 1, 'cancel': 0})
        
        self.price_stats = {
            'min_price': float(df['price_bid_local'].min()),
            'max_price': float(df['price_bid_local'].max()),
            'mean_price': float(df['price_bid_local'].mean()),
            'std_price': float(df['price_bid_local'].std()),
            'q25': float(df['price_bid_local'].quantile(0.25)),
            'q75': float(df['price_bid_local'].quantile(0.75)),
            'mean_increase_pct': float(((df['price_bid_local'] / df['price_start_local'] - 1) * 100).mean())
        }
        
        print(f"🎯 Выполнено заказов: {df['is_done'].sum()} из {len(df)} ({df['is_done'].mean():.1%})")
        print(f"💰 Статистика цен: {self.price_stats['min_price']:.0f}-{self.price_stats['max_price']:.0f} руб")
        print(f"📈 Среднее увеличение цены: {self.price_stats['mean_increase_pct']:.1f}%")
        
        self._perform_data_analysis(df)
        
        return df
    
    def _perform_data_analysis(self, df):
        print("\n📊 Глубокий анализ данных...")
        
        # Корреляция
        corr = df[CalculatorConfig.FEATURE_COLUMNS + ['is_done']].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Корреляционная матрица')
        plt.savefig(os.path.join(CalculatorConfig.VISUALIZATION_DIR, 'correlation_heatmap.png'))
        plt.close()
        
        # Зависимость от времени суток
        hourly_acceptance = df.groupby('hour')['is_done'].mean()
        plt.figure(figsize=(10, 6))
        hourly_acceptance.plot(kind='bar')
        plt.title('Вероятность принятия по часам')
        plt.ylabel('Вероятность')
        plt.savefig(os.path.join(CalculatorConfig.VISUALIZATION_DIR, 'hourly_acceptance.png'))
        plt.close()
        
        # Зависимость от дня недели
        weekly_acceptance = df.groupby('day_of_week')['is_done'].mean()
        plt.figure(figsize=(10, 6))
        weekly_acceptance.plot(kind='bar')
        plt.title('Вероятность принятия по дням недели')
        plt.ylabel('Вероятность')
        plt.savefig(os.path.join(CalculatorConfig.VISUALIZATION_DIR, 'weekly_acceptance.png'))
        plt.close()
        
        # Распределение увеличения цены
        df['price_increase_pct'] = (df['price_bid_local'] / df['price_start_local'] - 1) * 100
        accepted_increase = df[df['is_done'] == 1]['price_increase_pct']
        plt.figure(figsize=(10, 6))
        sns.histplot(accepted_increase, kde=True)
        plt.title('Распределение увеличения цены для принятых заказов')
        plt.xlabel('Увеличение цены (%)')
        plt.savefig(os.path.join(CalculatorConfig.VISUALIZATION_DIR, 'price_increase_distribution.png'))
        plt.close()
        
        print(f"📈 Визуализации сохранены в: {CalculatorConfig.VISUALIZATION_DIR}")
        print(f"🔑 Ключевые insights:")
        print(f"   - Корреляция цены и принятия: {corr['is_done']['price_bid_local']:.2f}")
        print(f"   - Максимальная вероятность в час: {hourly_acceptance.idxmax()} ({hourly_acceptance.max():.1%})")
        print(f"   - Среднее увеличение для принятых: {accepted_increase.mean():.1f}%")
    
    def prepare_order_features(self, order_data):
        features = {
            'distance_in_meters': float(order_data.get('distance_in_meters', 0)),
            'duration_in_seconds': float(order_data.get('duration_in_seconds', 0)),
            'pickup_in_meters': float(order_data.get('pickup_in_meters', 0)),
            'pickup_in_seconds': float(order_data.get('pickup_in_seconds', 0)),
            'price_start_local': float(order_data.get('price_start_local', 0)),
            'price_bid_local': float(order_data.get('price_bid_local', order_data.get('price_start_local', 0))),
            'driver_rating': float(order_data.get('driver_rating', 5.0)),
        }
        
        order_time = pd.to_datetime(order_data.get('order_timestamp', pd.Timestamp.now()))
        features['hour'] = int(order_time.hour)
        features['day_of_week'] = int(order_time.dayofweek)
        features['month'] = int(order_time.month)
        
        # Дополнительные фичи
        features['price_ratio'] = float(features['price_bid_local'] / features['price_start_local'] if features['price_start_local'] > 0 else 1)
        features['distance_per_minute'] = float(features['distance_in_meters'] / (features['duration_in_seconds'] / 60) if features['duration_in_seconds'] > 0 else 0)
        features['pickup_time_ratio'] = float(features['pickup_in_seconds'] / features['duration_in_seconds'] if features['duration_in_seconds'] > 0 else 0)
        
        for col in CalculatorConfig.CATEGORICAL_COLUMNS:
            value = order_data.get(col, 'unknown')
            encoder_key = f'{col}_encoded'
            if col in self.label_encoders:
                try:
                    classes = list(self.label_encoders[col].classes_)
                    if value in classes:
                        features[encoder_key] = int(classes.index(value))
                    else:
                        features[encoder_key] = 0
                except:
                    features[encoder_key] = 0
            else:
                features[encoder_key] = 0
        
        return features
    
    def scale_features(self, X, fit=False):
        """Масштабирование признаков для нейронной сети"""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.is_scaler_fitted = True
        else:
            if not self.is_scaler_fitted:
                raise ValueError("Scaler не был обучен! Сначала вызовите scale_features с fit=True")
            X_scaled = self.scaler.transform(X)
        return X_scaled