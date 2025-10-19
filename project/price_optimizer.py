import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from config import CalculatorConfig  # Импорт конфига
from data_processor import DataProcessor  # Импорт процессора
from nn_model import NeuralNetworkModel  # Импорт модели
from keras.models import load_model

class PriceOptimizer:
    def __init__(self):
        self.model = None
        self.data_processor = DataProcessor()
        self.feature_importances = None  # Добавлено
    
    def fit(self, df):
        print("🔄 Подготовка данных для нейронной сети...")
        df_processed = self.data_processor.prepare_features(df)
        
        if df_processed is None:
            print("❌ Не удалось подготовить данные")
            return False, None
        
        initial_size = len(df_processed)
        df_processed = df_processed.dropna(subset=CalculatorConfig.FEATURE_COLUMNS + ['is_done'])
        final_size = len(df_processed)
        
        print(f"📉 Удалено строк с пропусками: {initial_size - final_size}")
        
        if final_size == 0:
            print("❌ Нет данных для обучения после очистки")
            return False, None
        
        X = df_processed[CalculatorConfig.FEATURE_COLUMNS]
        y = df_processed['is_done']
        
        print(f"🎯 Размерность данных для обучения: {X.shape}")
        print(f"⚖️ Баланс классов: {y.mean():.1%} положительных")
        
        # Применение SMOTE для баланса классов
        print("🔄 Применение SMOTE для баланса классов...")
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        print(f"   Новый баланс: {y.mean():.1%}")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Масштабирование признаков
        X_train_scaled = self.data_processor.scale_features(X_train, fit=True)
        X_val_scaled = self.data_processor.scale_features(X_val, fit=False)
        
        print("🤖 Обучение нейронной сети...")
        self.model = NeuralNetworkModel(input_dim=X_train_scaled.shape[1])
        self.model.build_model()
        
        # Обучение модели
        history = self.model.train(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Визуализация обучения
        self.model.plot_training_history()
        
        # Оценка модели
        y_pred_proba = self.model.predict_proba(X_val_scaled)
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        print(f"📊 ROC-AUC: {roc_auc:.4f}")
        
        y_pred = (np.array(y_pred_proba) > 0.5).astype(int)
        print("\n📈 Отчет классификации:")
        print(classification_report(y_val, y_pred))
        
        # Анализ важности признаков (через permutation importance)
        self._calculate_feature_importance(X_val_scaled, y_val, CalculatorConfig.FEATURE_COLUMNS)
        
        self.save_model()
        return True, {"roc_auc": float(roc_auc)}
    
    def _calculate_feature_importance(self, X_val, y_val, feature_names, n_repeats=5):
        """Вычисление важности признаков через permutation importance"""
        print("\n🔑 Вычисление важности признаков...")
        
        baseline_score = roc_auc_score(y_val, self.model.predict_proba(X_val))
        importances = np.zeros(len(feature_names))
        
        for i, feature_idx in enumerate(range(len(feature_names))):
            scores = []
            for _ in range(n_repeats):
                X_permuted = X_val.copy()
                np.random.shuffle(X_permuted[:, feature_idx])
                permuted_score = roc_auc_score(y_val, self.model.predict_proba(X_permuted))
                scores.append(baseline_score - permuted_score)
            
            importances[i] = np.mean(scores)
        
        # Нормализация важностей
        if importances.sum() > 0:
            importances = importances / importances.sum()
        
        self.feature_importances = dict(zip(feature_names, importances))
        self._print_feature_importances()
    
    def _print_feature_importances(self):
        print("\n🔑 Важность признаков (Permutation Importance):")
        sorted_importances = sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for feature, importance in sorted_importances[:10]:
            print(f"   {feature}: {importance:.4f}")
    
    def save_model(self):
        try:
            # Сохраняем нейронную сеть
            if self.model and self.model.model:
                self.model.model.save(CalculatorConfig.MODEL_PATH)
            
            # Сохраняем остальные компоненты
            processor_data = {
                'label_encoders': self.data_processor.label_encoders,
                'price_stats': self.data_processor.price_stats,
                'feature_importances': self.feature_importances,
                'timestamp': datetime.now()
            }
            
            with open(CalculatorConfig.SCALER_PATH, 'wb') as f:
                pickle.dump({
                    'scaler': self.data_processor.scaler,
                    'is_scaler_fitted': self.data_processor.is_scaler_fitted,
                    'processor_data': processor_data
                }, f)
            
            print(f"💾 Модель сохранена в: {CalculatorConfig.MODEL_PATH}")
            print(f"💾 Scaler сохранен в: {CalculatorConfig.SCALER_PATH}")
        except Exception as e:
            print(f"❌ Ошибка сохранения модели: {e}")
    
    def load_model(self):
        try:
            if not os.path.exists(CalculatorConfig.MODEL_PATH) or not os.path.exists(CalculatorConfig.SCALER_PATH):
                print(f"❌ Файлы модели не найдены")
                return False
            
            # Загружаем нейронную сеть
            self.model = NeuralNetworkModel(input_dim=len(CalculatorConfig.FEATURE_COLUMNS))
            self.model.model = load_model(CalculatorConfig.MODEL_PATH)
            
            # Загружаем scaler и данные процессора
            with open(CalculatorConfig.SCALER_PATH, 'rb') as f:
                saved_data = pickle.load(f)
            
            self.data_processor.scaler = saved_data['scaler']
            self.data_processor.is_scaler_fitted = saved_data['is_scaler_fitted']
            self.data_processor.label_encoders = saved_data['processor_data']['label_encoders']
            self.data_processor.price_stats = saved_data['processor_data'].get('price_stats', {})
            self.feature_importances = saved_data['processor_data'].get('feature_importances', {})
            
            print(f"📂 Модель загружена из: {CalculatorConfig.MODEL_PATH}")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False
    
    def predict_probability(self, order_features, bid_price):
        if self.model is None:
            raise ValueError("Модель не обучена!")
        
        feature_dict = order_features.copy()
        feature_dict['price_bid_local'] = float(bid_price)
        feature_dict['price_ratio'] = float(bid_price / feature_dict['price_start_local'] if feature_dict['price_start_local'] > 0 else 1)
        
        feature_df = pd.DataFrame([feature_dict])[CalculatorConfig.FEATURE_COLUMNS]
        
        # Масштабирование признаков
        feature_scaled = self.data_processor.scale_features(feature_df)
        
        probability = self.model.predict_proba(feature_scaled)[0]
        return max(0.0, min(1.0, probability))