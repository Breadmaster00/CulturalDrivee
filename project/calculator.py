import numpy as np
import pandas as pd
from datetime import datetime
from config import CalculatorConfig  # Импорт конфига
from price_optimizer import PriceOptimizer  # Импорт оптимизатора
import os

class NeuralPriceCalculator:
    def __init__(self):
        self.price_optimizer = PriceOptimizer()
        self.is_model_loaded = False
        self.load_model()  # Загружаем модель сразу при инициализации
    
    def load_model(self):
        """Загрузка модели (предполагаем, что файлы существуют)"""
        if os.path.exists(CalculatorConfig.MODEL_PATH) and os.path.exists(CalculatorConfig.SCALER_PATH):
            print("🔄 Загрузка обученной модели...")
            self.is_model_loaded = self.price_optimizer.load_model()
            if self.is_model_loaded:
                print("✅ Модель успешно загружена!")
            else:
                raise ValueError("❌ Не удалось загрузить модель")
        else:
            raise ValueError("📝 Модель или скейлер не найдены. Убедитесь, что файлы существуют.")
    
    def calculate_optimal_price(self, order_data):
        """Основная функция расчета оптимальной цены с нейронной сетью (адаптировано под новые стратегии)"""
        if not self.is_model_loaded:
            raise ValueError("Модель не загружена!")
        
        order_features = self.price_optimizer.data_processor.prepare_order_features({
            'distance_in_meters': order_data.distance_in_meters,
            'duration_in_seconds': order_data.duration_in_seconds,
            'pickup_in_meters': order_data.pickup_in_meters,
            'pickup_in_seconds': order_data.pickup_in_seconds,
            'price_start_local': order_data.price_start_local,
            'driver_rating': order_data.driver_rating,
            'carname': order_data.carname,
            'carmodel': order_data.carmodel,
            'platform': order_data.platform,
            'order_timestamp': order_data.order_timestamp or datetime.now().isoformat()
        })
        
        current_price = order_data.current_price or order_data.price_start_local
        
        price_range = self._get_price_range_from_data(current_price)
        
        # Рассчитываем три стратегии + текущую
        result = self._find_three_price_strategies(order_features, price_range, current_price)
        
        return result
    
    def _get_price_range_from_data(self, current_price):
        if self.price_optimizer.data_processor.price_stats:
            min_price = max(50, self.price_optimizer.data_processor.price_stats.get('min_price', 50))
            max_price = min(2000, self.price_optimizer.data_processor.price_stats.get('max_price', 1000))
            return (min_price, max_price)
        else:
            return (max(50, current_price * 0.5), min(1000, current_price * 2.0))
    
    def _find_three_price_strategies(self, order_features, price_range, current_price):
        min_price, max_price = price_range
        current_prob = self.price_optimizer.predict_probability(order_features, current_price)
        current_income = current_prob * current_price
        
        # Генерация тестовых цен
        test_prices = np.unique(np.logspace(np.log10(min_price), np.log10(max_price), num=50, dtype=int))
        test_prices = np.append(test_prices, current_price)
        test_prices = np.sort(np.unique(test_prices))
        
        # Собираем данные по всем ценам
        price_data = []
        for price in test_prices:
            prob = self.price_optimizer.predict_probability(order_features, price)
            income = prob * price
            price_data.append({
                'price': price,
                'probability': prob,
                'income': income
            })
        
        # Сортируем по доходу для нахождения оптимальной цены
        price_data_sorted_by_income = sorted(price_data, key=lambda x: x['income'], reverse=True)
        
        # 1. ТОЧНЫЙ вариант (максимальная вероятность)
        precise_data = max(price_data, key=lambda x: x['probability'])
        
        # 2. УМЕРЕННЫЙ вариант (баланс вероятности и дохода)
        # Ищем цены с вероятностью не менее 70% от максимальной и берем ту, у которой максимальный доход
        min_prob_for_moderate = precise_data['probability'] * 0.7
        moderate_candidates = [p for p in price_data if p['probability'] >= min_prob_for_moderate]
        if moderate_candidates:
            moderate_data = max(moderate_candidates, key=lambda x: x['income'])
        else:
            moderate_data = price_data_sorted_by_income[1] if len(price_data_sorted_by_income) > 1 else price_data_sorted_by_income[0]
        
        # 3. РИСКОВАННЫЙ вариант (высокий доход, но вероятность < 50%)
        # Ищем цены с вероятностью < 50% и максимальным доходом
        risky_candidates = [p for p in price_data if p['probability'] < 0.5]
        if risky_candidates:
            risky_data = max(risky_candidates, key=lambda x: x['income'])
        else:
            # Если нет цен с вероятностью < 50%, берем самую высокую цену с разумной вероятностью
            risky_candidates = [p for p in price_data if p['probability'] > 0.3]
            if risky_candidates:
                risky_data = max(risky_candidates, key=lambda x: x['price'])
            else:
                risky_data = price_data_sorted_by_income[0]
        
        return {
            'precise': {
                'price': round(precise_data['price']),
                'probability': float(round(precise_data['probability'], 3)),
                'income': float(round(precise_data['income'], 2)),
                'description': 'Наибольшая вероятность выполнения заказа'
            },
            'moderate': {
                'price': round(moderate_data['price']),
                'probability': float(round(moderate_data['probability'], 3)),
                'income': float(round(moderate_data['income'], 2)),
                'description': 'Оптимальный баланс цены и вероятности'
            },
            'risky': {
                'price': round(risky_data['price']),
                'probability': float(round(risky_data['probability'], 3)),
                'income': float(round(risky_data['income'], 2)),
                'description': 'Высокий доход при умеренном риске'
            },
            'current': {
                'price': round(current_price),
                'probability': float(round(current_prob, 3)),
                'income': float(round(current_income, 2)),
                'description': 'Ваша текущая цена'
            }
        }