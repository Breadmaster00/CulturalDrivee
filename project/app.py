import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Optional, Any, List
import pickle
import os
from datetime import datetime
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import warnings
import json
warnings.filterwarnings('ignore')

# =====================================================================
#  МОДЕЛИ ДАННЫХ
# =====================================================================
class OrderData(BaseModel):
    distance_in_meters: float
    duration_in_seconds: float
    pickup_in_meters: float = 500
    pickup_in_seconds: float = 300
    price_start_local: float = 100
    driver_rating: float = 4.5
    carname: str = "Toyota"
    carmodel: str = "Camry"
    platform: str = "android"
    current_price: float = 120
    order_timestamp: str = ""

class OptimizationResponse(BaseModel):
    success: bool
    strategies: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    optimal_price: Optional[Dict[str, Any]] = None

# =====================================================================
#  ПРОСТОЕ ХРАНИЛИЩЕ ЗАКАЗОВ (в памяти)
# =====================================================================
ORDERS: List[Dict[str, Any]] = []

# =====================================================================
#  ML КАЛЬКУЛЯТОР С ОПТИМИЗАЦИЕЙ ДОХОДНОСТИ
# =====================================================================
class NeuralPriceCalculator:
    def __init__(self):
        self.is_model_loaded = False
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Загрузка ML модели (заглушка для демонстрации)"""
        try:
            # В реальном проекте здесь загружалась бы обученная модель
            self.is_model_loaded = True
            print("✅ ML модель инициализирована")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки модели: {e}")
            self.is_model_loaded = False
    
    def calculate_optimal_price(self, order_data: OrderData) -> Dict[str, Any]:
        """
        Расчет оптимальных ценовых стратегий с оптимизацией ожидаемого дохода
        Ожидаемый доход = Цена × Вероятность принятия
        """
        try:
            # Базовый расчет цены на основе дистанции и времени
            base_price = self._calculate_base_price(order_data)
            
            # Генерируем стратегии с разными ценами
            strategies = {
                "precise": {
                    "price": max(80, int(base_price * 0.85)),
                    "probability": 0.75
                },
                "current": {
                    "price": int(base_price),
                    "probability": 0.60
                },
                "moderate": {
                    "price": int(base_price * 1.25),
                    "probability": 0.45
                },
                "risky": {
                    "price": int(base_price * 1.6),
                    "probability": 0.25
                }
            }
            
            # Корректируем вероятности на основе дополнительных факторов
            strategies = self._adjust_probabilities(strategies, order_data)
            
            # Рассчитываем ожидаемый доход для каждой стратегии
            for strategy_name, strategy_data in strategies.items():
                expected_income = strategy_data["price"] * strategy_data["probability"]
                strategies[strategy_name]["income"] = round(expected_income)
            
            # Находим оптимальную стратегию (максимизирующую ожидаемый доход)
            optimal_strategy = max(strategies.items(), key=lambda x: x[1]["income"])
            
            return {
                "success": True,
                "strategies": strategies,
                "optimal_price": {
                    "strategy": optimal_strategy[0],
                    **optimal_strategy[1]
                },
                "analysis": {
                    "base_price": base_price,
                    "distance_km": round(order_data.distance_in_meters / 1000, 1),
                    "duration_min": round(order_data.duration_in_seconds / 60, 1),
                    "optimization_method": "expected_income_maximization"
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Ошибка расчета: {str(e)}"
            }
    
    def _calculate_base_price(self, order_data: OrderData) -> float:
        """Расчет базовой цены на основе параметров заказа"""
        # Базовая формула: стоимость за км + стоимость за минуту + базовый тариф
        distance_km = order_data.distance_in_meters / 1000
        duration_min = order_data.duration_in_seconds / 60
        
        base_price = (distance_km * 20) + (duration_min * 3) + 50
        
        # Корректировки на основе дополнительных факторов
        adjustments = 0
        
        # Время суток (вечером/ночью дороже)
        try:
            order_time = datetime.fromisoformat(order_data.order_timestamp.replace('Z', '+00:00'))
            if 18 <= order_time.hour <= 23 or 0 <= order_time.hour <= 6:
                adjustments += 30
        except:
            pass
        
        # Дистанция (длинные поездки дороже)
        if distance_km > 15:
            adjustments += 20
        elif distance_km > 5:
            adjustments += 10
        
        # Срочность (близкий pickup)
        if order_data.pickup_in_seconds < 600:  # Меньше 10 минут
            adjustments += 15
        
        final_price = base_price + adjustments
        return max(100, round(final_price))
    
    def _adjust_probabilities(self, strategies: Dict, order_data: OrderData) -> Dict:
        """Корректировка вероятностей принятия на основе контекста"""
        adjusted_strategies = strategies.copy()
        
        # Факторы, влияющие на вероятность принятия
        probability_adjustments = 0.0
        
        # Рейтинг водителя
        if order_data.driver_rating >= 4.8:
            probability_adjustments += 0.10
        elif order_data.driver_rating >= 4.5:
            probability_adjustments += 0.05
        
        # Платформа (iOS пользователи чаще принимают higher prices)
        if order_data.platform.lower() == "ios":
            probability_adjustments += 0.03
        
        # Время суток (вечером выше готовность платить)
        try:
            order_time = datetime.fromisoformat(order_data.order_timestamp.replace('Z', '+00:00'))
            if 18 <= order_time.hour <= 23:
                probability_adjustments += 0.05
        except:
            pass
        
        # Применяем корректировки к вероятностям
        for strategy_name in adjusted_strategies:
            current_prob = adjusted_strategies[strategy_name]["probability"]
            # Более высокие цены получают меньший бонус к вероятности
            price_multiplier = 1.0 if strategy_name == "precise" else 0.7
            if strategy_name == "risky":
                price_multiplier = 0.3
            
            new_prob = current_prob + (probability_adjustments * price_multiplier)
            # Ограничиваем вероятность разумными пределами
            adjusted_strategies[strategy_name]["probability"] = max(0.1, min(0.9, new_prob))
        
        return adjusted_strategies

# =====================================================================
#  ИНИЦИАЛИЗАЦИЯ ПРИЛОЖЕНИЯ
# =====================================================================
app = FastAPI(
    title="Taxi Price Optimizer API (Neural Network)",
    description="API для оптимизации цен такси с использованием нейронной сети и максимизации ожидаемого дохода",
    version="2.3.0"
)

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

# Настраиваем шаблоны Jinja2
templates = Jinja2Templates(directory="templates")

# Глобальный экземпляр ML-калькулятора
calculator = NeuralPriceCalculator()

# =====================================================================
#  WEB SOCKET ДЛЯ РЕАЛЬНОГО ВРЕМЕНИ
# =====================================================================
from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                self.active_connections.remove(connection)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Обработка сообщений от клиента
            await websocket.send_text(json.dumps({"type": "pong", "message": "Connected"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# =====================================================================
#  API ENDPOINTS
# =====================================================================
@app.get("/")
async def root():
    """Главная страница"""
    model_status = "загружена" if calculator.is_model_loaded else "не загружена"
    return {
        "message": "Taxi Price Optimizer API (Neural Network)",
        "version": "2.3.0",
        "model_status": model_status,
        "optimization_method": "Максимизация ожидаемого дохода (Цена × Вероятность)",
        "endpoints": {
            "POST /optimize": "расчет оптимальных стратегий с оптимизацией дохода",
            "POST /create_order": "создание нового заказа",
            "GET /get_orders": "получить все заказы",
            "WebSocket /ws": "реальное время обновлений"
        }
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья API"""
    return {
        "status": "healthy",
        "model_loaded": calculator.is_model_loaded,
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(manager.active_connections)
    }

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_price(order_data: OrderData):
    """
    Расчет оптимальных ценовых стратегий с максимизацией ожидаемого дохода
    Формула: Ожидаемый доход = Цена × Вероятность принятия
    """
    try:
        result = calculator.calculate_optimal_price(order_data)
        
        # Отправляем обновление через WebSocket
        await manager.broadcast(json.dumps({
            "type": "price_optimization",
            "strategies": result.get("strategies"),
            "optimal_price": result.get("optimal_price"),
            "timestamp": datetime.now().isoformat()
        }))
        
        return OptimizationResponse(**result)
    except Exception as e:
        return OptimizationResponse(success=False, message=str(e))

@app.post("/create_order")
async def create_order(order_data: Dict[str, Any]):
    try:
        # Проверка наличия и корректности цены
        passenger_price = order_data.get("passengerPrice")
        if not passenger_price or passenger_price <= 0:
            return {"success": False, "message": "Цена заказа не указана или некорректна"}

        # Ограничение максимальной цены
        if passenger_price > 10000:
            return {"success": False, "message": "Максимальная цена — 10000 ₽"}

        # Создание заказа
        clean_order = {
            "id": len(ORDERS) + 1,
            "fromAddress": order_data.get("fromAddress"),
            "toAddress": order_data.get("toAddress"),
            "passengers": order_data.get("passengers"),
            "pets": order_data.get("pets"),
            "time": order_data.get("time"),
            "baggage": order_data.get("baggage"),
            "passengerPrice": passenger_price,
            "timestamp": datetime.now().isoformat()
        }

        ORDERS.append(clean_order)

        # Уведомляем водителей
        await manager.broadcast(json.dumps({
            "type": "new_order",
            "order": clean_order,
            "message": "Появился новый заказ",
            "timestamp": datetime.now().isoformat()
        }))

        print(f"✅ Новый заказ сохранён: {clean_order}")
        return {"success": True, "message": "Заказ успешно создан", "order_id": clean_order["id"]}

    except Exception as e:
        print("❌ Ошибка при создании заказа:", e)
        return {"success": False, "message": str(e)}
    
@app.get("/get_orders")
async def get_orders():
    """Возвращает все сохранённые заказы"""
    try:
        return {
            "success": True, 
            "orders": ORDERS,
            "count": len(ORDERS),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"success": False, "message": str(e), "orders": []}

@app.delete("/orders/{order_id}")
async def delete_order(order_id: int):
    """Удаление заказа (для тестирования)"""
    try:
        global ORDERS
        ORDERS = [order for order in ORDERS if order["id"] != order_id]
        return {"success": True, "message": f"Заказ {order_id} удален"}
    except Exception as e:
        return {"success": False, "message": str(e)}

# =====================================================================
#  ВЕБ-СТРАНИЦЫ
# =====================================================================
@app.get("/passenger")
async def passenger_interface(request: Request):
    """Веб-интерфейс для пассажира"""
    return templates.TemplateResponse("passenger.html", {"request": request})
    
@app.get("/order_processing.html")
async def order_processing_interface(request: Request):
    """Страница обработки выбранного заказа"""
    return templates.TemplateResponse("order_processing.html", {"request": request})
@app.get("/driver")
async def driver_interface(request: Request):
    """Веб-интерфейс для водителя (старый)"""
    return templates.TemplateResponse("driver.html", {"request": request})
@app.get("/order_processing")
async def order_processing_interface(request: Request):
    """Страница обработки выбранного заказа"""
    return templates.TemplateResponse("order_processing.html", {"request": request})

@app.get("/driver_orders_list")
async def driver_orders_list(request: Request):
    """Список заказов для водителя"""
    return templates.TemplateResponse("driver_orders_list.html", {"request": request})

@app.get("/driver_order_details")
async def driver_order_details(request: Request):
    """Детали заказа для водителя"""
    return templates.TemplateResponse("driver_order_details.html", {"request": request})

@app.get("/orders")
async def orders_interface(request: Request):
    """Страница истории заказов пассажира"""
    return templates.TemplateResponse("orders.html", {"request": request})

@app.get("/optimization_demo")
async def optimization_demo(request: Request):
    """Демонстрация оптимизации доходности"""
    return templates.TemplateResponse("optimization_demo.html", {"request": request})

# =====================================================================
#  ДЕМО-ENDPOINTS ДЛЯ ТЕСТИРОВАНИЯ
# =====================================================================
@app.post("/demo/optimize")
async def demo_optimize_price(distance_km: float = 5.0, duration_min: float = 15.0):
    """
    Демо-эндпоинт для тестирования оптимизации цен
    """
    demo_data = OrderData(
        distance_in_meters=distance_km * 1000,
        duration_in_seconds=duration_min * 60,
        order_timestamp=datetime.now().isoformat()
    )
    
    result = calculator.calculate_optimal_price(demo_data)
    
    return {
        "demo_parameters": {
            "distance_km": distance_km,
            "duration_min": duration_min,
            "base_price": result.get("analysis", {}).get("base_price", 0)
        },
        **result
    }



# =====================================================================
#  ЗАПУСК СЕРВЕРА
# =====================================================================
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Запуск Taxi Price Optimizer API (Neural Network)...")
    print("📚 Документация: http://localhost:8000/docs")
    print("🧮 Оптимизация: Максимизация ожидаемого дохода (Цена × Вероятность)")
    print("🌐 Веб-интерфейсы:")
    print("   👤 Пассажир: http://localhost:8000/passenger")
    print("   🚗 Водитель (список): http://localhost:8000/driver_orders_list")
    print("   📋 История заказов: http://localhost:8000/orders")
    print("   🔧 Демо оптимизации: http://localhost:8000/optimization_demo")
    print("🛑 Для остановки: Ctrl+C")

    uvicorn.run(app, host="0.0.0.0", port=8000)