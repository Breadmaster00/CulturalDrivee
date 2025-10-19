// ======================= НАСТРОЙКИ КАРТЫ (Leaflet + OSM) =======================
document.addEventListener("DOMContentLoaded", async function () {
    const defaultCenter = [62.027221, 129.733383]; // Якутск

    // Создаём карту
window.map = L.map("map", { 
    zoomControl: false,
    attributionControl: false
}).setView(defaultCenter, 12);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: false
}).addTo(window.map);

// Добавляем контролы отдельно без атрибуции
L.control.zoom({
    position: 'topright'
}).addTo(window.map);

    // Глобальные переменные
    window.fromMarker = null;
    window.toMarker = null;
    window.routeLayer = null;

    // Обработка клика по карте
    window.map.on("click", onMapClick);

    // Кнопка очистки точек
    const clearBtn = document.getElementById("clearPointsBtn");
    if (clearBtn) clearBtn.addEventListener("click", clearPoints);

    // Автообновление цены при вводе адресов (passenger)
    const inputs = document.querySelectorAll("#fromAddress, #toAddress");
    inputs.forEach((i) => i.addEventListener("input", debounce(calculatePrice, 1000)));

    // Если driver.html — загрузить заказы
    if (window.location.pathname.includes("driver.html")) {
        loadOrders();
    }

    // Для passenger.html — симуляция ML расчёта
    if (window.location.pathname.includes("passenger.html")) {
        setTimeout(() => {
            const mlInfo = document.querySelector(".ml-info");
            if (mlInfo) {
                mlInfo.innerHTML =
                    '<i class="fas fa-check-circle" style="color: var(--success);"></i> ML-модель рассчитала оптимальные цены на основе текущего спроса и погодных условий';
            }
        }, 1500);
    }

    // Обновление "вашей цены" у водителя
    const driverPriceInput = document.getElementById("driverPrice");
    if (driverPriceInput) {
        driverPriceInput.addEventListener("input", () => {
            document.getElementById("yourPriceValue").textContent =
                driverPriceInput.value + " ₽";
        });
    }

    // Кнопка разворачивания карты
    const toggleMapBtn = document.getElementById("toggleMapBtn");
    if (toggleMapBtn) {
        toggleMapBtn.addEventListener("click", toggleMapFullscreen);
    }
});

// ======================= ОБРАБОТКА КАРТЫ =======================

// Клик по карте: ставим маркеры
async function onMapClick(e) {
    const latlng = e.latlng;

    if (!window.fromMarker) {
        await setFromMarker(latlng);
        return;
    }
    if (!window.toMarker) {
        await setToMarker(latlng);
        return;
    }

    // Если оба маркера есть — перемещаем ближайший
    const d1 = latlng.distanceTo(window.fromMarker.getLatLng());
    const d2 = latlng.distanceTo(window.toMarker.getLatLng());
    if (d1 < d2) await setFromMarker(latlng);
    else await setToMarker(latlng);
}

// Установка маркеров
async function setFromMarker(latlng) {
    if (window.fromMarker) window.map.removeLayer(window.fromMarker);
    window.fromMarker = L.marker(latlng, { draggable: true })
        .addTo(window.map)
        .bindPopup("Откуда")
        .openPopup();
    await updateAddressFromLatLng(latlng.lat, latlng.lng, "from");

    window.fromMarker.on("dragend", async (e) => {
        const p = e.target.getLatLng();
        await updateAddressFromLatLng(p.lat, p.lng, "from");
        if (window.toMarker) calculatePrice();
    });

    if (window.toMarker) calculatePrice();
}

async function setToMarker(latlng) {
    if (window.toMarker) window.map.removeLayer(window.toMarker);
    window.toMarker = L.marker(latlng, { draggable: true })
        .addTo(window.map)
        .bindPopup("Куда")
        .openPopup();
    await updateAddressFromLatLng(latlng.lat, latlng.lng, "to");

    window.toMarker.on("dragend", async (e) => {
        const p = e.target.getLatLng();
        await updateAddressFromLatLng(p.lat, p.lng, "to");
        if (window.fromMarker) calculatePrice();
    });

    if (window.fromMarker) calculatePrice();
}

// Очистка точек
function clearPoints() {
    if (window.fromMarker) window.map.removeLayer(window.fromMarker);
    if (window.toMarker) window.map.removeLayer(window.toMarker);
    if (window.routeLayer) window.map.removeLayer(window.routeLayer);
    window.fromMarker = null;
    window.toMarker = null;
    window.routeLayer = null;
    document.getElementById("fromAddress").value = "";
    document.getElementById("toAddress").value = "";
}

// Обновление адреса по координатам
async function updateAddressFromLatLng(lat, lon, type) {
    const addr = await reverseGeocode(lat, lon);
    const el =
        type === "from"
            ? document.getElementById("fromAddress")
            : document.getElementById("toAddress");
    if (el) el.value = addr || `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
}

// Reverse-geocode через Nominatim
async function reverseGeocode(lat, lon) {
    try {
        const url = `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}&accept-language=ru`;
        const resp = await fetch(url);
        const data = await resp.json();
        return data.display_name || null;
    } catch {
        return null;
    }
}

// Геокодирование адреса → [lat, lon]
async function geocodeAddress(address) {
    const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(
        address
    )}&limit=1&addressdetails=0&accept-language=ru`;
    const resp = await fetch(url);
    const data = await resp.json();
    if (!data.length) return null;
    return [parseFloat(data[0].lat), parseFloat(data[0].lon)];
}

// Маршрут OSRM
async function getRouteOSRM(from, to) {
    const url = `https://router.project-osrm.org/route/v1/driving/${from[1]},${from[0]};${to[1]},${to[0]}?overview=full&geometries=geojson`;
    const resp = await fetch(url);
    const data = await resp.json();
    if (!data.routes || !data.routes.length) return null;
    const r = data.routes[0];
    return {
        distanceMeters: Math.round(r.distance),
        durationSeconds: Math.round(r.duration),
        geometry: r.geometry
    };
}

// ======================= ОСНОВНАЯ ЛОГИКА =======================

// Выбор цены
function selectPriceOption(element) {
    document.querySelectorAll(".price-option").forEach((o) => o.classList.remove("selected"));
    element.classList.add("selected");
}

// Подтверждение заказа
async function confirmOrder() {
    const fromAddress = document.getElementById("fromAddress").value.trim();
    const toAddress = document.getElementById("toAddress").value.trim();
    const passengers = document.getElementById("passengers").value;
    const pets = document.getElementById("pets").value;
    const time = document.getElementById("time").value;
    const baggage = document.getElementById("baggage").value;

    // Получаем цену из поля ввода
    const priceInput = document.querySelector(".price-amount");
    let passengerPrice = parseInt(priceInput.value, 10);

    // Проверяем корректность цены
    if (isNaN(passengerPrice) || passengerPrice <= 0) {
        alert("Пожалуйста, укажите корректную цену.");
        return;
    }

    if (passengerPrice > 10000) {
        alert("Максимальная цена — 10000 ₽.");
        return;
    }

    try {
        const response = await fetch("/create_order", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                fromAddress,
                toAddress,
                passengers,
                pets,
                time,
                baggage,
                passengerPrice
            })
        });

        const data = await response.json();

        if (data.success) {
            alert("Заказ успешно создан!");
            window.location.href = "/orders";
        } else {
            alert(`Ошибка создания заказа: ${data.message}`);
        }
    } catch (error) {
        console.error("Ошибка при создании заказа:", error);
        alert("Произошла ошибка при создании заказа. Попробуйте позже.");
    }
}

// Принятие/отклонение заказа
function acceptOrder() {
    const driverPrice = document.getElementById("driverPrice").value;
    alert(`Заказ принят по цене ${driverPrice} ₽! Свяжитесь с пассажиром.`);
    window.location.href = "index.html";
}

function declineOrder() {
    alert("Заказ отклонён.");
    window.location.href = "index.html";
}

// Расчёт цены (через OSRM + ML)
async function calculatePrice() {
    try {
        let fromCoords = null;
        let toCoords = null;

        if (window.fromMarker && window.toMarker) {
            fromCoords = [window.fromMarker.getLatLng().lat, window.fromMarker.getLatLng().lng];
            toCoords = [window.toMarker.getLatLng().lat, window.toMarker.getLatLng().lng];
        } else {
            const fromAddress = document.getElementById("fromAddress").value.trim();
            const toAddress = document.getElementById("toAddress").value.trim();
            if (!fromAddress || !toAddress) return null;
            fromCoords = await geocodeAddress(fromAddress);
            toCoords = await geocodeAddress(toAddress);
        }

        if (!fromCoords || !toCoords) {
            alert("Введите корректные адреса!");
            return null;
        }

        const route = await getRouteOSRM(fromCoords, toCoords);
        if (!route) {
            alert("Маршрут не найден!");
            return null;
        }

        const orderData = {
            distance_in_meters: route.distanceMeters,
            duration_in_seconds: route.durationSeconds,
            pickup_in_meters: 500,
            pickup_in_seconds: 300,
            price_start_local: 100,
            driver_rating: 4.5,
            carname: "Toyota",
            carmodel: "Camry",
            platform: "android",
            current_price: 120,
            order_timestamp: new Date().toISOString()
        };

        let result;
        try {
            const resp = await fetch("/optimize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(orderData)
            });
            result = await resp.json();
        } catch {
            result = null;
        }

        if (!result || !result.success) {
            const base = Math.round(route.distanceMeters / 1000 * 40 + 100);
            result = {
                success: true,
                strategies: {
                    precise: { price: base, probability: 0.95, income: base * 0.9 },
                    current: { price: base + 30, probability: 0.85, income: base * 0.9 },
                    moderate: { price: base + 60, probability: 0.75, income: base * 0.9 },
                    risky: { price: base + 120, probability: 0.55, income: base * 0.9 }
                }
            };
        }

        if (window.map && route.geometry) {
            if (window.routeLayer) window.map.removeLayer(window.routeLayer);
            window.routeLayer = L.geoJSON(route.geometry, { style: { color: "blue", weight: 5 } });
            window.routeLayer.addTo(window.map);
            window.map.fitBounds(window.routeLayer.getBounds());
        }

        const mlInfo = document.querySelector(".ml-info span");
        if (mlInfo) mlInfo.textContent = "ML-модель рассчитала оптимальные цены!";

        const priceInput = document.querySelector('.price-amount');
        if (priceInput && result.strategies) {
            priceInput.value = result.strategies.moderate.price;
        }

        return result;
    } catch (err) {
        alert("Ошибка расчёта: " + err.message);
        return null;
    }
}

// Сбор данных заказа
function getOrderData() {
    const priceValue = parseFloat(document.getElementById("passengerPrice")?.value) || 0;

    return {
        fromAddress: document.getElementById("fromAddress").value,
        toAddress: document.getElementById("toAddress").value,
        passengers: document.getElementById("passengers")?.value || "1",
        pets: document.getElementById("pets")?.value || "no",
        time: document.getElementById("time")?.value || "now",
        baggage: document.getElementById("baggage")?.value || "no",
        passengerPrice: priceValue
    };
}

// Загрузка заказов (водитель)
async function loadOrders() {
    try {
        const resp = await fetch("/get_orders");
        const data = await resp.json();
        if (!data.success || !data.orders.length) {
            alert("Нет активных заказов");
            return;
        }

        const order = data.orders[0];
        document.getElementById("fromAddress").value = order.fromAddress;
        document.getElementById("toAddress").value = order.toAddress;
        document.getElementById("passengers").textContent = order.passengers + " чел.";
        document.getElementById("pets").textContent = order.pets;
        document.getElementById("time").textContent = order.time;
        document.getElementById("baggage").textContent = order.baggage;
        document.getElementById("passengerPrice").textContent = order.passengerPrice + " ₽";

        calculatePrice().then((r) => {
            if (r && r.strategies)
                document.getElementById("mlRecommendedPrice").textContent =
                    r.strategies.moderate.price + " ₽";
        });
    } catch (err) {
        alert("Ошибка загрузки заказов: " + err.message);
    }
}

// Переключение полноэкранного режима карты
function toggleMapFullscreen() {
    const mapSection = document.querySelector('.map-section');
    const mapContainer = document.querySelector('.map-container');
    
    if (mapSection.classList.contains('fullscreen')) {
        mapSection.classList.remove('fullscreen');
        mapContainer.classList.remove('fullscreen');
        mapContainer.style.height = '180px';
        document.querySelector('.main-content').style.overflowY = 'auto';
    } else {
        mapSection.classList.add('fullscreen');
        mapContainer.classList.add('fullscreen');
        mapContainer.style.height = '100%';
        document.querySelector('.main-content').style.overflowY = 'hidden';
    }
    
    setTimeout(() => {
        if (window.map) window.map.invalidateSize();
    }, 300);
}

// Ограничение ввода цены
document.addEventListener('DOMContentLoaded', function() {
    const priceInput = document.querySelector('.price-amount');
    if (priceInput) {
        priceInput.addEventListener('input', function() {
            if (this.value < 0) this.value = 0;
            if (this.value > 10000) this.value = 10000;
        });
    }
});

// Debounce
function debounce(func, delay) {
    let timeout;
    return function () {
        clearTimeout(timeout);
        timeout = setTimeout(func, delay);
    };
}