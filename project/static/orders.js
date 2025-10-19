let allOrders = [];

document.addEventListener("DOMContentLoaded", loadOrders);

async function loadOrders() {
    const container = document.getElementById("ordersList");
    container.innerHTML = `
        <div class="loading-state">
            <i class="fas fa-spinner fa-spin"></i>
            <p>Загрузка заказов...</p>
        </div>
    `;

    try {
        const response = await fetch("/get_orders");
        if (!response.ok) {
            throw new Error(`Ошибка загрузки данных: ${response.status}`);
        }

        const data = await response.json();

        if (!data.success || !data.orders.length) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-clipboard-list"></i>
                    <p>Заказов пока нет</p>
                    <a href="passenger" class="btn btn-primary">Создать первый заказ</a>
                </div>
            `;
            return;
        }

        console.log("Полученные заказы:", data.orders);

        // Используем реальные данные с сервера, не перезаписываем статусы
        allOrders = data.orders.map(order => ({
            id: order.id,
            timestamp: order.timestamp,
            fromAddress: order.fromAddress,
            toAddress: order.toAddress,
            passengers: order.passengers,
            pets: order.pets,
            time: order.time,
            baggage: order.baggage,
            status: order.status || "pending", // Используем статус с сервера
            driver_price: order.driver_price,
            passengerPrice: order.passengerPrice,
            accepted_at: order.accepted_at
        }));

        renderOrdersList();

    } catch (err) {
        console.error("Ошибка при загрузке заказов:", err);
        container.innerHTML = `
            <div class="error-state">
                <i class="fas fa-exclamation-triangle"></i>
                <p>Ошибка загрузки заказов</p>
                <button class="btn btn-secondary" onclick="loadOrders()">
                    <i class="fas fa-redo"></i>
                    Попробовать снова
                </button>
            </div>
        `;
    }
}

function renderOrdersList() {
    const container = document.getElementById("ordersList");
    container.innerHTML = "";

    if (allOrders.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-clipboard-list"></i>
                <p>Заказов пока нет</p>
                <a href="passenger" class="btn btn-primary">Создать первый заказ</a>
            </div>
        `;
        return;
    }

    allOrders.forEach(order => {
        const orderElement = document.createElement("div");
        orderElement.classList.add("order-card");

        // Добавляем кнопку "Обработать" только для заказов со статусом "pending"
        const processButton = order.status === 'pending' ? `
            <div class="order-actions">
                <button class="btn btn-primary btn-sm" onclick="processOrder(${order.id})">
                    <i class="fas fa-cog"></i>
                    Обработать
                </button>
            </div>
        ` : '';

        // Определяем отображаемую цену
        const displayPrice = order.driver_price || order.passengerPrice;

        orderElement.innerHTML = `
            <div class="order-header">
                <div class="order-id">
                    <i class="fas fa-hashtag"></i>
                    Заказ #${order.id}
                </div>
                <div class="order-date">
                    <i class="fas fa-calendar"></i>
                    ${new Date(order.timestamp).toLocaleString("ru-RU")}
                </div>
            </div>
            <div class="order-body">
                <div class="order-route">
                    <div class="route-point">
                        <i class="fas fa-circle" style="color: var(--success);"></i>
                        <span>${order.fromAddress || 'Не указан'}</span>
                    </div>
                    <div class="route-point">
                        <i class="fas fa-flag" style="color: var(--secondary);"></i>
                        <span>${order.toAddress || 'Не указан'}</span>
                    </div>
                </div>
                <div class="order-details-grid">
                    <div class="order-detail">
                        <i class="fas fa-users"></i>
                        <span>${order.passengers || '1'} чел.</span>
                    </div>
                    <div class="order-detail">
                        <i class="fas fa-paw"></i>
                        <span>${getPetsText(order.pets)}</span>
                    </div>
                    <div class="order-detail">
                        <i class="fas fa-clock"></i>
                        <span>${getTimeText(order.time)}</span>
                    </div>
                    <div class="order-detail">
                        <i class="fas fa-suitcase"></i>
                        <span>${getBaggageText(order.baggage)}</span>
                    </div>
                </div>
            </div>
            <div class="order-footer">
                <div class="order-price">
                    <i class="fas fa-ruble-sign"></i>
                    ${displayPrice || '0'}
                </div>
                <div class="order-status ${getStatusClass(order.status)}">
                    <i class="fas ${getStatusIcon(order.status)}"></i>
                    ${getStatusText(order.status)}
                </div>
            </div>
            ${processButton}
        `;

        container.appendChild(orderElement);
    });
}

function getStatusClass(status) {
    const classMap = {
        'pending': 'pending',
        'accepted': 'accepted', 
        'completed': 'completed',
        'declined': 'declined'
    };
    return classMap[status] || 'pending';
}

function getStatusIcon(status) {
    const iconMap = {
        'pending': 'fa-clock',
        'accepted': 'fa-check',
        'completed': 'fa-check-circle',
        'declined': 'fa-times-circle'
    };
    return iconMap[status] || 'fa-clock';
}

function getStatusText(status) {
    const textMap = {
        'pending': 'Ожидает',
        'accepted': 'Принят',
        'completed': 'Завершен',
        'declined': 'Отклонен'
    };
    return textMap[status] || status;
}

function processOrder(orderId) {
    const order = allOrders.find(o => o.id === orderId);
    if (!order) {
        alert('Заказ не найден');
        return;
    }

    if (order.status !== 'pending') {
        alert('Этот заказ уже обработан');
        return;
    }

    // Сохраняем выбранный заказ в localStorage
    localStorage.setItem('selectedOrder', JSON.stringify(order));
    
    // Переходим на страницу обработки заказа
    window.location.href = '/order_processing';
}

function getPetsText(pets) {
    const petsMap = { no: "Нет", small: "Маленькие", medium: "Средние", large: "Крупные" };
    return petsMap[pets] || "Нет";
}

function getTimeText(time) {
    const timeMap = { now: "Сейчас", "15min": "Через 15 мин", "30min": "Через 30 мин", "1hour": "Через 1 час" };
    return timeMap[time] || "Сейчас";
}

function getBaggageText(baggage) {
    const baggageMap = { no: "Нет", small: "Маленький", medium: "Средний", large: "Крупный" };
    return baggageMap[baggage] || "Нет";
}