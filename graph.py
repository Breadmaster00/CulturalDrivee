import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

class TaxiPriceAnalyzer:
    def __init__(self, file_path):
        """
        Инициализация анализатора для такси данных
        """
        self.file_path = file_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """
        Загрузка данных из CSV файла с улучшенной информацией
        """
        try:
            self.df = pd.read_csv(self.file_path)
            print("✅ Данные успешно загружены!")
            print("📊" + "="*50 + "📊")
            print(f"📈 Размер данных: {self.df.shape[0]:,} строк × {self.df.shape[1]} столбцов")
            print(f"📅 Период данных: {self.df['order_timestamp'].min()[:10]} - {self.df['order_timestamp'].max()[:10]}")
            
            # Базовая статистика при загрузке
            print("\n🔍 Первые 3 строки данных:")
            display(self.df.head(3).style.set_caption("Предварительный просмотр данных"))
            
        except Exception as e:
            print(f"❌ Ошибка при загрузке файла: {e}")
    
    def prepare_data(self):
        """
        Подготовка данных для анализа с улучшенной обработкой
        """
        print("\n🔄 Подготовка данных...")
        
        # Преобразуем timestamp в datetime
        datetime_columns = ['order_timestamp', 'tender_timestamp', 'driver_reg_date']
        for col in datetime_columns:
            self.df[col] = pd.to_datetime(self.df[col])
        
        # Сортируем по времени заказа
        self.df = self.df.sort_values('order_timestamp')
        
        # Создаем дополнительные признаки для анализа
        self.df['hour'] = self.df['order_timestamp'].dt.hour
        self.df['day_of_week'] = self.df['order_timestamp'].dt.day_name()
        self.df['date'] = self.df['order_timestamp'].dt.date
        self.df['price_difference'] = self.df['price_bid_local'] - self.df['price_start_local']
        self.df['price_ratio'] = self.df['price_bid_local'] / self.df['price_start_local']
        self.df['distance_km'] = self.df['distance_in_meters'] / 1000
        self.df['pickup_minutes'] = self.df['pickup_in_seconds'] / 60
        self.df['duration_minutes'] = self.df['duration_in_seconds'] / 60
        
        # Создаем категории для времени суток
        def time_of_day(hour):
            if 5 <= hour < 12: return 'Утро'
            elif 12 <= hour < 17: return 'День'
            elif 17 <= hour < 22: return 'Вечер'
            else: return 'Ночь'
        
        self.df['time_of_day'] = self.df['hour'].apply(time_of_day)
        
        print("✅ Данные подготовлены!")
        print(f"📅 Полный период: {self.df['order_timestamp'].min().strftime('%d.%m.%Y')} - {self.df['order_timestamp'].max().strftime('%d.%m.%Y')}")
        print(f"🕒 Всего дней: {self.df['date'].nunique()}")
    
    def set_style(self):
        """Установка красивого стиля для всех графиков"""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Устанавливаем стиль для красивых графиков
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['axes.labelsize'] = 14
    
    def create_summary_dashboard(self):
        """
        Создание информационной панели с ключевыми метриками
        """
        self.set_style()
        
        # Рассчитываем ключевые метрики
        total_orders = len(self.df)
        completed_orders = len(self.df[self.df['is_done'] == 'done'])
        completion_rate = (completed_orders / total_orders) * 100
        avg_price = self.df['price_bid_local'].mean()
        avg_distance = self.df['distance_km'].mean()
        avg_pickup_time = self.df['pickup_minutes'].mean()
        
        # Создаем дашборд
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 4, figure=fig)
        
        # 1. Заголовок и основные метрики
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.axis('off')
        ax_main.text(0.1, 0.8, '🚕 АНАЛИТИКА СЕРВИСА ТАКСИ', 
                    fontsize=24, fontweight='bold', color='#2E86AB')
        ax_main.text(0.1, 0.6, f'Период анализа: {self.df["date"].min()} - {self.df["date"].max()}', 
                    fontsize=14, color='#666666')
        
        # Отображаем ключевые метрики
        metrics = [
            ('📦 Всего заказов', f'{total_orders:,}', '#A23B72'),
            ('✅ Завершено', f'{completed_orders:,} ({completion_rate:.1f}%)', '#18A558'),
            ('💰 Средняя цена', f'{avg_price:.1f}', '#F18F01'),
            ('📏 Средняя дистанция', f'{avg_distance:.1f} км', '#C73E1D'),
            ('⏱️ Время подачи', f'{avg_pickup_time:.1f} мин', '#3E92CC')
        ]
        
        for i, (label, value, color) in enumerate(metrics):
            ax_main.text(0.1 + i*0.18, 0.3, label, fontsize=12, color='#666666')
            ax_main.text(0.1 + i*0.18, 0.2, value, fontsize=16, fontweight='bold', color=color)
        
        # 2. Распределение статусов заказов
        ax_pie = fig.add_subplot(gs[1, 0])
        status_counts = self.df['is_done'].value_counts()
        colors = ['#18A558', '#C73E1D'] if len(status_counts) == 2 else ['#18A558', '#C73E1D', '#666666']
        wedges, texts, autotexts = ax_pie.pie(status_counts.values, labels=status_counts.index, 
                                             autopct='%1.1f%%', colors=colors, startangle=90)
        ax_pie.set_title('📊 Статус заказов', fontweight='bold')
        
        # 3. Распределение по времени суток
        ax_time = fig.add_subplot(gs[1, 1])
        time_counts = self.df['time_of_day'].value_counts()
        bars = ax_time.bar(time_counts.index, time_counts.values, 
                          color=['#FF9F1C', '#2EC4B6', '#E71D36', '#6A0572'])
        ax_time.set_title('🕒 Распределение по времени суток', fontweight='bold')
        ax_time.set_ylabel('Количество заказов')
        
        # Добавляем значения на столбцы
        for bar in bars:
            height = bar.get_height()
            ax_time.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{height:,}', ha='center', va='bottom')
        
        # 4. Топ платформ
        ax_platform = fig.add_subplot(gs[1, 2])
        platform_counts = self.df['platform'].value_counts().head(5)
        bars = ax_platform.barh(range(len(platform_counts)), platform_counts.values,
                               color=sns.color_palette("viridis", len(platform_counts)))
        ax_platform.set_yticks(range(len(platform_counts)))
        ax_platform.set_yticklabels(platform_counts.index)
        ax_platform.set_title('🏢 Топ платформ', fontweight='bold')
        ax_platform.set_xlabel('Количество заказов')
        
        # 5. Средняя цена по дням недели
        ax_week = fig.add_subplot(gs[1, 3])
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_names_ru = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
        weekly_prices = self.df.groupby('day_of_week')['price_bid_local'].mean().reindex(day_order)
        
        ax_week.plot(day_names_ru, weekly_prices.values, marker='o', linewidth=3, 
                    markersize=8, color='#E4572E')
        ax_week.fill_between(day_names_ru, weekly_prices.values, alpha=0.3, color='#E4572E')
        ax_week.set_title('📈 Средняя цена по дням недели', fontweight='bold')
        ax_week.set_ylabel('Средняя цена')
        ax_week.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_price_analysis(self):
        """
        Улучшенный анализ цен во времени
        """
        self.set_style()
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('💰 Анализ динамики цен такси', fontsize=20, fontweight='bold', y=0.95)
        
        gs = GridSpec(2, 2, figure=fig)
        
        # 1. Динамика цен по часам
        ax1 = fig.add_subplot(gs[0, 0])
        hourly_stats = self.df.groupby('hour').agg({
            'price_start_local': 'mean',
            'price_bid_local': 'mean',
            'price_difference': 'mean'
        }).reset_index()
        
        ax1.plot(hourly_stats['hour'], hourly_stats['price_start_local'], 
                marker='o', linewidth=3, label='🎯 Стартовая цена', alpha=0.8, markersize=6)
        ax1.plot(hourly_stats['hour'], hourly_stats['price_bid_local'], 
                marker='s', linewidth=3, label='💰 Финальная цена', alpha=0.8, markersize=6)
        
        ax1.set_title('📊 Средние цены по часам суток', fontsize=16, fontweight='bold')
        ax1.set_xlabel('🕒 Час дня')
        ax1.set_ylabel('💵 Средняя цена')
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, 24, 2))
        ax1.fill_between(hourly_stats['hour'], hourly_stats['price_start_local'], 
                        hourly_stats['price_bid_local'], alpha=0.2)
        
        # 2. Разница цен по дням недели
        ax2 = fig.add_subplot(gs[0, 1])
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_names_ru = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
        daily_diff = self.df.groupby('day_of_week')['price_difference'].mean().reindex(day_order)
        
        colors = ['lightcoral' if x < 0 else 'lightgreen' for x in daily_diff.values]
        bars = ax2.bar(day_names_ru, daily_diff.values, color=colors, alpha=0.7, edgecolor='black')
        
        ax2.set_title('📈 Разница между финальной и стартовой ценой', fontsize=16, fontweight='bold')
        ax2.set_xlabel('День недели')
        ax2.set_ylabel('Разница цен')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Добавляем аннотации
        for bar, value in zip(bars, daily_diff.values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:+.1f}', ha='center', va='bottom' if value > 0 else 'top',
                    fontweight='bold', fontsize=10)
        
        # 3. Зависимость цены от расстояния
        ax3 = fig.add_subplot(gs[1, 0])
        scatter = ax3.scatter(self.df['distance_km'], self.df['price_bid_local'],
                            c=self.df['duration_minutes'], alpha=0.6, 
                            cmap='viridis', s=50)
        ax3.set_title('📏 Зависимость цены от расстояния и времени', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Расстояние (км)')
        ax3.set_ylabel('Финальная цена')
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Время поездки (минуты)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Распределение цен
        ax4 = fig.add_subplot(gs[1, 1])
        n, bins, patches = ax4.hist(self.df['price_bid_local'], bins=30, alpha=0.7, 
                                   color='lightgreen', edgecolor='black', density=True)
        
        # Добавляем линии среднего и медианы
        mean_price = self.df['price_bid_local'].mean()
        median_price = self.df['price_bid_local'].median()
        
        ax4.axvline(mean_price, color='red', linestyle='--', linewidth=2, 
                   label=f'Среднее: {mean_price:.1f}')
        ax4.axvline(median_price, color='blue', linestyle='--', linewidth=2, 
                   label=f'Медиана: {median_price:.1f}')
        
        ax4.set_title('📊 Распределение цен заказов', fontsize=16, fontweight='bold')
        ax4.set_xlabel('Финальная цена')
        ax4.set_ylabel('Плотность вероятности')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_platform_comparison(self):
        """
        Улучшенное сравнение платформ
        """
        self.set_style()
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('🏢 Сравнение такси-платформ', fontsize=20, fontweight='bold', y=0.95)
        
        gs = GridSpec(2, 2, figure=fig)
        
        # 1. Сравнение цен по платформам
        ax1 = fig.add_subplot(gs[0, 0])
        platform_stats = self.df.groupby('platform').agg({
            'price_start_local': 'mean',
            'price_bid_local': 'mean',
            'is_done': lambda x: (x == 'done').mean() * 100
        }).round(2).sort_values('price_bid_local', ascending=False)
        
        x_pos = np.arange(len(platform_stats.index))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, platform_stats['price_start_local'], 
                       width, label='🎯 Стартовая цена', alpha=0.7, color='#FF6B6B')
        bars2 = ax1.bar(x_pos + width/2, platform_stats['price_bid_local'], 
                       width, label='💰 Финальная цена', alpha=0.7, color='#4ECDC4')
        
        ax1.set_title('💵 Средние цены по платформам', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Платформа')
        ax1.set_ylabel('Средняя цена')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(platform_stats.index, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Процент завершения заказов
        ax2 = fig.add_subplot(gs[0, 1])
        completion_data = platform_stats['is_done'].sort_values(ascending=False)
        colors = plt.cm.RdYlGn(completion_data / 100)
        
        bars = ax2.bar(completion_data.index, completion_data.values, color=colors, alpha=0.7)
        ax2.set_title('✅ Процент успешных заказов', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Платформа')
        ax2.set_ylabel('Процент завершения (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Добавляем значения на столбцы
        for bar, value in zip(bars, completion_data.values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Boxplot распределения цен
        ax3 = fig.add_subplot(gs[1, 0])
        platform_data = [self.df[self.df['platform'] == platform]['price_bid_local'] 
                        for platform in self.df['platform'].unique()]
        
        boxplot = ax3.boxplot(platform_data, labels=self.df['platform'].unique(),
                             patch_artist=True)
        
        # Раскрашиваем boxplot
        colors = plt.cm.Set3(np.linspace(0, 1, len(platform_data)))
        for patch, color in zip(boxplot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax3.set_title('📦 Распределение цен по платформам', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Платформа')
        ax3.set_ylabel('Финальная цена')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Время подачи по платформам
        ax4 = fig.add_subplot(gs[1, 1])
        pickup_stats = self.df.groupby('platform')['pickup_minutes'].agg(['mean', 'std']).sort_values('mean')
        
        y_pos = np.arange(len(pickup_stats.index))
        ax4.barh(y_pos, pickup_stats['mean'], xerr=pickup_stats['std'], 
                alpha=0.7, color='#6A0572', capsize=5)
        
        ax4.set_title('⏱️ Среднее время подачи такси', fontsize=16, fontweight='bold')
        ax4.set_xlabel('Время подачи (минуты)')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(pickup_stats.index)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_car_model_analysis(self):
        """
        Улучшенный анализ по моделям автомобилей
        """
        self.set_style()
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('🚗 Анализ по моделям автомобилей и рейтингу водителей', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        gs = GridSpec(1, 2, figure=fig)
        
        # 1. Топ моделей автомобилей по ценам
        ax1 = fig.add_subplot(gs[0, 0])
        top_carmodels = self.df['carmodel'].value_counts().head(10).index
        carmodel_data = self.df[self.df['carmodel'].isin(top_carmodels)]
        
        carmodel_stats = carmodel_data.groupby('carmodel').agg({
            'price_bid_local': ['mean', 'count'],
            'driver_rating': 'mean'
        }).round(2)
        carmodel_stats.columns = ['mean_price', 'order_count', 'mean_rating']
        carmodel_stats = carmodel_stats.sort_values('mean_price', ascending=True)
        
        # Создаем scatter plot с размером точек по количеству заказов
        scatter = ax1.scatter(carmodel_stats['mean_price'], carmodel_stats.index,
                             s=carmodel_stats['order_count']/10, 
                             c=carmodel_stats['mean_rating'], cmap='RdYlGn', 
                             alpha=0.7, edgecolors='black')
        
        ax1.set_title('💎 Средние цены по моделям автомобилей\n(размер точки = количество заказов)', 
                     fontsize=16, fontweight='bold')
        ax1.set_xlabel('Средняя финальная цена')
        ax1.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Средний рейтинг водителя')
        
        # 2. Влияние рейтинга водителя
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Создаем группы рейтинга
        self.df['rating_group'] = pd.cut(self.df['driver_rating'], 
                                       bins=[0, 3.0, 3.5, 4.0, 4.5, 5.0],
                                       labels=['<3.0', '3.0-3.5', '3.5-4.0', '4.0-4.5', '4.5-5.0'])
        
        rating_stats = self.df.groupby('rating_group').agg({
            'price_bid_local': 'mean',
            'is_done': lambda x: (x == 'done').mean() * 100,
            'pickup_minutes': 'mean'
        }).dropna()
        
        x_pos = np.arange(len(rating_stats.index))
        width = 0.25
        
        # График средней цены
        bars1 = ax2.bar(x_pos - width, rating_stats['price_bid_local'], width,
                       label='💰 Средняя цена', alpha=0.7, color='#2E86AB')
        ax2.set_ylabel('Средняя цена', color='#2E86AB')
        ax2.tick_params(axis='y', labelcolor='#2E86AB')
        
        # График процента завершения на второй оси
        ax2b = ax2.twinx()
        bars2 = ax2b.bar(x_pos + width, rating_stats['is_done'], width,
                        label='✅ % завершения', alpha=0.7, color='#A23B72')
        ax2b.set_ylabel('Процент завершения (%)', color='#A23B72')
        ax2b.tick_params(axis='y', labelcolor='#A23B72')
        
        ax2.set_title('📊 Влияние рейтинга водителя', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Группа рейтинга водителя')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(rating_stats.index)
        
        # Объединяем легенды
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def plot_completion_analysis(self):
        """
        Улучшенный анализ завершения заказов
        """
        self.set_style()
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('📊 Анализ завершения заказов', fontsize=20, fontweight='bold', y=0.95)
        
        gs = GridSpec(2, 2, figure=fig)
        
        completed = self.df[self.df['is_done'] == 'done']
        canceled = self.df[self.df['is_done'] == 'cancel']
        
        # 1. Сравнение ключевых метрик
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['price_bid_local', 'distance_km', 'pickup_minutes', 'duration_minutes']
        metric_names = ['Цена', 'Расстояние (км)', 'Время подачи (мин)', 'Длительность (мин)']
        
        completed_means = [completed[metric].mean() for metric in metrics]
        canceled_means = [canceled[metric].mean() for metric in metrics]
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, completed_means, width, 
                       label='✅ Завершено', color='#18A558', alpha=0.7)
        bars2 = ax1.bar(x_pos + width/2, canceled_means, width, 
                       label='❌ Отменено', color='#C73E1D', alpha=0.7)
        
        ax1.set_title('📈 Сравнение ключевых метрик', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Метрика')
        ax1.set_ylabel('Среднее значение')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(metric_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Распределение расстояний
        ax2 = fig.add_subplot(gs[0, 1])
        max_distance = min(self.df['distance_km'].quantile(0.95), 50)  # Ограничиваем для лучшей визуализации
        filtered_data = self.df[self.df['distance_km'] <= max_distance]
        
        completed_filtered = filtered_data[filtered_data['is_done'] == 'done']
        canceled_filtered = filtered_data[filtered_data['is_done'] == 'cancel']
        
        ax2.hist([completed_filtered['distance_km'], canceled_filtered['distance_km']],
                bins=20, alpha=0.7, label=['✅ Завершено', '❌ Отменено'],
                color=['#18A558', '#C73E1D'], edgecolor='black', density=True)
        ax2.set_title('📏 Распределение расстояний поездок', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Расстояние (км)')
        ax2.set_ylabel('Плотность вероятности')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Время подачи по статусу
        ax3 = fig.add_subplot(gs[1, 0])
        max_pickup = min(self.df['pickup_minutes'].quantile(0.95), 30)
        filtered_pickup = self.df[self.df['pickup_minutes'] <= max_pickup]
        
        completed_pickup = filtered_pickup[filtered_pickup['is_done'] == 'done']
        canceled_pickup = filtered_pickup[filtered_pickup['is_done'] == 'cancel']
        
        ax3.hist([completed_pickup['pickup_minutes'], canceled_pickup['pickup_minutes']],
                bins=15, alpha=0.7, label=['✅ Завершено', '❌ Отменено'],
                color=['#18A558', '#C73E1D'], edgecolor='black', density=True)
        ax3.set_title('⏱️ Распределение времени подачи такси', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Время подачи (минуты)')
        ax3.set_ylabel('Плотность вероятности')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Процент завершения по часам
        ax4 = fig.add_subplot(gs[1, 1])
        hourly_completion = self.df.groupby('hour')['is_done'].apply(
            lambda x: (x == 'done').mean() * 100
        )
        
        line = ax4.plot(hourly_completion.index, hourly_completion.values, 
                       marker='o', linewidth=3, markersize=6, color='#6A0572')
        ax4.fill_between(hourly_completion.index, hourly_completion.values, 
                        alpha=0.3, color='#6A0572')
        ax4.axhline(50, color='red', linestyle='--', alpha=0.7, 
                   label='Порог 50%')
        
        # Закрашиваем области выше и ниже порога
        ax4.fill_between(hourly_completion.index, hourly_completion.values, 50,
                        where=(hourly_completion.values >= 50), 
                        color='green', alpha=0.2, interpolate=True)
        ax4.fill_between(hourly_completion.index, hourly_completion.values, 50,
                        where=(hourly_completion.values < 50), 
                        color='red', alpha=0.2, interpolate=True)
        
        ax4.set_title('🕒 Процент успешных заказов по часам', fontsize=16, fontweight='bold')
        ax4.set_xlabel('Час дня')
        ax4.set_ylabel('Процент завершения (%)')
        ax4.set_xticks(range(0, 24, 2))
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.show()
    
    def comprehensive_analysis(self):
        """
        Полный улучшенный анализ данных
        """
        print("🚕" + "="*60 + "🚕")
        print("           КОМПЛЕКСНЫЙ АНАЛИЗ ДАННЫХ ТАКСИ")
        print("🚕" + "="*60 + "🚕")
        
        # Подготовка данных
        self.prepare_data()
        
        # Создаем информационную панель
        self.create_summary_dashboard()
        
        # Расширенная статистика
        print("\n📊" + "="*50 + "📊")
        print("ДЕТАЛЬНАЯ СТАТИСТИКА:")
        print("="*50)
        
        # Основные метрики
        total_orders = len(self.df)
        completed = len(self.df[self.df['is_done'] == 'done'])
        canceled = len(self.df[self.df['is_done'] == 'cancel'])
        completion_rate = (completed / total_orders) * 100
        
        stats_data = {
            'Метрика': [
                '📦 Всего заказов',
                '✅ Успешных заказов', 
                '❌ Отмененных заказов',
                '📈 Процент завершения',
                '💰 Средняя финальная цена',
                '🎯 Средняя стартовая цена',
                '📏 Средняя разница цен',
                '🛣️ Среднее расстояние',
                '⏱️ Среднее время подачи',
                '⏳ Средняя длительность'
            ],
            'Значение': [
                f'{total_orders:,}',
                f'{completed:,}',
                f'{canceled:,}',
                f'{completion_rate:.1f}%',
                f'{self.df["price_bid_local"].mean():.2f}',
                f'{self.df["price_start_local"].mean():.2f}',
                f'{self.df["price_difference"].mean():.2f}',
                f'{self.df["distance_km"].mean():.2f} км',
                f'{self.df["pickup_minutes"].mean():.1f} мин',
                f'{self.df["duration_minutes"].mean():.1f} мин'
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        display(stats_df.style.set_caption("📈 Ключевые метрики анализа").hide_index())
        
        # Статистика по платформам
        print("\n🏢" + "="*50 + "🏢")
        print("СТАТИСТИКА ПО ПЛАТФОРМАМ:")
        print("="*50)
        
        platform_analysis = self.df.groupby('platform').agg({
            'price_bid_local': ['mean', 'std', 'count'],
            'is_done': lambda x: (x == 'done').mean() * 100,
            'distance_km': 'mean',
            'pickup_minutes': 'mean'
        }).round(2)
        
        platform_analysis.columns = ['Средняя цена', 'Стд цена', 'Кол-во заказов', 
                                   '% завершения', 'Ср. расстояние', 'Ср. время подачи']
        platform_analysis = platform_analysis.sort_values('Кол-во заказов', ascending=False)
        
        display(platform_analysis.style.set_caption("📊 Сравнение платформ").format({
            '% завершения': '{:.1f}%',
            'Средняя цена': '{:.2f}',
            'Ср. расстояние': '{:.2f} км',
            'Ср. время подачи': '{:.1f} мин'
        }))
        
        # Построение графиков
        print("\n📈" + "="*50 + "📈")
        print("ВИЗУАЛИЗАЦИЯ ДАННЫХ:")
        print("="*50)
        
        self.plot_price_analysis()
        self.plot_platform_comparison()
        self.plot_car_model_analysis()
        self.plot_completion_analysis()
        
        print("\n🎉 Анализ завершен! Все графики и статистика доступны выше.")

# Использование
if __name__ == "__main__":
    # Укажите путь к вашему CSV файлу
    file_path = "train.csv"  # Замените на актуальный путь
    
    analyzer = TaxiPriceAnalyzer(file_path)
    analyzer.comprehensive_analysis()