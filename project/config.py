class CalculatorConfig:
    MODEL_PATH = 'price_optimizer_nn_model.h5'
    SCALER_PATH = 'scaler.pkl'
    DATA_PATH = 'train.csv'
    FEATURE_COLUMNS = [
        'distance_in_meters', 'duration_in_seconds', 'pickup_in_meters', 'pickup_in_seconds',
        'price_start_local', 'price_bid_local', 'driver_rating',
        'hour', 'day_of_week', 'month', 'carname_encoded', 'carmodel_encoded', 'platform_encoded',
        'price_ratio', 'distance_per_minute', 'pickup_time_ratio'
    ]
    CATEGORICAL_COLUMNS = ['carname', 'carmodel', 'platform']
    REQUIRED_COLUMNS = [
        'order_timestamp', 'distance_in_meters', 'duration_in_seconds', 
        'pickup_in_meters', 'pickup_in_seconds', 'price_start_local', 
        'price_bid_local', 'driver_rating', 'carname', 'carmodel', 
        'platform', 'is_done'
    ]
    VISUALIZATION_DIR = 'visualizations'  # Добавлено из вашего кода
    USE_SMOTE = True  # Добавлено
    NN_EPOCHS = 200
    NN_BATCH_SIZE = 256
    NN_LEARNING_RATE = 0.001
    NN_HIDDEN_LAYERS = [1024, 512, 258, 128, 64, 32]
    NN_DROPOUT_RATE = 0.2