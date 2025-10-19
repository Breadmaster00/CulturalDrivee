import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l2
from sklearn.metrics import roc_auc_score, classification_report
from config import CalculatorConfig  # Импорт конфига
import matplotlib.pyplot as plt
import os

class NeuralNetworkModel:
    def __init__(self, input_dim):
        self.model = None
        self.input_dim = input_dim
        self.history = None

    def build_model(self):
        """Построение архитектуры нейронной сети"""
        model = Sequential()
        
        # Входной слой
        model.add(Dense(CalculatorConfig.NN_HIDDEN_LAYERS[0], input_dim=self.input_dim, activation='relu', 
                       kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(CalculatorConfig.NN_DROPOUT_RATE))
        
        # Скрытые слои
        for units in CalculatorConfig.NN_HIDDEN_LAYERS[1:]:
            model.add(Dense(units, activation='relu', kernel_regularizer=l2(0.001)))
            model.add(BatchNormalization())
            model.add(Dropout(CalculatorConfig.NN_DROPOUT_RATE))
        
        # Выходной слой
        model.add(Dense(1, activation='sigmoid'))
        
        # Компиляция модели
        optimizer = Adam(learning_rate=CalculatorConfig.NN_LEARNING_RATE)
        model.compile(optimizer=optimizer, 
                     loss='binary_crossentropy', 
                     metrics=['accuracy', 'AUC'])
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=CalculatorConfig.NN_EPOCHS, batch_size=CalculatorConfig.NN_BATCH_SIZE):
        """Обучение нейронной сети"""
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7),
            ModelCheckpoint(CalculatorConfig.MODEL_PATH, monitor='val_auc', save_best_only=True, mode='max')
        ]
        
        # Обучение
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict_proba(self, X):
        """Предсказание вероятностей"""
        if self.model is None:
            raise ValueError("Модель не обучена!")
        predictions = self.model.predict(X, verbose=0).flatten()
        # Преобразуем numpy array в Python list
        return [float(x) for x in predictions]
    
    def plot_training_history(self):
        """Визуализация процесса обучения"""
        if self.history is None:
            print("Нет истории обучения для визуализации")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        # Accuracy
        axes[1].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        
        # AUC - используем правильные ключи
        auc_key = 'auc' if 'auc' in self.history.history else 'val_auc'
        val_auc_key = 'val_auc' if 'val_auc' in self.history.history else 'val_auc'
        
        if auc_key in self.history.history:
            axes[2].plot(self.history.history[auc_key], label='Training AUC')
            if val_auc_key in self.history.history:
                axes[2].plot(self.history.history[val_auc_key], label='Validation AUC')
            axes[2].set_title('Model AUC')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('AUC')
            axes[2].legend()
        else:
            # Если AUC нет, показываем точность еще раз или другую метрику
            axes[2].plot(self.history.history['accuracy'], label='Training Accuracy', alpha=0.5)
            axes[2].plot(self.history.history['val_accuracy'], label='Validation Accuracy', alpha=0.5)
            axes[2].set_title('Model Accuracy (Detailed)')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Accuracy')
            axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(CalculatorConfig.VISUALIZATION_DIR, 'training_history.png'))
        plt.close()
        print("📊 График обучения сохранен")