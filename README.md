### Файлы сайта лежат в папке project
### Развёртывание проекта на Linux
```
git clone https://github.com/Breadmaster00/CulturalDrivee
cd CulturalDrivee
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd project

uvicorn --reload app:app --port "1111"
```
далее переходим на http://127.0.0.1:1111/orders

app.py - файл для запуска сайта

calculator.py - расчеты

config.py - конфиги нейронки

data_processor.py - для анализа файла csv

graph.py - файл создающий на основе csv таблицы графики информации

nn_model.py - файл с голой нейронкой

prise_optimizer.py - вычесляет наилучший вариант

shemas.py - определяет структуру данных

tamplates - html страницы

static - css стили и js скрипты использующиеся на сайте

requirements.txt - все используемые библиотеки
