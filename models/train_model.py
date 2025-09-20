import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Загружаем данные
data_path = 'data/Student_Marks.csv'
df = pd.read_csv(data_path)

# Признаки и целевая переменная
X = df[['number_courses', 'time_study']]  # 2 признака
y = df['Marks']                           # Целевая переменная

# Обучаем модель
model = LinearRegression()
model.fit(X, y)

# Создаем папку models, если не существует
os.makedirs('models', exist_ok=True)

# Сохраняем модель
model_path = 'models/model.joblib'
joblib.dump(model, model_path)
print(f"Модель обучена и сохранена в {model_path}")

# Выводим коэффициенты модели
print("Коэффициенты модели (веса признаков):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

# Вывод интерсепта (свободного члена)
print(f"Свободный член (intercept): {model.intercept_}")

# Предсказания на обучающей выборке
predictions = model.predict(X)

# Добавим предсказания в DataFrame и выведем первые 5 строк
df['Prediction'] = predictions
print("\nПервые 5 строк с предсказаниями:")
print(df.head())
