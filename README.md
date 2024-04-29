# Задача ранжирования документов

## Описание проекта
Реализованая модель предназначена для предсказания рангов документов на основе их фичей в рамках одной сессии.

## Структура данных
Датасет [intern_task.csv](https://drive.google.com/file/d/1viFKqtYTtTiP9_EdBXVpCmWbNmxDiXWG/view?usp=sharing) содержит следующие колонки:
- `query_id` - идентификатор поисковой сессии.
- `rank` - оценка релевантности документа.
- Фичи документа.

## Подготовка данных
В ходе предобработки, были удалены сессии с одним объектом, произведено разделенние на тренировочную, валидационную и тестовую выборки с учетом стратификации по `query_id`.

## Модель
Использовалась модель LightGBM с функцией потерь `lambdarank`. Основные параметры модели включают скорость обучения, количество листьев и метрику `ndcg`.

## Оценка модели
Модель оценивалась с помощью метрики NDCG@5, где было достигнуто значение 0.999 на тестовом наборе данных.

## Заключение
Результаты показывают высокую эффективность подхода в задаче ранжирования.

## Как использовать
Для использования проекта следует установить библиотеку LightGBM и [загрузить датасет](https://drive.google.com/file/d/1viFKqtYTtTiP9_EdBXVpCmWbNmxDiXWG/view?usp=sharing). Далее выполнить представленный скрипт для обучения модели и оценки её работы.