# Решение домашнего задания по машинному обучению. Арсеньев Иван 972202
 

### Для корректной работы кода:
 

1.  Из папки `data` поместить все файлы в ваш репозиторий
     
2.  Из папки `notebooks` выбрать файл с моделью, которую вы будете использовать

-  В папке `models` хранятся сохраненные модели

-  Так выглядит папка с репозиторием
  
        v Project
        | v data
        | | test.csv
        | | train.csv
        | v models
        | | catboost_model.bin
        | v notebooks
        | | model.ipynb
        | v log
        | | model_building.log
        | model.py
        | requires.whl

- Для запуска скрипта можено использовать команды:
- Для обучения: model.py train 'data/train.csv'
- Для предсказаний: model.py predict 'data/test.csv'
