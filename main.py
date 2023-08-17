import odometr.odometr

import cv2 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO


start_img = cv2.imread('foto/10.jpg')

if __name__ == '__main__':
    
        test_odo = odometr.odometr.Odometr()
        position = test_odo.detection_model_result(start_img)
        a = position.loc[0]['upper_left_y'].astype(int)
        b = position.loc[0]['lower_right_y'].astype(int)
        c = position.loc[0]['upper_left_x'].astype(int)
        d = position.loc[0]['lower_right_x'].astype(int)
        odo_img = start_img[a:b, c:d]
        # Получение таблицы параметров по найденным моделью символам
def get_digits_table(model, path_to_image, classes, conf=0.25, iou=0.7):
    results = model.predict(source=path_to_image, conf=conf, classes=classes, save=False, iou=iou)

    positions = pd.DataFrame(results[0].boxes.xywh, columns=['x_center', 'y_center', 'width', 'height'])
    positions['digit'] = results[0].boxes.cls
    positions['number_id'] = -1
    positions['confidence'] = results[0].boxes.conf
    positions = positions.sort_values(['x_center']).reset_index()
    positions = positions.drop(columns=['index'])

    return positions


# Поиск групп рядом стоящих символов, для составления из них чисел (слов)
# Этот код могу переписать более оптимально с точки зрения алгоритмической сложности
def find_symbol_groups(digits_table, x_distance_coef = 0.8, y_distance_coef = 0.5, height_coef = 0.1):
    digits = np.array(digits_table)

    # вместо циферных индексов таблицы digit:
    x = 0
    y = 1
    width = 2
    height = 3
    digit = 4
    number_id = 5
    conf = 6

    i = 0
    numbers = []
    while i < ( len(digits) - 1):
        digit = digits[i]
        if digit[number_id] != -1 :
            i+=1
            continue

        digit[number_id] = i # set number_id
        number = np.array([digit])
        # print(number)
        # Просматриваем цифры справа
        j = i + 1
        while j < len(digits):
            next_digit = digits[j]
            # Если цифру уже отнесли к другому числу, берем следующую цифру
            if next_digit[number_id] != -1 :
                j+=1
                continue
            else:
                # Если координата х следующей цифры слишком далеко,
                # считаем число законченным, цифру к нему не добавляем
                if (next_digit[x] > digit[x] + x_distance_coef *(digit[width] + next_digit[width])): # расстояние по х между цифрами
                    # print('координата х следующей цифры слишком далеко')
                    break
                # иначе проверяем высоту и координату у
                elif ((1 - height_coef) * digit[height] < next_digit[height] < (1 + height_coef) * digit[height] and  # высота букв
                      (digit[y] - y_distance_coef * next_digit[height] < next_digit[y] < digit[y] + y_distance_coef * next_digit[height])): # расстояние по у
                    number = np.append(number, next_digit)
                    next_digit[number_id] = i
                    digit = next_digit
                j+=1

        numbers.append(number)
        i+=1
    return pd.DataFrame(digits, columns = digits_table.columns).sort_values(['number_id', 'x_center'])

# Получение списка чисел со степенью уверенности алгоритма в распозновании для каждой цифры
def get_symbols_list(digits_table):
    result = []
    number_ids = list(digits_table.number_id.unique())
    table = np.array(digits_table[['digit',	'number_id',	'confidence']])

    i = 0
    for id in number_ids:
        number = ''
        confs = []
        while i < table.shape[0] and table[i][1] == id:
            number += str(int(table[i][0]))
            confs.append(table[i][2])
            i += 1
        confs = tuple(confs)
        result.append((number, confs))
    result.sort(key=lambda x: -int(x[0])) # сортировка по полученному числу в обратном порядке
    return result

# Получение пробега из фотографии одометра, усредненной уверенности по полученному числу и уверенностей по каждой цифре
def get_odo_value(model, path_to_image, classes, conf=0.25, iou=0.7, x_distance_coef = 0.8, y_distance_coef = 0.5, height_coef = 0.1):
    digits_table = get_digits_table(model, path_to_image, classes, conf=conf, iou=iou)
    # Если на одометре нет цифр
    if digits_table[digits_table.digit < 10].shape[0] < 1:
        return ('', np.nan, tuple())
    digits_table = find_symbol_groups(digits_table, x_distance_coef = x_distance_coef, y_distance_coef = y_distance_coef, height_coef = height_coef)
    digits_table = digits_table[digits_table.digit < 10] # выкидываем символы

    digits_table.digit = digits_table.digit.astype(int)

    symbols = get_symbols_list(digits_table)
    val = symbols[0][0]
    mean_conf = sum(symbols[0][1])/len(symbols[0][1])
    confs = symbols[0][1]
    return (val, mean_conf, confs)
model = YOLO('model_weights/digit_recognize_best.pt')
classes = [0,1,2,3,4,5,6,7,8,9,10]
path_to_image = odo_img


res = get_odo_value(model, path_to_image, classes, conf=0.3, iou=0.7, x_distance_coef = 0.8, y_distance_coef = 0.5, height_coef = 0.1)

print(f'\nПоказания одометра {res[0]}\nУверенность алгоритма по числу {res[1]:.2f}')
print(f'Уверенность по каждому символу:')
i = 0
for conf in res[2]:
    print(f'\t{res[0][i]}: {conf:.2f}')
    i += 1
image = odo_img

cv2.imshow('Odometr Image', start_img)
#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.waitKey(0) 
cv2.destroyAllWindows()    