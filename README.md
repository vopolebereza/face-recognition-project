# Face Recognition Project

Проект по курсу: face alignment (Stacked Hourglass Network) + face recognition (CE / ArcFace / Triplet Loss)
на CelebA, полный пайплайн распознавания и дополнительные задания.

## Структура репозитория

```
├── README.md
├── Task1_FaceAlignment.ipynb
├── Task2_FaceRecognition.ipynb
├── Task3_FullPipeline.ipynb
├── Dop1_IdentificationRateMetric.ipynb
├── Dop2_TripletLoss.ipynb
├── Dop3_ArcFacePlusTriplet.ipynb
├── Dop4_OpenSourceReview.ipynb
└── data/
    ├── task1_selected_dataset.csv          # Датасет Task 1 (оригинальные имена файлов CelebA)
    └── task2_selected_dataset.csv          # Датасет Task 2 (оригинальные имена файлов CelebA)
```

## Веса моделей (облако)

Все веса моделей выложены в одном публичном Kaggle Dataset:
**https://www.kaggle.com/datasets/iuliiaburmistrova/face-recognition-project-model-weights**

| Модель | Файл в датасете |
|---|---|
| Stacked Hourglass (Task 1) | `hourglass_best.pth` |
| CE baseline (Task 2) | `ce_baseline_best.pth` |
| ArcFace (Task 2) | `arcface_best.pth` |
| Triplet Loss (Доп. 2) | `triplet_best.pth` |
| ArcFace + Triplet (Доп. 3) | `arcface_triplet_best.pth` |

## Где что смотреть

### Task 1 — Face Alignment

`Task1_FaceAlignment.ipynb`

- Датасет: 10 500 изображений CelebA (in-the-wild), отобранных случайно, ≤30 фото на личность.
- Архитектура: Stacked Hourglass Network, 4 stack'а, 256 каналов, глубина 4, intermediate supervision.
- Обучение: 20 эпох, MSE по heatmap'ам всех stack'ов, финальный val_loss = 0.00077.
- Результат: `align_face_from_photo(model, image_rgb)` — фото лица → выровненное лицо 112×112.
- Веса модели: `hourglass_best.pth` в общем датасете (см. раздел «Веса моделей» выше)

### Task 2 — Face Recognition (CE vs ArcFace)

`Task2_FaceRecognition.ipynb`

- Датасет: 400 личностей CelebA, identity-aware отбор (≥15 фото на личность, ≤30 фото на человека), 9 658 изображений, выровнены моделью из Task 1.
- Backbone: EfficientNet-B0, предобучен на ImageNet.
- Обучены и сравнены: `Linear + CrossEntropyLoss` (baseline) и `ArcMarginProduct` (ArcFace).

| Модель | val_accuracy (прогон 1) | val_accuracy (прогон 2) | Порог задания (0.7) |
|---|---|---|---|
| Cross-Entropy | 0.6197 | 0.6052 | не достигнут |
| ArcFace | 0.7212 | **0.7544** | ✅ достигнут |

Ход экспериментов (перебор регуляризаций для CE, ни одна не подняла результат выше ~0.62) и полный
теоретический разбор — в отчёте внутри ноутбука.

- Веса моделей: `ce_baseline_best.pth`, `arcface_best.pth` в общем датасете (см. раздел «Веса моделей» выше)

### Task 3 — Полный пайплайн

`Task3_FullPipeline.ipynb`

Пайплайн: детектор лиц MTCNN (facenet-pytorch) → alignment (Hourglass из Task 1) → эмбеддинги (ArcFace
backbone из Task 2). Демонстрация на паре "тот же человек / разные люди":

| Пара | Cosine similarity |
|---|---|
| Тот же человек | **0.803** |
| Разные люди | **−0.023** |

Важная техническая деталь: MTCNN-бокс требует того же отступа (`margin=0.35`), что использовался при
обучении Hourglass в Task 1 — без этого align даёт сильные искажения (описано в ноутбуке).

### Доп. 1 — Identification Rate Metric

`Dop1_IdentificationRateMetric.ipynb`

TPR@FPR на 15 query-личностях и 60 дистракторах, **не пересекающихся** с 400 обучающими личностями
Task 2. Сравнены все три обученные модели — CE, ArcFace и Triplet Loss (из Dop 2):

| FPR | CE | ArcFace | Triplet |
|---|---|---|---|
| 0.50 | 0.9889 | 0.9333 | 0.9889 |
| 0.20 | 0.8778 | 0.8111 | 0.7889 |
| 0.10 | 0.7000 | **0.7333** | 0.6444 |
| 0.05 | 0.5667 | **0.6111** | 0.4333 |

На строгих (практически значимых) порогах FPR ArcFace устойчиво лидирует среди всех подходов.

### Доп. 2 — Triplet Loss

`Dop2_TripletLoss.ipynb`

Backbone EfficientNet-B0 без классификационной головы, `TripletMarginWithDistanceLoss` на косинусном
расстоянии, эмбеддинги нормализованы через `BatchNorm1d(affine=False)`. Verification accuracy на
валидации выросла с ~0.58 до пика **0.7700** (эпоха 11), финал — 0.7612. Честное сравнение с CE/ArcFace
через Identification Rate metric — см. таблицу выше (Доп. 1): Triplet уступает обоим на строгих порогах
FPR, вероятно из-за отсутствия hard negative mining при формировании троек.

- Веса модели: `triplet_best.pth` в общем датасете (см. раздел «Веса моделей» выше)

### Доп. 3 — ArcFace + Triplet Loss (смесь)

`Dop3_ArcFacePlusTriplet.ipynb`

Комбинированный лосс `L = L_ArcFace + λ·L_Triplet` (λ=0.5), тройки формируются внутри
сбалансированного батча (`BalancedBatchSampler`). Результат — **лучший val_accuracy во всём проекте**:

| Модель | Лучший val_accuracy |
|---|---|
| CE baseline | 0.60–0.62 |
| ArcFace (чистый) | 0.72–0.75 |
| **ArcFace + Triplet** | **0.7865** |

Triplet-часть лосса обнулилась уже к эпохе 18-19 — весь дальнейший прогресс (до эпохи 40) шёл за счёт
ArcFace margin поверх более удачной стартовой конфигурации эмбеддингов, заданной ранним triplet-сигналом.

- Веса модели: `arcface_triplet_best.pth` в общем датасете (см. раздел «Веса моделей» выше)

### Доп. 4 — Обзор open-source решений

`Dop4_OpenSourceReview.ipynb`

Разбор InsightFace, DeepFace и facenet-pytorch (возможности, зоопарк моделей, установка, требования к
ресурсам). Все три протестированы на собственной паре фото "тот же человек / разные люди":

| Библиотека / модель | Тот же человек | Разные люди | Верно? |
|---|---|---|---|
| Наша ArcFace-модель | cos_sim = 0.803 | cos_sim = −0.023 | ✅ |
| InsightFace (buffalo_l) | cos_sim = 0.585 | cos_sim = 0.080 | ✅ |
| facenet-pytorch | cos_sim = 0.593 | cos_sim = −0.082 | ✅ |
| DeepFace / VGG-Face | distance = 0.484 | — | ✅ |
| DeepFace / ArcFace | distance = 0.816 | distance = 0.956 | ❌ |

DeepFace показала нестабильный результат на большинстве встроенных моделей — вероятная причина
разобрана в ноутбуке (конфликт формата входа: библиотека заново детектирует лицо внутри уже тесно
кропнутого изображения).

## Как запустить

Все ноутбуки рассчитаны на Kaggle Notebooks (бесплатный GPU T4 достаточно) или Google Colab.
Task 1 и Task 2 разбиты на Фазу A (CPU — скачивание/подготовка данных) и Фазу B (GPU — обучение/инференс).
Task 3 и все доп. задания используют выходы (веса, датасеты) предыдущих ноутбуков — подключаются через
`+ Add Data → Notebook Output Files` на Kaggle, либо вручную с Google Drive.

Датасет CelebA скачивается автоматически (Kaggle-зеркало `kevinpatel04/celeba-original-wild-images` +
landmarks/identity с Google Drive) — см. первую ячейку каждого ноутбука.

## Технические заметки

- **Честный подсчёт accuracy для ArcFace.** `ArcMarginProduct` вычитает margin из логита истинного
  класса — если считать accuracy по этим же логитам, результат занижается искусственно (в нашем случае
  с 0.72 до 0.28). Используется метод `get_logits`/`predict_logits` — margin применяется только к
  функции потерь во время обучения, не при измерении итогового качества.
- **CE не достигает порога 0.7** несмотря на перебор регуляризаций (label smoothing, аугментации,
  dropout, заморозка backbone, mixup, сокращение числа классов) — согласуется с теоретическим
  обоснованием ArcFace: обычный softmax+CE не даёт явного давления на компактность эмбеддингов внутри
  класса, чем и объясняется разница с margin-based подходами.
- **Margin вокруг bbox важен на каждом этапе пайплайна**, где используется Hourglass-alignment — как
  при обучении (bbox CelebA + запас 0.35), так и при инференсе (MTCNN-бокс + тот же запас) —
  несовпадение форматов входа даёт заметные искажения align'а (см. Task 3).
