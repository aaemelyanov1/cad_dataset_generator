# CAD Dataset Generator

Генератор синтетического датасета 3D CAD-моделей для задач RLVR (reinforcement learning with verifiable rewards). Каждая модель представляется:

- **AST** — синтаксическое дерево операций (примитивы, эскизы, булевы операции, трансформации);
- **геометрией CadQuery** (OCCT) — твёрдое тело, валидируемое по объёму/габаритам;
- **программой на CadQuery** — исполняемый `.py`-код, порождающий ту же геометрию.

Генерация детерминирована: один и тот же `--seed` даёт один и тот же датасет (seed сэмпла выводится из глобального seed и индекса). При неудаче валидации сэмпл пересобирается с новым seed (до `max_seed_retries`, по умолчанию 50) — в результате гарантируется ровно N сэмплов без «пропусков».

Каждый сэмпл дополнительно защищён **жёстким таймаутом** (`sample_timeout`, по сложности: easy 40s / medium 75s / hard 100s / expert 150s): некоторые булевы/массивные операции OCCT недетерминированно зависают на отдельных наборах параметров. Сэмпл исполняется в отдельном воркере `multiprocessing.Pool`; если он не уложился в таймаут — воркеры принудительно убиваются, а сэмпл перезапускается со следующим seed. Сборка внутри билдера ограничена wall-clock лимитом (4×`attempt_timeout`), поэтому превышение таймаута означает зависшую OCCT-операцию. Ни один «плохой» seed не блокирует генерацию и не зацикливает её: каждый перезапуск тратит одну попытку из `max_seed_retries`.

## Установка

```bash
python -m venv venv
.\venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

CadQuery требует Windows/Linux (устанавливается как обычный wheel). Для рендера (`render.png`) также нужна рабочая среда OpenGL/EGL.

## Быстрый старт

Базовый запуск — 100 моделей в папку `output`:

```bash
python main.py --num_samples 100 --output output --seed 42
```

Через виртуальное окружение:

```bash
.\venv\Scripts\python.exe main.py --num_samples 100 --output output --seed 42
```

## Параллельная генерация

```bash
python main.py --num_samples 100 --output output --seed 42 --parallel
```

Сэмплы распределяются по процессам пула; каждому индексу гарантируется свой файл. В параллельном и последовательном режиме действует одинаковый `sample_timeout`: зависший сэмпл принудительно убивается и перезапускается со следующим seed (до `max_seed_retries`).

## Возобновление генерации (`--start-sample`)

Если генерация была прервана, её можно продолжить с нужного сэмпла, указав индекс
старта (по умолчанию 0). Работает во всех режимах вывода (только программы,
`--flat`, полный с экспортами):

```bash
python main.py --num_samples 100 --output output --seed 42 --start-sample 50
```

Генерация детерминирована по **абсолютному индексу**: seed сэмпла `i` зависит от
`--seed` и самого `i`, но не от размера чанка. Поэтому прогон 30 сэмплов за раз
даёт те же программы, что и два прогона по 15 (первый `--start-sample 0`, второй
`--start-sample 15`) при том же `--seed`. Для продолжения с места остановки просто
передайте индекс следующего сэмпла и те же `--seed` и `--num_samples`.

## Только программы, разбитые по сложности

По умолчанию генератор пишет **только исполняемые программы**, разбитые по
папкам сложности. Это позволяет сгенерировать корпус один раз, а на этапе
обучения варьировать сложность выбором подпапки:

```bash
python main.py --num_samples 100 --output out --seed 42
```

## Основной режим — только программы, разбитые по сложности из venv параллельно

```bash
.\venv\Scripts\python.exe main.py --num_samples 100 --output output --seed 42 --parallel
```

## Такой же, но с импортом stl

```bash
.\venv\Scripts\python.exe main.py --num_samples 100 --output output --seed 42 --parallel --export-formats stl
```

Результат:

```
out/
  easy/    sample_00000.py, sample_00004.py, ...
  medium/  sample_00001.py, sample_00005.py, ...
  hard/    sample_00002.py, ...
  expert/  sample_00003.py, ...
```

Никаких подпапок, `metadata.json` и экспортов (step/stl/render/…) по умолчанию
нет — только `.py`. Воспроизводимость сохраняется: сложность и seed каждого
файла детерминированы и выводятся из `--seed` и номера сэмпла.

## Дополнительные форматы (step/stl/render и др.)

Формат-экспорты включаются явным флагом `--export-formats`:

```bash
python main.py --num_samples 100 --output out --seed 42 --export-formats step,stl,render
```

В этом режиме каждый сэмпл получает подпапку внутри своей сложности:

```
out/
  easy/sample_00000/
    model.step       # STEP-экспорт
    model.stl        # STL-экспорт
    pointcloud.npy   # облако точек (4096 точек по умолчанию)
    mesh.obj         # меш
    render.png       # рендер (512x512)
    ast.json         # AST в JSON
    program.py       # программа на CadQuery (исполняемая)
    program.txt      # та же программа текстом
    metadata.json    # метаданные (операции, объём, seed, сложность)
  easy/sample_00001/
    ...
```

В полном режиме `program.py`, `metadata.json` и `ast.json` пишутся всегда.
Доступные форматы: `step, stl, pointcloud, mesh, render, ast_json, metadata_json,
program_txt`. Набор форматов и лимиты (количество операций, булевых, отверстий
и т.д.) настраиваются в `config.py` (`GeneratorConfig.save_formats`, `max_*`).

## Плоский режим

`--flat` отключает разбиение по сложности — все сэмплы пишутся прямо в `output`:

```bash
python main.py --num_samples 100 --output out --seed 42 --flat
```

## Опции командной строки

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--num_samples` | int | `100` | Количество моделей |
| `--output` | str | `output` | Папка для сохранения |
| `--seed` | int | `42` | Глобальный seed генерации |
| `--parallel` | флаг | выкл. | Параллельная генерация |
| `--export-formats` | str | `-` | Дополнительные форматы через запятую: `step,stl,pointcloud,mesh,render,ast_json,metadata_json,program_txt` |
| `--flat` | флаг | выкл. | Не разбивать на `easy/medium/hard/expert` (плоская папка) |

## Структура выходных данных (режим с экспортами)

```
output/
  easy/sample_00000/
    model.step       # STEP-экспорт
    model.stl        # STL-экспорт
    pointcloud.npy   # облако точек (4096 точек по умолчанию)
    mesh.obj         # меш
    render.png       # рендер (512x512)
    ast.json         # AST в JSON
    program.py       # программа на CadQuery (исполняемая)
    program.txt      # та же программа текстом
    metadata.json    # метаданные (операции, объём, seed, сложность)
  medium/sample_00001/
    ...
```

Набор форматов и лимиты (количество операций, булевых, отверстий и т.д.) настраиваются в `config.py` (`GeneratorConfig.save_formats`, `max_*`).

## Сложности

Каждый сэмпл получает уровень сложности по кругу из `config.complexity_levels`:

| Уровень | Диапазон операций |
|---------|-------------------|
| easy    | 3–6   |
| medium  | 7–12  |
| hard    | 13–20 |
| expert  | 20–35 |

## Тесты

```bash
.\venv\Scripts\python.exe -m pytest tests -q
```

## Основные модули

- `builder/ast_builder.py` — построение AST со случайными операциями и валидацией геометрии;
- `executor/executor.py` — исполнение AST в `cq.Shape`;
- `validators/geometry_validator.py` — проверка тела (Solid, объём, габариты, BRepCheck);
- `code_generator.py` — генерация нативной программы на CadQuery из AST;
- `dataset/dataset_builder.py` — сборка датасета (включая seed-retry, `sample_timeout` и параллельный режим);
- `grammar/state_machine.py` — порядок операций в «программе» (эскиз → выдавливание → булевы/модификаторы);
- `syntax_tree/nodes.py` — узлы AST и их регистрация (`class_map`);
- `operations/` — примитивы, эскизы, булевы операции, модификаторы (fillet, chamfer, shell, hole, split, трансформации, массивы);
- `exporters/` — STEP/STL/точки/меш/рендер.

## Поддерживаемые операции

- **Примитивы**: box, cylinder, sphere, cone, wedge, torus.
- **Эскизы**: rect, circle, ellipse, polygon, slot, polyline, spline, roundrect, frame, sector, arc_profile, ellipse_arc, bent, mirrored.
- **Объёмные из эскизов**: extrude, revolve, twist_extrude (скрученное выдавливание), loft, sweep.
- **Булевы**: union, cut, intersect.
- **Модификаторы**: fillet, chamfer, shell, hole (through / blind / cbore / csk), split (разрез со сдвигом половины).
- **Трансформации**: translate, rotate, mirror.
- **Массивы**: rarray, polarArray, scatter.

Доступность операций и лимиты на их количество задаются в `builder/ast_builder.py` и `config.py` (`max_holes`, `max_booleans`, `max_patterns`, `max_splits`, `max_twist_extrudes` и т.д.).
