# Floor-plan vectorizer

Внутренний инструмент для автоматической векторизации поэтажных планов.
На входе — изображение плана с цветной заливкой помещений.
На выходе — `lots.svg` (чистая геометрия), `lots_preview.svg` (с подложкой),
`lots.json` (метаданные) и `validation_report.json`.

Есть три способа использования:

1. **CLI** — `python vectorize_floor.py --plan ... --out-dir ...`
2. **Web UI** — FastAPI-приложение (`webapp/`), развёрнутое на Railway.
3. **Programmatic** — через класс `Pipeline` из пакета `vectorize_floor`.

## Структура

```
floor-vectorizer/
├── vectorize_floor.py          # CLI entry-point shim
├── vectorize_floor/            # core package
│   ├── cli.py
│   ├── pipeline.py
│   ├── models.py
│   ├── io_utils.py
│   ├── segmentation.py
│   ├── geometry.py
│   ├── mapping.py
│   ├── ocr.py
│   └── rendering.py
├── webapp/                     # FastAPI UI
│   ├── main.py
│   ├── templates/
│   └── static/
├── tests/
├── examples/
├── Dockerfile
├── railway.json
├── Procfile
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

## Локальный запуск

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# CLI:
python vectorize_floor.py --plan examples/plan.png --out-dir out/

# Web UI:
uvicorn webapp.main:app --reload --port 8080
# открыть http://localhost:8080
```

Tesseract нужен только при `--ocr-fallback`:

- Ubuntu/Debian: `apt-get install -y tesseract-ocr`
- macOS: `brew install tesseract`

## Railway

Репозиторий готов к one-click деплою:

1. Подключите репозиторий к Railway (`New Project → Deploy from GitHub`).
2. Railway увидит `Dockerfile` и `railway.json` и соберёт образ автоматически.
3. Переменные окружения (все опциональны):
   - `JOBS_DIR` (default: `/app/jobs`) — куда складывать результаты.
   - `MAX_UPLOAD_BYTES` (default: `31457280` = 30 MB).
   - `JOB_TTL_SECONDS` (default: `21600` = 6 часов) — TTL для авто-очистки.
   - `LOG_LEVEL` (default: `INFO`).
4. Healthcheck — `/healthz`.

Для персистентности результатов между рестартами подключите Railway Volume
и смонтируйте его в `/app/jobs` (иначе данные обнулятся при редеплое).

## CLI — полный пример

```bash
python vectorize_floor.py \
  --plan ./input/floor_2_plan.png \
  --overlay ./input/floor_2_overlay.png \
  --mapping ./input/floor_2_lots.csv \
  --floor-id floor_2 \
  --out-dir ./output/floor_2 \
  --debug
```

Пример `lots.csv`:

```csv
lot_id,label,status,area,anchor_x,anchor_y
F2-401,F2-401,available,85.4,1200,800
F2-402,F2-402,reserved,92.1,1800,800
```

## Тесты

```bash
pytest -q
```

Синтетический план в `tests/fixtures.py` прогоняется через весь пайплайн
и через HTTP-эндпоинты `webapp`.

## Расширение

- REST API / worker: логика уже изолирована в `Pipeline(cfg).run(inputs)`.
- Batch: оборачивается в `ProcessPoolExecutor` по директории `floors/<id>/`.
- Ручная валидация: поверх `lots_preview.svg` — отдельный UI для правок,
  patch-файл `lots_overrides.json` применяется после пайплайна.
