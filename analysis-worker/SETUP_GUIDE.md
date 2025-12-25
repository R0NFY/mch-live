# Настройка Yandex Serverless Container с триггером на Object Storage

## Обзор архитектуры

```
📁 Object Storage
   └── recordings/video.mp4  ──┐
                               │ Триггер (create-object)
                               ▼
   🐳 Serverless Container
   └── handler.py обрабатывает файл
       ├── Скачивает video.mp4
       ├── Находит transcript.json
       ├── Запускает OceanAI
       ├── Запускает ChatGPT
       └── Сохраняет report.json
                               │
   📁 Object Storage           │
   └── reports/report.json  ◄──┘
```

**Оплата**: Только за время работы контейнера (~5-15 мин на видео).

---

## Шаг 1: Создание Container Registry

```bash
# Установка Yandex CLI (если не установлен)
curl https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash

# Авторизация
yc init

# Создание реестра контейнеров
yc container registry create --name analysis-registry
```

Запомните ID реестра (например: `crp1234abcd5678efgh`)

---

## Шаг 2: Сборка и загрузка Docker-образа

```bash
cd analysis-worker

# Авторизация в Container Registry
yc container registry configure-docker

# Сборка образа
docker build -t cr.yandex/crpv9gnnri1vqg1cof2b/analysis-worker:latest .

# Загрузка в реестр
docker push cr.yandex/crpv9gnnri1vqg1cof2b/analysis-worker:latest
```

---

## Шаг 3: Создание сервисного аккаунта

```bash
# Создание сервисного аккаунта
yc iam service-account create --name analysis-sa

# Получение ID
SA_ID=$(yc iam service-account get analysis-sa --format json | jq -r .id)

# Назначение ролей
yc resource-manager folder add-access-binding <FOLDER_ID> \
  --role storage.viewer \
  --subject serviceAccount:$SA_ID

yc resource-manager folder add-access-binding <FOLDER_ID> \
  --role storage.uploader \
  --subject serviceAccount:$SA_ID

yc resource-manager folder add-access-binding <FOLDER_ID> \
  --role serverless.containers.invoker \
  --subject serviceAccount:$SA_ID
```

---

## Шаг 4: Создание Serverless Container

### Через консоль (проще):
1. Откройте [console.yandex.cloud](https://console.yandex.cloud)
2. **Serverless Containers** → **Создать контейнер**
3. Настройки:
   - **Имя**: `analysis-worker`
   - **Образ**: `cr.yandex/crpv9gnnri1vqg1cof2b/analysis-worker:latest`
   - **Сервисный аккаунт**: `analysis-sa`
   - **Память**: `8 GB` (максимум для OceanAI)
   - **Таймаут**: `3600 сек` (1 час)
   - **Переменные окружения**:
     - `S3_ENDPOINT` = `https://storage.yandexcloud.net`
     - `S3_BUCKET` = `ваш-бакет`
     - `S3_ACCESS_KEY` = `ваш-ключ`
     - `S3_SECRET` = `ваш-секрет`
     - **Вариант (по умолчанию): через наш relay/proxy**:
       - `OPENAI_BASE_URL` = `https://openai-relay.mch.expert` (можно без `/v1`, код добавит сам)
       - `RELAY_TOKEN` = `<ваш relay токен>`
     - **Fallback: напрямую в OpenAI**:
       - `OPENAI_API_KEY` = `sk-...`
4. **Создать**

### Через CLI:
```bash
yc serverless container create \
  --name analysis-worker \
  --memory 8g \
  --execution-timeout 3600s \
  --service-account-id $SA_ID

yc serverless container revision deploy \
  --container-name analysis-worker \
  --image cr.yandex/crpv9gnnri1vqg1cof2b/analysis-worker:latest \
  --service-account-id $SA_ID \
  --environment S3_BUCKET=ваш-бакет \
  --environment S3_ACCESS_KEY=ключ \
  --environment S3_SECRET=секрет \
  # Вариант (по умолчанию): relay
  --environment OPENAI_BASE_URL=https://openai-relay.mch.expert \
  --environment RELAY_TOKEN=<token>
  # Fallback: direct
  # --environment OPENAI_API_KEY=sk-...
```

---

## Шаг 5: Создание триггера на Object Storage

### Через консоль:
1. **Serverless Containers** → **Триггеры** → **Создать триггер**
2. Настройки:
   - **Тип**: Object Storage
   - **Бакет**: выберите ваш бакет
   - **Типы событий**: ✓ Создание объекта
   - **Префикс**: `recordings/`
   - **Суффикс**: `.mp4`
   - **Контейнер**: `analysis-worker`
   - **Сервисный аккаунт**: `analysis-sa`
3. **Создать**

### Через CLI:
```bash
yc serverless trigger create object-storage \
  --name analysis-trigger \
  --bucket-id <BUCKET_ID> \
  --events create-object \
  --prefix "recordings/" \
  --suffix ".mp4" \
  --invoke-container-name analysis-worker \
  --invoke-container-service-account-id $SA_ID
```

---

## Проверка работы

1. Загрузите тестовое видео в `recordings/`:
   ```bash
   aws s3 cp test.mp4 s3://ваш-бакет/recordings/test.mp4 \
     --endpoint-url https://storage.yandexcloud.net
   ```

2. Проверьте логи контейнера в консоли Yandex Cloud

3. Через 5-15 минут в `reports/` появится файл `test_report.json`

---

## Стоимость

| Ресурс | Расход | Примерная цена |
|--------|--------|----------------|
| Serverless Container | ~10 мин на видео | ~5-10 ₽ за вызов |
| Object Storage | Хранение файлов | ~2 ₽/ГБ/месяц |
| Триггеры | Бесплатно | 0 ₽ |

**Итого**: ~10 ₽ за обработку одного звонка (без стоимости OpenAI API)

---

## Возможные проблемы

### Ошибка "Out of memory"
→ Увеличьте память контейнера до 8 GB

### Таймаут
→ Увеличьте таймаут до 3600 сек

### OceanAI не загружает модели
→ При первом запуске модели скачиваются (~2GB). 
   Рекомендуется предварительно скачать и включить в образ.
