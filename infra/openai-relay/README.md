## OpenAI Relay (production quickstart)

Это минимальный прод‑сетап **OpenAI relay/proxy**:

- Клиенты ходят на ваш домен (например `openai-relay.mch.expert`) по HTTPS
- Relay требует `Authorization: Bearer <RELAY_TOKEN>` (если задан)
- Relay сам ходит в OpenAI по вашему `OPENAI_API_KEY` (ключ наружу не отдаётся)

Поддерживает **streaming** (SSE) и проксирует любые пути (`/v1/...`) к `https://api.openai.com`.

### Переменные окружения

- `OPENAI_API_KEY` (**обязательно**): ключ OpenAI для запросов *от имени сервера*
- `RELAY_TOKEN` или `OPENAI_RELAY_TOKEN` (**рекомендуется**): токен, который должны присылать клиенты
- `OPENAI_BASE_URL` (опционально, по умолчанию `https://api.openai.com`)
- `PORT` (опционально, по умолчанию `3001`)

### Прод: поднять с нуля на Ubuntu 22.04 (Docker + Nginx + TLS)

#### 0) DNS

Создайте `A` запись, например:

- `openai-relay` → `151.243.169.187`

#### 1) Установка зависимостей на VPS

```bash
ssh root@YOUR_SERVER_IP

apt-get update
apt-get install -y ca-certificates curl gnupg nginx certbot python3-certbot-nginx

# Docker
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker nginx
```

#### 2) Залить код на VPS

Вариант A (рекомендуется): скопировать только `infra/openai-relay` с локальной машины:

```bash
scp -r infra/openai-relay root@YOUR_SERVER_IP:/root/openai-relay
```

#### 3) Создать `/root/openai-relay/.env`

Важно: **не кладите весь `.env.local` целиком**. Достаточно минимального `.env`:

```bash
cd /root/openai-relay

# 1) сгенерить токен для клиентов
RELAY_TOKEN="$(openssl rand -hex 32)"

# 2) вписать ключ OpenAI (вставьте ваш)
cat > .env <<EOF
OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
RELAY_TOKEN=${RELAY_TOKEN}
EOF

chmod 600 .env
```

Если у вас ключ хранится в локальном `.env.local`, вы можете **однократно** перенести его и извлечь строку:

```bash
# на локальной машине
scp .env.local root@YOUR_SERVER_IP:/root/openai-relay/.env.local.upload

# на сервере
cd /root/openai-relay
grep -E '^OPENAI_API_KEY=' .env.local.upload >> .env
rm -f .env.local.upload
chmod 600 .env
```

#### 4) Запустить relay

```bash
cd /root/openai-relay
docker compose up -d --build
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Ports}}"
```

Relay слушает `127.0.0.1:3001` (наружу не торчит).

#### 5) Nginx reverse-proxy (80/443 → 127.0.0.1:3001)

Создайте vhost:

```bash
cat > /etc/nginx/sites-available/openai-relay.conf <<'NGINX'
server {
  server_name openai-relay.mch.expert;

  location / {
    proxy_pass http://127.0.0.1:3001;

    # Важно для uvicorn + streaming
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_buffering off;
    proxy_request_buffering off;
    proxy_read_timeout 300s;

    # полезные заголовки
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header Authorization $http_authorization;
  }
}
NGINX

ln -sf /etc/nginx/sites-available/openai-relay.conf /etc/nginx/sites-enabled/openai-relay.conf
nginx -t && systemctl reload nginx
```

#### 6) TLS сертификат (Let’s Encrypt)

```bash
certbot --nginx -d openai-relay.mch.expert --redirect --non-interactive --agree-tos -m you@example.com
```

#### 7) Быстрая проверка

```bash
curl -sS https://openai-relay.mch.expert/v1/models \
  -H "Authorization: Bearer <RELAY_TOKEN>"
```

Пример chat:

```bash
curl -sS https://openai-relay.mch.expert/v1/chat/completions \
  -H "Authorization: Bearer <RELAY_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"gpt-4o-mini",
    "messages":[{"role":"user","content":"Привет! Сколько будет 2+2?"}],
    "temperature":0
  }'
```

### Операционка: логи / рестарт / обновление

```bash
cd /root/openai-relay

# логи
docker compose logs -f --tail=200

# рестарт
docker compose restart

# после изменений кода
docker compose up -d --build
```



