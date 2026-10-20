# 🤖 Discord Bot AI

Discord-бот для общения с ИИ на вашем сервере. Бот отвечает на обращения пользователей, учитывает контекст разговора и сохраняет данные в PostgreSQL.

Также доступны память о сервере, система рангов и развлекательные возможности. Для работы нужны Discord, PostgreSQL и доступ к AI API.

## Что понадобится

- Docker
- PostgreSQL — локальная или удалённая база данных
- Discord-бот, добавленный на ваш сервер
- Доступ к AI API с интерфейсом, совместимым с OpenAI API

## 1. Подготовьте Discord-бота

Для работы нужен Discord-бот, добавленный на ваш сервер. Токен бота указывается в `.env` в переменной `DC_TOKEN`.

## 2. Скачайте проект

Откройте терминал и выполните:

```bash
git clone https://github.com/ATAGAEV95/Discord_BotAI.git
cd Discord_BotAI
cp .env.example .env
```

## 3. Подготовьте PostgreSQL

Нужна доступная база PostgreSQL — локальная или удалённая. Укажите строку подключения в `.env` в переменной `DATABASE_URL`. Таблицы будут созданы автоматически при первом запуске.

## 4. Заполните `.env`

Заполните переменные окружения в файле `.env`. В файле `.env.example` есть подсказки для настройки Discord, PostgreSQL, AI API и embeddings.

Не публикуйте `.env`, токен Discord и ключи AI API.

## 5. Соберите и запустите бота

Выполните команды из папки проекта:

```bash
docker build -t discord-bot-ai .
docker run -d --restart=always --name discord-bot-ai --env-file .env discord-bot-ai
```

После запуска бот подключится к Discord.

Остановить контейнер можно командой:

```bash
docker stop discord-bot-ai
```