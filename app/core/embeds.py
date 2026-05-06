import asyncio
import io
from typing import Any

import aiohttp
import discord
from discord import File
from PIL import Image, ImageDraw, ImageFont

from app.data.request import get_user_rank
from app.tools.prompt import RANK_CONFIG
from app.tools.utils import darken_color, get_rank_description


def create_help_embed() -> discord.Embed:
    """Создает embed для команды !help."""
    embed = discord.Embed(
        title="📋 Справка по командам бота",
        color=discord.Color.blue(),
        description="Все доступные команды и их использование:",
    )

    embed.add_field(
        name="🎯 Основные команды",
        value=(
            "`!help` - показать эту справку\n"
            "`!rank` - узнать свой ранг и статистику\n"
            "`!rank list` - показать все возможные ранги\n"
            "`!birthday DD.MM.YYYY` - добавить/обновить дату рождения\n"
            "`!toxic` - прожарка чата (по умолчанию 20 сообщений)\n"
            "`!toxic [число]` - прожарка указанного количества сообщений\n"
            "`!toxic [образ]` - прожарка в образе (или `list` для списка)"
        ),
        inline=False,
    )

    embed.add_field(
        name="📺 Команды для YouTube",
        value=(
            "`!add_youtube [ID канала] [ID чата] [Название]` - добавить канал\n"
            "`!youtube [on/off] [Название]` - вкл/выкл отслеживание канала\n"
            "*(только для администраторов)*"
        ),
        inline=False,
    )

    embed.add_field(
        name="🌤️ Погода",
        value="`погода [город]` - текущая погода\n`погода [город] завтра` - прогноз на завтра",
        inline=False,
    )

    embed.add_field(
        name="📝 Описания пользователей",
        value=(
            "`!desc_list` - показать все описания\n"
            "`!desc_add [ник] [описание]` - добавить/обновить\n"
            "`!desc_remove [ник]` - удалить описание\n"
            "`!desc_reload` - перезагрузить из БД\n"
            "*(все команды кроме desc_list только для админов)*"
        ),
        inline=False,
    )

    embed.add_field(
        name="🛡️ Администрирование",
        value=(
            "`!reset` - очистка истории чата\n"
            "`!ai` - переключить/выбрать AI-провайдера\n"
            "`!holiday [DD.MM] [Название]` - добавить праздник\n"
            "`!check_holiday` - принудительная проверка праздников\n"
            "`!check_birthday` - принудительная проверка дней рождения\n"
            "*(только для администраторов)*"
        ),
        inline=False,
    )
    return embed


async def create_rang_embed(
    display_name: str,
    message_count: int,
    rang_description: str,
    avatar_url: str,
    server_id: int,
    user_id: int,
) -> tuple[discord.Embed, File]:
    """Создает embed для команды !rang с цветом и фоном в зависимости от ранга."""
    rank = get_rank_description(message_count)

    progress_bar = f"{message_count}/{rank['next_threshold']}"
    exp_title = "EXP"

    server_rank = await get_user_rank(user_id, server_id)

    image_buffer = await create_image_with_text_async(
        display_name,
        rang_description,
        progress_bar,
        exp_title,
        server_rank,
        rank["rank_level"],
        text_color=rank["text_color"],
        bg_filename=rank["bg_filename"],
        avatar_url=avatar_url,
    )
    file = File(image_buffer, filename="rang_with_text.png")

    embed = discord.Embed(color=rank["color"])
    embed.set_image(url="attachment://rang_with_text.png")

    return embed, file


async def download_avatar_async(avatar_url: str | None) -> Image.Image | None:
    """Асинхронная загрузка аватара пользователя."""
    if not avatar_url:
        return None

    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(avatar_url) as response:
                if response.status == 200:
                    avatar_data = await response.read()
                    avatar_img = Image.open(io.BytesIO(avatar_data)).convert("RGBA")
                    return avatar_img
    except Exception as e:
        print(f"Ошибка загрузки аватара: {e}")

    return None


async def create_image_with_text_async(
    display_name: str,
    rang_description: str,
    progress_bar: str,
    exp_title: str,
    server_rank: int,
    rank_level: int,
    text_color: tuple[int, int, int] = (44, 255, 109),
    bg_filename: str = "rang0.jpg",
    avatar_url: str | None = None,
) -> io.BytesIO:
    """Асинхронно создает изображение с текстом и аватаром пользователя."""
    avatar_img = await download_avatar_async(avatar_url)

    return await asyncio.to_thread(
        create_image_with_text,
        display_name,
        rang_description,
        progress_bar,
        exp_title,
        server_rank,
        rank_level,
        text_color,
        bg_filename,
        avatar_img,
    )


def create_rang_list_embed() -> discord.Embed:
    """Создает embed для команды !rang list."""
    embed = discord.Embed(
        title="🎖️ Система рангов",
        color=discord.Color.blurple(),
        description="Все возможные ранги и условия их получения:",
    )

    for i, rank in enumerate(RANK_CONFIG):
        if i == 0:
            value = "0 сообщений"
        elif i == len(RANK_CONFIG) - 1:
            value = f"{rank['threshold']}+ сообщений"
        else:
            next_threshold = RANK_CONFIG[i + 1]["threshold"]
            value = f"{rank['threshold']}-{next_threshold - 1} сообщений"
        embed.add_field(name=rank["name"], value=value, inline=False)

    embed.set_footer(text="Пишите сообщения, чтобы повысить свой ранг!")
    return embed


def create_image_with_text(
    display_name: str,
    rang_description: str,
    progress_bar: str,
    exp_title: str,
    server_rank: int,
    rank_level: int,
    text_color: tuple[int, int, int] = (44, 255, 109),
    bg_filename: str = "rang0.jpg",
    avatar_img: Image.Image | None = None,  # Уже загруженное изображение
) -> io.BytesIO:
    """Создает изображение с текстом и аватаром пользователя."""
    background = Image.open(f"./app/resource/{bg_filename}").convert("RGBA")
    background = background.resize((1920, 480))

    width, height = background.size
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rect_color = (30, 30, 30, 180)

    # --- цвета для разных надписей ---
    main_dark_color = darken_color(text_color, 0.75)

    # Отступы и параметры блоков
    margin_x = int(width * 0.035)
    margin_y = int(height * 0.12)
    radius = 28

    # Определяем координаты блоков
    a_left = margin_x
    a_top = margin_y
    a_right = int(width * 0.70) - margin_x // 2
    a_bottom = height - margin_y
    b_left = a_right + margin_x
    b_top = margin_y
    b_right = width - margin_x
    b_bottom = margin_y + (height - 2 * margin_y) // 2 - 5
    c_left = b_left
    c_top = b_bottom + margin_y // 2
    c_right = b_right
    c_bottom = a_bottom

    # Сначала рисуем все прямоугольники
    draw.rounded_rectangle([a_left, a_top, a_right, a_bottom], radius, fill=rect_color)
    draw.rounded_rectangle([b_left, b_top, b_right, b_bottom], radius, fill=rect_color)
    draw.rounded_rectangle([c_left, c_top, c_right, c_bottom], radius, fill=rect_color)

    # --- АВАТАР ПОЛЬЗОВАТЕЛЯ ---
    avatar_size = int((a_bottom - a_top) * 0.7)
    avatar_margin = int(avatar_size * 0.08)
    avatar_left = a_left + avatar_margin + 20
    avatar_top = a_top + ((a_bottom - a_top) - avatar_size) // 2

    # Обрабатываем аватар, если он был загружен
    if avatar_img is not None:
        try:
            # Изменяем размер аватара
            avatar_resized = avatar_img.resize((avatar_size, avatar_size))

            # Создаем круглую маску
            mask = Image.new("L", (avatar_size, avatar_size), 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.ellipse((0, 0, avatar_size, avatar_size), fill=255)

            # Применяем маску к аватару
            avatar_resized.putalpha(mask)

            # Вставляем аватар поверх прямоугольников
            overlay.paste(avatar_resized, (avatar_left, avatar_top), avatar_resized)
        except Exception as e:
            print(f"Ошибка обработки аватара: {e}")
            avatar_img = None

    # --- Шрифты ---
    try:
        main_font = ImageFont.truetype("./app/resource/montserrat.ttf", 70)
        aux_font = ImageFont.truetype("./app/resource/montserrat.ttf", 40)
        aux_value_font = ImageFont.truetype("./app/resource/montserrat.ttf", 40)
        server_rank_font = ImageFont.truetype("./app/resource/montserrat.ttf", 50)
        main_font_small = ImageFont.truetype("./app/resource/montserrat.ttf", 60)
    except Exception:
        main_font = aux_font = aux_value_font = server_rank_font = main_font_small = (
            ImageFont.load_default()
        )

    def draw_centered_text_block(
        texts_fonts_colors: list[tuple[str, Any, tuple[int, int, int]]],
        center_x: int,
        center_y: int,
        gapp: int = 10,
    ) -> None:
        """Отрисовка блока текста с вертикальным выравниванием по центру."""
        heights = []
        for text, font, _ in texts_fonts_colors:
            bbox = draw.textbbox((0, 0), text, font=font)
            heights.append(bbox[3] - bbox[1])

        total_heights = sum(heights) + gapp * (len(heights) - 1)
        current_y = center_y - total_heights // 2

        for (text, font, color), text_height in zip(texts_fonts_colors, heights):
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            draw.text((center_x - text_width // 2, current_y), text, font=font, fill=color)
            current_y += text_height + gapp

    # ------ ВЫРАВНИВАНИЕ A ------
    a_text_left = (
        avatar_left + avatar_size + avatar_margin + 20 if avatar_img is not None else a_left + 10
    )
    a_cy = a_top + (a_bottom - a_top) // 2

    server_rank_text = f"Server rank #{server_rank}"

    # Размеры текстов
    dn_bbox = draw.textbbox((0, 0), display_name, font=main_font)
    dn_height = dn_bbox[3] - dn_bbox[1]

    rd_bbox = draw.textbbox((0, 0), rang_description, font=main_font)
    rd_height = rd_bbox[3] - rd_bbox[1]

    sr_bbox = draw.textbbox((0, 0), server_rank_text, font=server_rank_font)
    sr_height = sr_bbox[3] - sr_bbox[1]

    # Распределение по вертикали
    if len(display_name) < 20:
        gap = 25
    elif len(display_name) >= 28:
        gap = 5
    else:
        gap = 15
    total_height = dn_height + rd_height + sr_height + 2 * gap
    top_block = a_cy - total_height // 2

    # Отрисовка текстов
    if len(display_name) < 20:
        draw.text((a_text_left, top_block), display_name, font=main_font, fill=main_dark_color)
    elif len(display_name) >= 28:
        draw.text(
            (a_text_left, top_block), display_name, font=server_rank_font, fill=main_dark_color
        )
    else:
        draw.text((a_text_left, top_block), display_name, font=main_font_small, fill=main_dark_color)
    draw.text(
        (a_text_left, top_block + dn_height + gap), rang_description, font=main_font, fill=text_color
    )
    draw.text(
        (a_text_left, top_block + dn_height + gap + rd_height + gap),
        server_rank_text,
        font=server_rank_font,
        fill=main_dark_color,
    )

    # ------ ВЫРАВНИВАНИЕ B ------
    b_cx = b_left + (b_right - b_left) // 2
    b_cy = b_top + (b_bottom - b_top) // 2

    draw_centered_text_block(
        [("LEVEL", aux_font, main_dark_color), (str(rank_level), aux_value_font, text_color)],
        b_cx,
        b_cy,
        gapp=20,
    )

    # ------ ВЫРАВНИВАНИЕ C ------
    c_cx = c_left + (c_right - c_left) // 2
    c_cy = c_top + (c_bottom - c_top) // 2

    draw_centered_text_block(
        [(exp_title, aux_font, main_dark_color), (progress_bar, aux_value_font, text_color)],
        c_cx,
        c_cy,
        gapp=20,
    )

    # Собираем картинку
    background = Image.alpha_composite(background, overlay)
    img_buffer = io.BytesIO()
    background.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    return img_buffer
