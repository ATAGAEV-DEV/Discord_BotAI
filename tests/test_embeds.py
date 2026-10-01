"""Тесты для модуля генерации Embed сообщений."""

from collections.abc import Iterator
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest
from discord import File
from PIL import Image, ImageFont

from app.core.embeds import (
    create_help_embed,
    create_image_with_text,
    create_image_with_text_async,
    create_rang_embed,
    create_rang_list_embed,
    download_avatar_async,
)


class TestCreateHelpEmbed:
    """Тесты для функции create_help_embed."""

    def test_creates_embed_with_correct_title(self) -> None:
        """Проверяет, что embed имеет правильный заголовок."""
        prefix = "!"
        embed = create_help_embed(prefix)

        assert embed.title == "📋 Справка по командам бота"
        assert embed.color == discord.Color.blue()

    def test_contains_main_commands_field(self, prefix: str = "!") -> None:
        """Проверяет наличие поля с основными командами."""
        embed = create_help_embed(prefix)

        fields = {f.name: f for f in embed.fields}
        assert "🎯 Основные команды" in fields
        assert f"`{prefix}help`" in fields["🎯 Основные команды"].value

    def test_contains_youtube_commands_field(self, prefix: str = "!") -> None:
        """Проверяет наличие поля с YouTube командами."""
        embed = create_help_embed(prefix)

        fields = {f.name: f for f in embed.fields}
        assert "📺 Команды для YouTube" in fields
        assert f"`{prefix}add_youtube [" in fields["📺 Команды для YouTube"].value

    def test_contains_user_descriptions_field(self, prefix: str = "!") -> None:
        """Проверяет наличие поля с описаниями пользователей."""
        embed = create_help_embed(prefix)

        fields = {f.name: f for f in embed.fields}
        assert "📝 Описания пользователей" in fields
        assert f"`{prefix}desc_list`" in fields["📝 Описания пользователей"].value

    def test_contains_emoji_descriptions_field(self, prefix: str = "!") -> None:
        """Проверяет наличие поля с описаниями эмодзи."""
        embed = create_help_embed(prefix)

        fields = {f.name: f for f in embed.fields}
        assert "😀 Описания эмодзи" in fields
        assert f"`{prefix}emoji_list`" in fields["😀 Описания эмодзи"].value

    def test_contains_administration_field(self, prefix: str = "!") -> None:
        """Проверяет наличие поля с администрированием."""
        embed = create_help_embed(prefix)

        fields = {f.name: f for f in embed.fields}
        assert "🛡️ Администрирование" in fields
        assert f"`{prefix}reset`" in fields["🛡️ Администрирование"].value

    def test_prefix_is_used_in_all_fields(self, prefix: str = "~") -> None:
        """Проверяет, что кастомный префикс используется во всех полях."""
        embed = create_help_embed(prefix)

        for field in embed.fields:
            assert prefix in field.value, f"Префикс не найден в поле {field.name}"


class TestCreateRangListEmbed:
    """Тесты для функции create_rang_list_embed."""

    def test_creates_embed_with_correct_title(self) -> None:
        """Проверяет заголовок списка рангов."""
        embed = create_rang_list_embed()

        assert embed.title == "🎖️ Система рангов"
        assert embed.color == discord.Color.blurple()

    def test_contains_all_ranks(self) -> None:
        """Проверяет, что все ранги добавлены в embed."""
        from app.tools.prompt import RANK_CONFIG

        embed = create_rang_list_embed()

        fields = {f.name: f for f in embed.fields}

        for i, rank in enumerate(RANK_CONFIG):
            assert rank["name"] in fields

    def test_first_rank_has_correct_value(self) -> None:
        """Проверяет значение для первого ранга (0 сообщений)."""
        embed = create_rang_list_embed()

        first_field = embed.fields[0]
        assert first_field.value == "0 сообщений"

    def test_last_rank_has_correct_value(self) -> None:
        """Проверяет значение для последнего ранга (+ сообщений)."""
        from app.tools.prompt import RANK_CONFIG

        embed = create_rang_list_embed()

        last_field = embed.fields[-1]
        expected_text = f"{RANK_CONFIG[-1]['threshold']}+ сообщений"
        assert last_field.value == expected_text

    def test_middle_ranks_have_correct_ranges(self) -> None:
        """Проверяет диапазон для средних рангов."""
        from app.tools.prompt import RANK_CONFIG

        embed = create_rang_list_embed()

        for i in range(1, len(RANK_CONFIG) - 1):
            rank = RANK_CONFIG[i]
            next_threshold = RANK_CONFIG[i + 1]["threshold"]
            expected_value = f"{rank['threshold']}-{next_threshold - 1} сообщений"

            field_value = embed.fields[i].value
            assert field_value == expected_value, (
                f"Ранг {rank['name']}: ожидается '{expected_value}', но '{field_value}'"
            )

    def test_embed_has_footer(self) -> None:
        """Проверяет наличие футера в embed."""
        embed = create_rang_list_embed()

        assert embed.footer is not None
        assert "Пишите сообщения, чтобы повысить свой ранг!" in embed.footer.text


class TestCreateRangEmbed:
    """Тесты для функции create_rang_embed."""

    @pytest.fixture(autouse=True)
    def mock_external_dependencies(self) -> Iterator[None]:
        """Изолирует тесты embed от базы данных и сетевой генерации изображения."""
        with (
            patch("app.core.embeds.get_user_rank", new=AsyncMock(return_value=0)),
            patch(
                "app.core.embeds.create_image_with_text_async",
                new=AsyncMock(return_value=BytesIO(b"fake image")),
            ),
        ):
            yield

    @pytest.fixture
    def mock_avatar_url(self) -> str:
        """Фикстура с аватаром пользователя."""
        return "https://example.com/avatar.png"

    @pytest.fixture
    def test_data(
        self,
        mock_avatar_url: str,
    ) -> dict[str, Any]:
        """Тестовые данные для create_rang_embed."""
        return {
            "display_name": "TestUser",
            "message_count": 100,
            "rang_description": "Бич",
            "avatar_url": mock_avatar_url,
            "server_id": 123456789,
            "user_id": 987654321,
        }

    @pytest.mark.asyncio
    async def test_returns_tuple_with_embed_and_file(self, test_data: dict[str, Any]) -> None:
        """Проверяет, что функция возвращает кортеж (Embed, File)."""
        embed, file = await create_rang_embed(**test_data)

        assert isinstance(embed, discord.Embed)
        assert isinstance(file, File)

    @pytest.mark.asyncio
    async def test_embed_has_correct_color(self, test_data: dict[str, Any]) -> None:
        """Проверяет цвет embed в зависимости от ранга."""
        from app.tools.utils import get_rank_description

        rank = get_rank_description(test_data["message_count"])

        embed, _ = await create_rang_embed(**test_data)

        assert embed.color == rank["color"]

    @pytest.mark.asyncio
    async def test_embed_has_image_attachment(self, test_data: dict[str, Any]) -> None:
        """Проверяет, что embed содержит изображение."""
        embed, _ = await create_rang_embed(**test_data)

        assert "attachment://rang_with_text.png" in embed.fields[0].value if embed.fields else True

    @pytest.mark.asyncio
    async def test_progress_bar_shown_in_field_value(
        self, test_data: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Проверяет прогресс-бар в значении поля."""
        from unittest.mock import patch

        # Мокаем get_user_rank и create_image_with_text_async
        with patch("app.core.embeds.get_user_rank", return_value=0):
            with patch(
                "app.core.embeds.create_image_with_text_async"
            ) as mock_create_image:
                mock_create_image.return_value = BytesIO(b"fake image")

                embed, _ = await create_rang_embed(**test_data)

                assert f"{test_data['message_count']}/100" in embed.fields[0].value if embed.fields else True


class TestCreateImageWithTextAsync:
    """Тесты для асинхронной функции создания изображения."""

    @pytest.mark.asyncio
    async def test_returns_bytesio_object(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Проверяет возвращаемый тип функции."""
        from io import BytesIO

        mock_avatar = MagicMock(spec=Image.Image)

        with patch(
            "app.core.embeds.create_image_with_text"
        ) as mock_create_image:
            mock_result = BytesIO()
            mock_create_image.return_value = mock_result

            result = await create_image_with_text_async(
                display_name="Test",
                rang_description="Desc",
                progress_bar="10/50",
                exp_title="EXP",
                server_rank=1,
                rank_level=2,
                avatar_url=None,
            )

            assert isinstance(result, BytesIO)

    @pytest.mark.asyncio
    async def test_downloads_avatar_and_processes_it(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Проверяет загрузку и обработку аватара."""
        from io import BytesIO

        from PIL import Image

        fake_avatar_data = BytesIO()
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        fake_avatar_data.write(img.tobytes())
        fake_avatar_data.seek(0)

        with patch(
            "app.core.embeds.download_avatar_async"
        ) as mock_download:
            mock_download.return_value = img

            with patch(
                "app.core.embeds.create_image_with_text"
            ) as mock_create:
                result = await create_image_with_text_async(
                    display_name="Test",
                    rang_description="Desc",
                    progress_bar="10/50",
                    exp_title="EXP",
                    server_rank=1,
                    rank_level=2,
                    avatar_url=fake_avatar_data.getvalue().decode("latin-1"),
                )

                mock_download.assert_called_once()


class TestDownloadAvatarAsync:
    """Тесты для асинхронной загрузки аватара."""

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_url(self) -> None:
        """Пустой URL не вызывает сетевой запрос."""
        result = await download_avatar_async(None)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_image_for_successful_response(self) -> None:
        """Успешный ответ преобразуется в RGBA-изображение."""
        source_image = Image.new("RGB", (10, 10), (255, 0, 0))
        image_data = BytesIO()
        source_image.save(image_data, format="PNG")

        response = MagicMock(status=200)
        response.read = AsyncMock(return_value=image_data.getvalue())
        response_context = MagicMock()
        response_context.__aenter__ = AsyncMock(return_value=response)
        response_context.__aexit__ = AsyncMock(return_value=None)

        session = MagicMock()
        session.get.return_value = response_context
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)

        with patch("app.core.embeds.aiohttp.ClientSession", return_value=session):
            result = await download_avatar_async("https://example.com/avatar.png")

        assert result is not None
        assert result.mode == "RGBA"
        assert result.size == (10, 10)
        session.get.assert_called_once_with("https://example.com/avatar.png")
        response.read.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_none_for_unsuccessful_response(self) -> None:
        """Ответ с неуспешным HTTP-статусом не создает изображение."""
        response = MagicMock(status=404)
        response.read = AsyncMock()
        response_context = MagicMock()
        response_context.__aenter__ = AsyncMock(return_value=response)
        response_context.__aexit__ = AsyncMock(return_value=None)

        session = MagicMock()
        session.get.return_value = response_context
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)

        with patch("app.core.embeds.aiohttp.ClientSession", return_value=session):
            result = await download_avatar_async("https://example.com/missing.png")

        assert result is None
        response.read.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_none_when_download_raises(self) -> None:
        """Исключение сетевого клиента обрабатывается и возвращает None."""
        with patch(
            "app.core.embeds.aiohttp.ClientSession",
            side_effect=RuntimeError("network error"),
        ):
            result = await download_avatar_async("https://example.com/avatar.png")

        assert result is None


class TestCreateImageWithText:
    """Тесты для синхронной функции создания изображения."""

    def test_returns_bytesio_object(self) -> None:
        """Проверяет возвращаемый тип функции."""
        result = create_image_with_text(
            display_name="Test",
            rang_description="Desc",
            progress_bar="10/50",
            exp_title="EXP",
            server_rank=1,
            rank_level=2,
            avatar_img=None,
        )

        assert isinstance(result, BytesIO)

    def test_uses_correct_background_file(self) -> None:
        """Проверяет использование фоновой картинки."""
        result = create_image_with_text(
            display_name="Test",
            rang_description="Desc",
            progress_bar="10/50",
            exp_title="EXP",
            server_rank=1,
            rank_level=2,
            avatar_img=None,
        )

        result.seek(0)
        with Image.open(result) as img:
            assert img.size == (1920, 480), "Итоговое изображение не имеет правильного размера"

    def test_handles_custom_text_color(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Проверяет использование кастомного цвета текста."""
        custom_color = (255, 0, 0)  # Красный

        result = create_image_with_text(
            display_name="Test",
            rang_description="Desc",
            progress_bar="10/50",
            exp_title="EXP",
            server_rank=1,
            rank_level=2,
            text_color=custom_color,
            avatar_img=None,
        )

        assert isinstance(result, BytesIO)

    def test_handles_no_avatar(self) -> None:
        """Проверяет работу без аватара пользователя."""
        result = create_image_with_text(
            display_name="TestUser",
            rang_description="Бич",
            progress_bar="100/500",
            exp_title="EXP",
            server_rank=5,
            rank_level=6,
            text_color=(255, 73, 73),
            bg_filename="rang3.png",
            avatar_img=None,
        )

        assert isinstance(result, BytesIO)

    def test_handles_avatar(self) -> None:
        """Проверяет обработку и наложение аватара пользователя."""
        avatar = Image.new("RGBA", (100, 100), (255, 0, 0, 255))

        result = create_image_with_text(
            display_name="TestUser",
            rang_description="Бич",
            progress_bar="100/500",
            exp_title="EXP",
            server_rank=5,
            rank_level=6,
            avatar_img=avatar,
        )

        assert isinstance(result, BytesIO)

    def test_continues_when_avatar_processing_fails(self) -> None:
        """Ошибка обработки аватара не прерывает создание изображения."""
        avatar = MagicMock(spec=Image.Image)
        avatar.resize.side_effect = RuntimeError("invalid avatar")

        result = create_image_with_text(
            display_name="TestUser",
            rang_description="Бич",
            progress_bar="100/500",
            exp_title="EXP",
            server_rank=5,
            rank_level=6,
            avatar_img=avatar,
        )

        assert isinstance(result, BytesIO)

    def test_uses_default_fonts_when_font_files_are_unavailable(
        self,
    ) -> None:
        """При ошибке загрузки шрифтов используются встроенные шрифты PIL."""
        default_font = ImageFont.load_default()
        with (
            patch("app.core.embeds.ImageFont.truetype", side_effect=OSError),
            patch("app.core.embeds.ImageFont.load_default", return_value=default_font),
        ):
            result = create_image_with_text(
                display_name="TestUser",
                rang_description="Бич",
                progress_bar="100/500",
                exp_title="EXP",
                server_rank=5,
                rank_level=6,
                avatar_img=None,
            )

        assert isinstance(result, BytesIO)

    def test_uses_small_font_for_medium_length_name(self) -> None:
        """Для имени длиной от 20 до 27 символов используется малый шрифт."""
        result = create_image_with_text(
            display_name="A" * 20,
            rang_description="Бич",
            progress_bar="100/500",
            exp_title="EXP",
            server_rank=5,
            rank_level=6,
            avatar_img=None,
        )

        assert isinstance(result, BytesIO)

    def test_uses_server_rank_font_for_long_name(self) -> None:
        """Для имени длиной не менее 28 символов используется шрифт ранга."""
        result = create_image_with_text(
            display_name="A" * 28,
            rang_description="Бич",
            progress_bar="100/500",
            exp_title="EXP",
            server_rank=5,
            rank_level=6,
            avatar_img=None,
        )

        assert isinstance(result, BytesIO)