from discord.ext import commands
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from app.core.ai_config import get_client, get_model
from app.core.bot import DisBot
from app.data import user_descriptions_cache
from app.tools.prompt import ROAST_PERSONAS, ROAST_PROMPT
from app.tools.utils import clean_text, replace_emojis


class Toxic(commands.Cog):
    """Команда прожарки чата."""

    def __init__(self, bot: DisBot) -> None:
        """Инициализация Cog."""
        self.bot = bot

    @commands.command(name="toxic")
    @commands.cooldown(rate=1, per=30.0, type=commands.BucketType.user)
    async def roast_command(self, ctx: commands.Context, *args: str) -> None:
        """Прожарка последних сообщений чата.

        Использование:
        !toxic [количество] [persona]
        Примеры:
        !toxic 20
        !toxic babka
        !toxic 50 babka
        !toxic list
        """
        try:
            persona = None
            limit = 20

            for arg in args:
                if arg.isdigit():
                    limit = max(1, min(int(arg), 80))
                else:
                    persona = arg

            if persona == "list":
                keys = ", ".join(f"`{k}`" for k in ROAST_PERSONAS.keys())
                await ctx.send(f"🎭 **Доступные режимы:** {keys}")
                return

            messages = []
            history_limit = limit * 2

            async for msg in ctx.channel.history(limit=history_limit):
                if len(messages) >= limit:
                    break

                if msg.author == self.bot.user:
                    continue

                content = msg.content
                if content and (
                    content.startswith(str(ctx.prefix))
                    or content.startswith(self.bot.command_prefix)
                ):
                    continue

                if not content:
                    if msg.attachments:
                        content = "[Пользователь скинул картинку/файл]"
                    elif msg.stickers:
                        content = "[Пользователь отправил стикер]"
                    else:
                        continue

                if content.startswith("http"):
                    content = "[Пользователь отправил ссылку]"

                messages.append(f"[{msg.author.name}]: {content}")

            if not messages:
                await ctx.send("Тут слишком тихо, некого прожаривать. 🦗")
                return

            messages.reverse()
            history_text = "\n".join(messages)

            descriptions = user_descriptions_cache.get_all()
            user_info_text = "\n".join(
                [f"- {k}: {v}" for k, v in descriptions.items()]
            )

            system_content = ROAST_PROMPT.format(user_info=user_info_text)

            if persona and persona in ROAST_PERSONAS:
                selected_persona = ROAST_PERSONAS[persona]
                system_content += f"\n\nВАЖНОЕ ДОПОЛНЕНИЕ К РОЛИ:\n{selected_persona}"
            elif persona:
                keys = ", ".join(f"`{k}`" for k in ROAST_PERSONAS.keys())
                await ctx.send(f"❌ Нет такого режима `{persona}`. Доступные: {keys}")
                return

            msgs = [
                ChatCompletionSystemMessageParam(role="system", content=system_content),
                ChatCompletionUserMessageParam(
                    role="user", content=f"Вот последние сообщения чата:\n{history_text}"
                ),
            ]

            async with ctx.typing():
                completion = await get_client().chat.completions.create(
                    model=get_model(),
                    messages=msgs,
                    temperature=0.9,
                    max_tokens=600,
                )
                response = completion.choices[0].message.content or ""
                cleaned_response_text = clean_text(response)
                emoji_response_text = replace_emojis(cleaned_response_text)
                await ctx.send(emoji_response_text)

        except Exception as e:
            await ctx.send(f"❌ Не удалось прожарить: {e}")


async def setup(bot: DisBot) -> None:
    """Загрузка Cog в бота."""
    await bot.add_cog(Toxic(bot))
