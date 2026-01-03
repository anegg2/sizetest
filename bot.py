import asyncio
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message, InputFile
from aiogram.utils.keyboard import InlineKeyboardBuilder

from config import BOT_TOKEN
from sizing import estimate_measurements, recommend_yasneg, format_size_display

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

MEDIA_DIR = Path("media")
MEDIA_DIR.mkdir(exist_ok=True)

# Простое "состояние" в памяти
user_photo_paths: dict[int, Path] = {}
user_waits_height: set[int] = set()


@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "Привет! Пришли фото в полный рост (человек целиком спереди), "
        "а затем напиши рост в сантиметрах, например: 175."
    )


@dp.message(F.photo)
async def handle_photo(message: Message):
    user_id = message.from_user.id

    # Берём самое большое фото
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)

    photo_path = MEDIA_DIR / f"user_{user_id}.jpg"
    await bot.download_file(file.file_path, destination=photo_path)

    user_photo_paths[user_id] = photo_path
    user_waits_height.add(user_id)

    await message.answer(
        "Фото получил ✅\nТеперь напиши свой рост в сантиметрах, например: 175."
    )


@dp.message(F.text)
async def handle_height(message: Message):
    user_id = message.from_user.id

    if user_id not in user_waits_height:
        await message.answer(
            "Сначала пришли фото в полный рост, потом напиши рост в сантиметрах."
        )
        return

    text = message.text.strip().lower()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        await message.answer("Не нашёл число в сообщении. Напиши только рост, например: 175.")
        return

    height_cm = int(digits)
    if not (130 <= height_cm <= 220):
        await message.answer("Рост выглядит странно. Напиши рост в диапазоне 130–220 см.")
        return

    photo_path = user_photo_paths.get(user_id)
    if not photo_path or not photo_path.exists():
        await message.answer("Не могу найти твоё фото, пришли его ещё раз.")
        user_waits_height.discard(user_id)
        return

    await message.answer("Секунду, подбираю размер…")

    try:
        meas = estimate_measurements(photo_path, height_cm)
    except Exception as e:
        await message.answer(f"Не удалось распознать позу: {e}")
        user_waits_height.discard(user_id)
        return

    raw_size = recommend_yasneg(meas, height_cm)
    size_text = format_size_display(raw_size)

    reply_text = (
        f"Рост: {height_cm} см\n"
        f"Пояс (D) ≈ {2 * meas.waist_girth:.1f} см\n"
        f"Бёдра (E) ≈ {2 * meas.hip_girth:.1f} см\n"
        f"Длина брюк (F) ≈ {meas.pants_length:.1f} см\n\n"
        f"Рекомендуемый размер: <b>{size_text}</b>"
    )

    # Кнопка "Купить" с ссылкой (пока тестовая ya.ru)
    kb = InlineKeyboardBuilder()
    kb.button(text=f"Купить {size_text}", url="https://yasneg.ru/")
    kb.adjust(1)

    await message.answer(reply_text, parse_mode="HTML", reply_markup=kb.as_markup())

    # Фото с точками
    if meas.debug_path and meas.debug_path.exists():
        await message.answer_photo(InputFile(meas.debug_path), caption="Точки позы")

    user_waits_height.discard(user_id)


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
