import os


# дефолт на случай, если переменная не задана
_DEFAULT_TOKEN = "8207572887:AAGOrCuGWxGMOOEUy_RlMlMR53FaQLbZT5Y"

BOT_TOKEN = os.getenv("BOT_TOKEN", _DEFAULT_TOKEN)