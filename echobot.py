#!/usr/bin/env python
# pylint: disable=unused-argument
# This program is dedicated to the public domain under the CC0 license.

"""
Simple Bot to reply to Telegram messages.

First, a few handler functions are defined. Then, those functions are passed to
the Application and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""


from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackContext
from keras.api.preprocessing import image
from keras.api.models import load_model
import numpy as np
import os
import logging
from dotenv import load_dotenv
import cv2


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""

    await update.message.reply_text("Бот для определения наличия маски на лице")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""

    user = update.effective_user
    await update.message.reply_html(rf"Здравствуй, {user.mention_html()}! Загрузи фото с лицом в маске или без нее.",
                                    reply_markup=ForceReply(selective=True))


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""

    await update.message.reply_text("Загрузите фото с лицом в маске или без нее!")


class MaskPredictor:
    def __init__(self):
        """Bot initialisation: model and logger loading."""

        logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        self.logger = logging.getLogger(__name__)
        self.model = load_model('data/mask_detection_v2.h5')

    async def predict_mask(self, update: Update, context: CallbackContext):
        """Predict if person with mask."""

        photo = update.message.photo[-1]

        file = await context.bot.get_file(photo.file_id)
        img_path = "temp.jpg"
        await file.download_to_drive(img_path)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            await update.message.reply_text("Лицо не обнаружено!")
            os.remove(img_path)
            return

        for (x, y, w, h) in faces:
            face_img = img[y:y + h, x:x + w]

            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = image.array_to_img(face_img)
            face_img = face_img.resize((224, 224))
            img_array = image.img_to_array(face_img) / 255
            img_array = np.expand_dims(img_array, axis=0)

            predictions = self.model.predict(img_array)
            predicted_class_index = np.round(predictions).astype(int)

            label_map = {
                0: 'Маска надета',
                1: 'Маски нет'
            }

            await update.message.reply_text(f"{label_map.get(predicted_class_index.item())}")

        os.remove(img_path)

    def main(self) -> None:
        """Start the bot."""

        application = Application.builder().token(TOKEN).build()
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
        application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND & ~filters.TEXT, self.predict_mask))
        application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    load_dotenv()
    TOKEN = os.getenv("TOKEN")
    MaskPredictor().main()
