import os
import logging
import asyncio
import whisper
import google.generativeai as genai
from aiogram import Bot, Dispatcher, types, Router, F
from aiogram.types.input_file import FSInputFile
from pydub import AudioSegment
from gtts import gTTS

# ✅ API Tokens
TELEGRAM_BOT_TOKEN = ""
GEMINI_API_KEY = ""

# ✅ Init
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
router = Router()

# ✅ Load Whisper model
whisper_model = whisper.load_model("small")

# ✅ Gemini Init
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

@router.message(F.voice)
async def handle_voice(message: types.Message):
    try:
        # 🔽 Download voice
        file = await bot.get_file(message.voice.file_id)
        ogg_path = f"{file.file_id}.ogg"
        wav_path = f"{file.file_id}.wav"
        await bot.download_file(file.file_path, ogg_path)

        # 🔁 Convert to WAV
        audio = AudioSegment.from_file(ogg_path, format="ogg")
        audio.export(wav_path, format="wav")

        # 🧠 Transcribe with Whisper
        result = whisper_model.transcribe(wav_path)
        recognized_text = result.get("text", "").strip()

        if not recognized_text:
            await message.reply("❌ I couldn't understand that. Try again.")
            return

        logging.info(f"🎙 Recognized Text: {recognized_text}")

        # 🤖 Generate Gemini Response
        prompt = f"You are a friendly and helpful assistant. Respond in a natural, human-like tone. The user said: \"{recognized_text}\""
        response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        ai_reply = response.text.strip() if hasattr(response, "text") else "Sorry, I couldn't generate a reply."

        logging.info(f"💬 Gemini Reply: {ai_reply}")

        # 🎤 Generate speech with gTTS
        mp3_path = f"{message.message_id}_reply.mp3"
        ogg_reply_path = f"{message.message_id}_reply.ogg"
        tts = gTTS(text=ai_reply, lang='en')
        tts.save(mp3_path)

        # 🔁 Convert MP3 to OGG for Telegram
        sound = AudioSegment.from_file(mp3_path, format="mp3")
        sound.export(ogg_reply_path, format="ogg", codec="libopus")

        # 💬 Show typing/recording status
        await bot.send_chat_action(message.chat.id, action="record_voice")

        # 📨 Send voice reply
        await message.reply_voice(FSInputFile(ogg_reply_path))

        # 🧹 Cleanup
        for f in [ogg_path, wav_path, mp3_path, ogg_reply_path]:
            if os.path.exists(f):
                os.remove(f)

    except Exception as e:
        logging.error(f"❌ Error: {e}")
        await message.reply(f"⚠️ Error: {e}")

# 🟢 Main Loop
async def main():
    dp.include_router(router)
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
