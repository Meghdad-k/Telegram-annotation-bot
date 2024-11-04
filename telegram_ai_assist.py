from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
import cv2
import numpy as np
from PIL import Image
import io
from deep_translator import GoogleTranslator
from langdetect import detect
import requests
import base64
import json
import asyncio
import mimetypes

# Replace with your Telegram token
TELEGRAM_TOKEN = "8193027587:AAGGsSwMha_kMNwJ8d7Y65056wJBJe__YcU"

# Persian messages
PROMPT_MESSAGE = ("لطفاً یک عکس برای من ارسال کنید تا گزینه‌های زیر را در اختیار شما قرار دهم:\n"
                 "الف) پیشنهاد کپشن اینستاگرام برای عکس\n"
                 "ب) اعمال فیلتر زیبایی روی عکس\n\n"
                 "همچنین می‌توانید با من چت کنید! کافیست پیام خود را بنویسید.")

WELCOME_MESSAGE = f"به ربات پردازش عکس خوش آمدید! 👋\n\n{PROMPT_MESSAGE}"

GENERATING_CAPTION = "در حال تولید و ترجمه کپشن... لطفاً صبر کنید."
FILTER_APPLIED = "فیلتر با موفقیت اعمال شد!"
FILTER_CAPTION = "عکس شما با فیلتر زیبایی 🌟"
WHAT_TO_DO = "مایل به انجام چه کاری با این عکس هستید؟"
NO_PHOTO = "لطفاً ابتدا یک عکس ارسال کنید!"
THINKING = "در حال فکر کردن... 🤔"
HISTORY_CLEARED = "تاریخچه گفتگو پاک شد! 🗑️"
INVALID_FILE = "فایل ارسالی معتبر نیست. لطفاً یک عکس ارسال کنید!"

class InstagramCaptionBot:
    def __init__(self, telegram_token):
        self.telegram_token = telegram_token
        self.translator = GoogleTranslator(source='en', target='fa')
        
        # Initialize Telegram application
        self.application = Application.builder().token(telegram_token).build()
        self.setup_handlers()
        
        # Ollama API endpoint
        self.ollama_endpoint = "http://localhost:11434/api/generate"
        
        # Maximum conversation history to maintain
        self.max_history = 10

    def setup_handlers(self):
        """Configure bot command and message handlers"""
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("clear", self.clear_history))
        self.application.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        # Add handler for document messages
        self.application.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_chat))
        self.application.add_handler(CallbackQueryHandler(self.button_callback))

    def get_clear_button_markup(self):
        """Create inline keyboard with clear history button"""
        keyboard = [[InlineKeyboardButton("پاک کردن تاریخچه 🗑️", callback_data='clear_history')]]
        return InlineKeyboardMarkup(keyboard)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Welcome message handler"""
        if 'chat_history' not in context.user_data:
            context.user_data['chat_history'] = []
        await update.message.reply_text(WELCOME_MESSAGE)

    async def clear_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Clear chat history"""
        context.user_data['chat_history'] = []
        await update.message.reply_text(HISTORY_CLEARED)

    async def handle_chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages for chat with streaming response"""
        if 'chat_history' not in context.user_data:
            context.user_data['chat_history'] = []
            
        user_message = update.message.text
        thinking_message = await update.message.reply_text(THINKING)
        
        context.user_data['chat_history'].append({"role": "user", "content": user_message})
        
        # Keep only last N messages
        if len(context.user_data['chat_history']) > self.max_history:
            context.user_data['chat_history'] = context.user_data['chat_history'][-self.max_history:]
        
        try:
            # System prompt in Persian
            system_prompt = """تو یک دستیار مفید و دانش‌آموز هستی که پاسخ‌های واضح، دقیق و دوستانه ارائه می‌دهد."""
            
            # Format the conversation history
            formatted_prompt = "\n".join([
                f"System: {system_prompt}",
                *[f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" 
                for msg in context.user_data['chat_history']]
            ])
            
            # Make streaming request to Ollama
            response = requests.post(
                self.ollama_endpoint,
                json={
                    "model": "aya:35b",
                    "prompt": formatted_prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.6,
                        "top_p": 0.9,
                    }
                },
                stream=True
            )
            
            # Initialize variables for streaming
            full_response = ""
            last_update_time = asyncio.get_event_loop().time()
            update_interval = 0.5  # Update message every 0.5 seconds
            last_message_content = THINKING  # Track last message content
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            json_response = json.loads(line)
                            chunk = json_response.get('response', '')
                            full_response += chunk
                            
                            # Update the message periodically and only if content changed
                            current_time = asyncio.get_event_loop().time()
                            current_content = full_response.strip()
                            
                            if (current_time - last_update_time >= update_interval and 
                                current_content != last_message_content and 
                                current_content):  # Ensure content isn't empty
                                try:
                                    await thinking_message.edit_text(
                                        current_content,
                                        reply_markup=self.get_clear_button_markup()
                                    )
                                    last_message_content = current_content
                                    last_update_time = current_time
                                except Exception as edit_error:
                                    print(f"Edit error: {str(edit_error)}")
                                    continue
                                
                        except json.JSONDecodeError:
                            continue
                
                # Final update with complete response, only if different from last update
                final_content = full_response.strip()
                if final_content and final_content != last_message_content:
                    try:
                        await thinking_message.edit_text(
                            final_content,
                            reply_markup=self.get_clear_button_markup()
                        )
                        # Add response to chat history
                        context.user_data['chat_history'].append({
                            "role": "assistant",
                            "content": final_content
                        })
                    except Exception as final_edit_error:
                        print(f"Final edit error: {str(final_edit_error)}")
                
            else:
                error_msg = "متأسفانه در پردازش پیام شما مشکلی پیش آمد. لطفاً دوباره تلاش کنید."
                if error_msg != last_message_content:
                    await thinking_message.edit_text(
                        error_msg,
                        reply_markup=self.get_clear_button_markup()
                    )
                
        except Exception as e:
            print(f"Error in processing: {str(e)}")
            error_msg = "متأسفانه در پردازش پیام شما مشکلی پیش آمد. لطفاً دوباره تلاش کنید."
            if error_msg != last_message_content:
                await thinking_message.edit_text(
                    error_msg,
                    reply_markup=self.get_clear_button_markup()
                )

    def is_valid_image(self, mime_type):
        """Check if the file is a valid image type"""
        return mime_type in ['image/jpeg', 'image/png', 'image/jpg']


    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle document messages that might contain images"""
        document = update.message.document
        
        # Check if the document is an image
        mime_type = document.mime_type
        if not self.is_valid_image(mime_type):
            await update.message.reply_text(INVALID_FILE)
            return
        
        # Store the file ID and show options
        context.user_data['photo'] = document.file_id
        context.user_data['is_document'] = True
        
        keyboard = [[
            InlineKeyboardButton("دریافت کپشن اینستاگرام", callback_data='caption'),
            InlineKeyboardButton("اعمال فیلتر زیبایی", callback_data='filter')
        ]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(WHAT_TO_DO, reply_markup=reply_markup)


    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Process received photos and show options"""
        context.user_data['photo'] = update.message.photo[-1].file_id
        context.user_data['is_document'] = False
        
        keyboard = [[
            InlineKeyboardButton("دریافت کپشن اینستاگرام", callback_data='caption'),
            InlineKeyboardButton("اعمال فیلتر زیبایی", callback_data='filter')
        ]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(WHAT_TO_DO, reply_markup=reply_markup)


    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()
        
        if query.data == 'clear_history':
            context.user_data['chat_history'] = []
            await query.edit_message_text(
                HISTORY_CLEARED,
                reply_markup=self.get_clear_button_markup()
            )
            return
            
        if 'photo' not in context.user_data:
            await query.edit_message_text(NO_PHOTO)
            return

        if query.data == 'caption':
            await query.edit_message_text(GENERATING_CAPTION)
            photo_file = await context.bot.get_file(context.user_data['photo'])
            photo_bytes = await photo_file.download_as_bytearray()
            
            # Generate and translate caption
            caption = await self.generate_creative_caption(photo_bytes)
            persian_caption = await self.translate_text(caption)
            
            # Format with emojis
            formatted_caption = self.format_persian_caption(persian_caption)
            await query.edit_message_text(
                f"کپشن پیشنهادی:\n\n{formatted_caption}",
                reply_markup=self.get_clear_button_markup()
            )

        elif query.data == 'filter':
            photo_file = await context.bot.get_file(context.user_data['photo'])
            photo_bytes = await photo_file.download_as_bytearray()
            filtered_photo = self.apply_beauty_filter(photo_bytes)
            
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=filtered_photo,
                caption=FILTER_CAPTION
            )
            await query.edit_message_text(
                FILTER_APPLIED,
                reply_markup=self.get_clear_button_markup()
            )

    async def generate_creative_caption(self, photo_bytes):
        """Generate creative caption using Llava model through Ollama"""
        try:
            # Convert photo bytes to base64
            base64_image = base64.b64encode(photo_bytes).decode('utf-8')
            
            # Prepare the prompt for Instagram-style caption
            prompt = """Generate a creative and engaging Instagram caption for this image. 
            Make it personal and emotional, focusing on the story or feeling rather than just describing what's visible. 
            Keep it short and natural, like something a real person would write. include hashtags."""
            
            # Prepare the request for Ollama
            payload = {
                "model": "llava:13b",
                "prompt": prompt,
                "images": [base64_image],
                "stream": False
            }
            
            # Make the request to Ollama
            response = requests.post(self.ollama_endpoint, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                caption = result.get('response', '').strip()
                
                # Clean up the caption
                if caption.lower().startswith("this is"):
                    caption = caption[8:]
                
                return caption.strip() or "Capturing life's beautiful moments, one frame at a time ✨"
            else:
                print(f"Error from Ollama API: {response.status_code}")
                return "Capturing life's beautiful moments, one frame at a time ✨"
            
        except Exception as e:
            print(f"Error generating caption: {str(e)}")
            return "Capturing life's beautiful moments, one frame at a time ✨"

    async def translate_text(self, text):
        """Translate text from English to Persian"""
        try:
            translated = self.translator.translate(text)
            return translated if translated else text
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return text

    def format_persian_caption(self, caption):
        """Format the Persian caption with emoji"""
        if not any(char in caption for char in ['✨', '💫', '🌟']):
            caption = f"✨ {caption}"
        return caption

    def apply_beauty_filter(self, photo_bytes):
        """Apply beauty filter to photo"""
        nparr = np.frombuffer(photo_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Enhanced beauty filter
        smooth = cv2.bilateralFilter(img, 9, 75, 75)
        brightness = 1.1
        filtered = cv2.convertScaleAbs(smooth, alpha=brightness, beta=10)
        
        # Color enhancement
        hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.2
        filtered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        is_success, buffer = cv2.imencode(".jpg", filtered)
        return io.BytesIO(buffer)

    def run(self):
        """Start the bot"""
        print("Bot is running...")
        self.application.run_polling()

def main():
    bot = InstagramCaptionBot(TELEGRAM_TOKEN)
    bot.run()

if __name__ == '__main__':
    main()