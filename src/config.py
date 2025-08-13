import os
from dotenv import load_dotenv
load_dotenv()

CHROMA_PATH = './chroma_db'
COLLECTION_NAME = "eora_data"
JSON_PATH = 'data.json'

GIGACHAT_CLIENT_SECRET = os.getenv('GIGACHAT_CLIENT_SECRET')
BOT_TOKEN = os.getenv('BOT_TOKEN')
