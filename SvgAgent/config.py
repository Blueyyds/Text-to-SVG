import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# API配置
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.deepseek.com/v1")
API_KEY = os.getenv("API_KEY", "sk-40efc12af89e43048bf861e322e75560")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

# SVG配置
DEFAULT_SVG_WIDTH = 800
DEFAULT_SVG_HEIGHT = 600
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 1  # seconds

# 缓存配置
CACHE_DIR = "cache"
CACHE_ENABLED = True
CACHE_EXPIRY = 3600  # 1 hour

# 验证配置
VALIDATION_TIMEOUT = 10  # seconds
