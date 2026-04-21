# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# 你的 OpenAI API Key（优先从环境变量读取）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 模型选择
# - 创意探索：dall-e-3 或 gpt-image-1-mini
# - 定稿统一：gpt-image-1
MODELS = {
    "LLM_MODEL":    "gpt-4.1-mini",   # 文本/推理
    "VISION_MODEL": "gpt-4o-mini",    # 看图识别（Landmark Detector）
    "IMAGE_MODEL":  "gpt-image-1-mini"        # 图像生成（可改：gpt-image-1 / gpt-image-1-mini）
}

# 图像尺寸（受支持：1024x1024 / 1024x1536 / 1536x1024 / "auto"）
IMAGE_SIZE = "1024x1024"

# 创意采样数量（一次生成多张，随后自动评审挑最佳）
CREATIVE_SAMPLES = 4

# 评分阈值（orchestrator 用）
TARGETS = {
    "clarity": 80,
    "aesthetic": 80,
    "recognizability": 80
}
