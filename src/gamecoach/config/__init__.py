"""配置层 — 环境变量加载（自动 load_dotenv）、LLM/Embedding 模型工厂。

settings.py: 从 .env 和系统环境变量读取配置，提供 Settings 数据类。
llm.py: DeepSeek Chat（get_chat_model）和 DashScope Embedding（get_embeddings）全局单例。
"""
