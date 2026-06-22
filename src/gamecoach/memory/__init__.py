"""Player Memory — 玩家长期画像的 JSON 持久化存储。

提供 load / save / update 三个接口，MVP 阶段使用 JSON 文件存储。
update_player_memory 实现增量合并（弱点和角色偏好的去重追加）。
"""

from gamecoach.memory.store import load_player_memory, save_player_memory, update_player_memory

__all__ = ["load_player_memory", "save_player_memory", "update_player_memory"]
