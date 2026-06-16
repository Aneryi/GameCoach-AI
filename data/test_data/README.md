# Test Data

本目录包含 GameCoach AI 的测试/演示数据。所有数据为通用的游戏概念，不绑定特定游戏。

## 文件说明

| 文件 | 内容 | 使用方 |
|------|------|--------|
| `player_memory.json` | 玩家长期画像（偏好角色、弱点、目标） | `memory/store.py` |
| `matches.json` | 玩家最近 20 场对战记录 | `tools/match_history.py` |
| `characters.json` | 5 个通用角色（Alpha-Echo）的属性数据 | `tools/hero_database.py` |
| `patch_meta.json` | 当前版本 meta 信息（强势角色、装备改动） | `tools/patch_meta.py` |
| `items.json` | 11 件通用装备（鞋子、攻击、防御） | `tools/items.py` |

## 数据结构

### player_memory.json
```json
{
  "player_id": "player_001",
  "favorite_characters": ["Delta", "Alpha"],
  "main_roles": ["damage"],
  "weaknesses": ["poor teamfight positioning", "excessive mid-game deaths"],
  "goals": ["reach Diamond rank"],
  "rank": "Platinum I"
}
```

### matches.json
每场记录包含：`match_id`, `result` (win/loss), `character`, `role`, `kills`, `deaths`, `assists`, `score`, `damage`, `teamfight_participation`, `duration_minutes`

### characters.json
每个角色包含：`role` (damage/tank/support), `difficulty` (low/medium/high), `strengths`, `weaknesses`

### patch_meta.json
包含：`patch` 版本号, `strong_characters`, `nerfed_characters`, `buffed_items`, `meta_summary`

### items.json
每件装备包含：`category` (boots/attack/defense), `stat`, `role_fit`, `note`

## 修改数据

1. 直接编辑对应的 JSON 文件
2. 运行 `pytest tests/ -v` 确认测试仍然通过
3. 如需切换玩家，修改 `player_id` 字段并在 `matches.json` 中增加对应的 key

## 添加新数据

- **新玩家**：在 `matches.json` 中增加新的 `player_id` key，复制 `player_001` 的对战记录格式
- **新角色**：在 `characters.json` 中增加新的角色条目
- **新装备**：在 `items.json` 中增加新的装备条目
- **新版本**：在 `patch_meta.json` 的 `game` 下增加新的 patch key
