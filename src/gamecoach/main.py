from gamecoach.graph.workflow import graph


def run_demo() -> None:
    result = graph.invoke(
        {
            "user_message": "我最近胜率很低，玩射手总是团战暴毙，怎么上分？",
            "player_id": "player_001",
            "game": "moba",
        }
    )
    print(result["final_response"])


if __name__ == "__main__":
    run_demo()

