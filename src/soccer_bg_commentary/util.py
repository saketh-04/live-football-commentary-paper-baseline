import logging
import textwrap


logger = logging.getLogger(__name__)


def wrap_text(text: str, max_width: int = 80) -> str:
    """
    英語の場合は単語単位で折り返し、
    日本語の場合は単純にmax_width文字ごとに改行します。
    """
    # 英語判定: 全ての文字がASCIIなら英語とみなす（簡易判定）
    if all(ord(c) < 128 for c in text):
        return "\n".join(textwrap.wrap(text, width=max_width))
    else:
        # 日本語の場合は指定した文字数ごとに分割
        return "\n".join(
            [text[i: i + max_width] for i in range(0, len(text), max_width)]
        )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def log_documents(docs):
    for doc in docs:
        logger.info(f"Document: {doc.page_content}")
    return docs


def log_prompt(prompt: str) -> str:
    logger.info(f"Overall Prompt: {prompt}")
    return prompt


def to_gametime(half, seconds: float) -> str:
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milsec = (seconds - int(seconds)) * 100
    return f"{int(half)} - {int(minutes):02d}:{int(seconds):02d}.{int(milsec):02d}"


def get_utterance_length(utterance: str):
    # 200 wpm (早口)という設定で，発話の長さを計算
    # return 秒数
    words = utterance.split()
    word_count = len(words)
    time = word_count * (60.0 / 200.0)
    return time


def get_utterance_length_ja(
    utterance: str, base_time: float = 0.12, pause_time: float = 0.2
) -> float:
    """
    日本語テキストのおおよその発話時間（秒）を計算する

    各文字の発話に base_time 秒、句読点（「、」「。」）の後には追加で pause_time 秒のポーズを想定

    Parameters:
        utterance (str): 発話テキスト
        base_time (float, optional): 1文字あたりの発話時間（秒）。デフォルトは 0.12 秒 (o1の提案を採用)
        pause_time (float, optional): 句読点後の追加ポーズ時間（秒）。デフォルトは 0.2 秒

    Returns:
        float: 推定発話時間（秒）
    """
    # punctuation = "、。"
    # 句読点の数をカウント
    # punctuation_count = sum(1 for c in utterance if c in punctuation)

    # 各文字に対する時間 + 句読点に対する追加ポーズ
    total_time = len(utterance) * base_time  # + punctuation_count * pause_time
    return total_time
