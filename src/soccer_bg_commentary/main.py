"""
デモ動画を作成するためのスクリプト
"""

import os
import logging
from datetime import datetime
from traceback import print_exc
from typing import Literal, Optional

from dotenv import load_dotenv
from tap import Tap
import pandas as pd

from soccer_bg_commentary.demo import DemoRunner


# .env の環境変数を読み込む
load_dotenv()


# ===========
# Logger
# ===========
time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# フォーマッタ
formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ログファイル出力
file_handler = logging.FileHandler(f"logs/main--{time_string}.log")
file_handler.setFormatter(formatter)

# 標準出力
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# ログ+標準出力
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


# ===========
# Arguments
# ===========
class MainArgument(Tap):
    """
    メインスクリプトの引数
    """

    mode: Literal["run", "reference"] = "run"
    lang: Literal["ja", "en"] = "en"
    input_method: Literal["manual", "csv"] = "manual"
    comment_csv: str = "data/commentary/scbi-v2.csv"
    video_csv: str = "data/from_video/players_in_frames_sn_gamestate.csv"
    label_csv: str = "data/from_video/soccernet_spotting_labels.csv"
    # input_method == "manual"
    game: str = None
    half: int = None
    start: float = None
    end: float = None
    save_jsonl: str = "outputs/demo-step2/commentary.jsonl"
    save_srt: str = "outputs/demo-step2/commentary.srt"
    pbp_jsonl: Optional[str] = None
    # input_method == "csv"
    input_csv: Optional[str] = None
    output_base_dir: Optional[str] = "outputs/demo-step2"
    pbp_base_dir: Optional[str] = "data/demo/pbp"

    # その他
    default_rate: float = 0.18
    default_text_threshold: float = 1.0

    seed: int = 100


def run_commentary_generation_for_video(
    mode: Literal["run", "reference"],
    game: str,
    half: int,
    start: float,
    end: float,
    comment_csv: str,
    video_csv: str,
    label_csv: str,
    save_jsonl: str,
    save_srt: str,
    lang: str = "ja",
    seed: int = 100,
    play_by_play_jsonl: str = None,
):
    demo_runner = DemoRunner(
        game, half, comment_csv, video_csv, label_csv,
        lang=lang, seed=seed,
        force_rate=0.5, default_rate=0.18
    )
    try:
        if mode == "run":
            demo_runner.run(
                start, end, save_jsonl, save_srt, play_by_play_jsonl=play_by_play_jsonl
            )
        elif mode == "reference":
            demo_runner.reference(start, end, save_jsonl, save_srt)
        else:
            raise ValueError(f"無効なモードです: {mode}")
    except Exception:
        logger.error(f"エラーが出ました: {save_jsonl=}")
        print_exc()
        return


# ===========
# Main Procedure
# ===========
if __name__ == "__main__":
    args = MainArgument().parse_args()

    if args.input_method == "manual":
        assert (
            (args.game is not None)
            and (args.half is not None)
            and (args.start is not None)
            and (args.end is not None)
        )

        logger.info(
            f"実行中 => ゲーム: {args.game}, ハーフ: {args.half}, 開始時間: {args.start}, 終了時間: {args.end}, "
            f"JSONL保存先: {args.save_jsonl}, SRT保存先: {args.save_srt}"
        )
        run_commentary_generation_for_video(
            args.mode, args.game, args.half, args.start, args.end,
            args.comment_csv, args.video_csv, args.label_csv,
            args.save_jsonl, args.save_srt,
            lang=args.lang, seed=args.seed,
            play_by_play_jsonl=args.pbp_jsonl
        )

    elif args.input_method == "csv":
        assert (args.input_csv is not None) and (args.output_base_dir is not None)

        # 複数動画 まとめて実況生成
        input_df = pd.read_csv(args.input_csv)
        assert {"id", "game", "half", "start", "end"}.issubset(set(input_df.columns))

        # 保存ファイル名
        save_basename = "commentary" if args.mode == "run" else "ref"

        for _, row in input_df.iterrows():
            pbp_jsonl = os.path.join(
                f"{args.pbp_base_dir}",
                f"{row['id']:04d}",
                f"play-by-play-{args.lang}.jsonl",
            )
            save_jsonl = os.path.join(
                f"{args.output_base_dir}",
                f"{row['id']:04d}",
                f"{save_basename}-full-{args.lang}.jsonl",
            )
            save_srt = os.path.join(
                f"{args.output_base_dir}",
                f"{row['id']:04d}",
                f"{save_basename}-full-{args.lang}.srt",
            )

            if (
                os.path.exists(save_jsonl)
                and os.path.exists(save_srt)
                and input(f"{row['id']} 上書きしますか？ [y/n] ").lower() != "y"
            ):
                logger.info(f"すでに存在するためSkip: {row['id']}")
                continue

            if not os.path.exists(pbp_jsonl):
                logger.info(f"Play by Play がないためSkip: {row['id']}")
                continue

            logger.info(
                f"RUN => Game: {row['game']}, Half: {row['half']}, Start: {row['start']}, End: {row['end']},"
                f"Save JSONL: {save_jsonl}, Save SRT: {save_srt}"
            )
            run_commentary_generation_for_video(
                args.mode, row["game"], row["half"], row["start"], row["end"],
                args.comment_csv, args.video_csv, args.label_csv,
                save_jsonl, save_srt,
                lang=args.lang, seed=args.seed,
                play_by_play_jsonl=pbp_jsonl,
            )

        logger.info("Finished")
    else:
        raise ValueError(f"無効な入力方法です: {args.input_method}")
