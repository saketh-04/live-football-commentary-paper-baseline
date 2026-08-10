from uuid import uuid4

import os
from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request

import logging
import subprocess
import pandas as pd

from soccer_bg_commentary.demo import DemoRunner


valid_examples = pd.read_csv("data/demo/sample_metadata.csv")

assert set(valid_examples.columns) == {"id", "game", "half", "start", "end"}
# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要に応じて許可するオリジンを指定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate/")
async def generate_video(
    request: Request,
    game: str = Form(...),
    half: int = Form(...),
    start: float = Form(...),
    end: float = Form(...),
):
    ############################
    # リクエストのバリデーション
    ############################
    if not (0 <= start < end):
        logger.error("Invalid start and end time.: %s, %s", start, end)
        raise HTTPException(
            status_code=400,
            detail="Invalid start and end time."
        )
    # 各exampleのstart, endの範囲を超えていないか
    valid_example = valid_examples[
        (valid_examples["game"] == game) & (valid_examples["half"] == half)
    ]
    if valid_example.empty:
        logger.error("Invalid game and half.: %s, %s", game, half)
        raise HTTPException(
            status_code=400,
            detail="Invalid game and half."
        )

    valid_start = valid_example["start"].values[0]
    valid_end = valid_example["end"].values[0]
    if not (valid_start <= start < end <= valid_end):
        logger.error("Invalid start and end time.: %s, %s", start, end)
        raise HTTPException(
            status_code=400,
            detail="Invalid start and end time."
        )

    ############################
    # データの準備
    ############################
    sample_id = valid_example["id"].values[0]
    # play-by-playファイルのパス
    pbp_path = f"data/demo/pbp/{sample_id:04d}/play-by-play-en.jsonl"

    # もとの動画
    src_video_path = f"outputs/demo-step2/{sample_id:04d}/video.mp4"

    # UUIDを生成してファイル名に使用
    unique_id = str(uuid4())
    save_jsonl = f"outputs/demo-step2/output_{unique_id}.jsonl"
    save_srt = f"outputs/demo-step2/output_{unique_id}.srt"

    audio_path = f"outputs/demo-step2/output_{unique_id}.wav"
    mid_video_path = f"outputs/demo-step2/mid_{unique_id}.mp4"
    dst_video_path = f"outputs/demo-step2/{unique_id}.mp4"

    ############################
    # デモ動画作成
    ############################
    # DemoRunnerのインスタンスを作成
    demo_runner = DemoRunner(
        game=game,
        half=half,
        comment_csv="data/commentary/scbi-v2.csv",
        video_csv="data/from_video/players_in_frames_sn_gamestate.csv",
        label_csv="data/from_video/soccernet_spotting_labels.csv",
        lang="en",
        mode="run"
    )

    # デモの実行
    demo_runner.run(start, end, save_jsonl, save_srt, play_by_play_jsonl=pbp_path)

    # 音声合成と動画生成
    gen_wav_command = "python src/soccer_bg_commentary/srt_to_wav.py"
    gen_wav_command += f" --input_srt {save_srt} --output_wav {audio_path} --lang openai"

    # 焼き付け
    add_sub_command = f"ffmpeg -y -i {src_video_path} -vf subtitles={save_srt} {mid_video_path}"

    # 音声を動画に結合
    add_wav_command = \
        f"ffmpeg -y -i {mid_video_path} -i {audio_path} -c:v copy -map 0:v:0 -map 1:a:0 -shortest {dst_video_path}"

    try:
        # 音声合成
        subprocess.run(gen_wav_command, shell=True, check=True)
        # 字幕焼付け
        subprocess.run(add_sub_command, shell=True, check=True)
        # 音声を動画に結合
        subprocess.run(add_wav_command, shell=True, check=True)
    except subprocess.CalledProcessError:
        logger.error("Failed to generate video.")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate video (some internal processes were broken)."
        )

    if not os.path.exists(dst_video_path):
        logger.error("Failed to generate video. (no resulting mp4 data was generated)")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate video (for some reasons, no resulting mp4 data was generated)."
        )

    # 生成された動画ファイルをダウンロード
    return FileResponse(
        dst_video_path,
        media_type='video/mp4',
        filename=f"{unique_id}.mp4"
    )
