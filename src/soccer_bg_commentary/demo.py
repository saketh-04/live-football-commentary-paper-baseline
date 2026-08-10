"""
デモ動画を作成する字幕を生成するためのクラスを提供するモジュール
"""

import os
from typing import Optional, Tuple

from langchain_openai import ChatOpenAI as LangChainOpenAI
import numpy as np

from soccer_bg_commentary.entity import (
    CommentDataList,
    SpottingDataList,
    VideoData,
    CommentData,
    SpottingData,
)
from soccer_bg_commentary.util import (
    format_docs,
    log_documents,
    log_prompt,
    get_utterance_length,
    get_utterance_length_ja
)
from soccer_bg_commentary.spotting_module import SpottingModule, SpottingArgment
from soccer_bg_commentary.construct_query import build_query
from soccer_bg_commentary.addinfo_retrieval import get_rag_chain, get_retriever
from soccer_bg_commentary.play_by_play import PlayByPlayGenerator
from soccer_bg_commentary.constants import (
    MODEL_CONFIG,
    EMBEDDING_CONFIG,
    SEARCH_CONFIG,
    PERSIST_LANGCHAIN_DIR,
    INSTRUCTION,
    INSTRUCTION_JA,
    COMMENT_TO_EVENT_TEXT_PROMPT,
    COMMENT_TO_EVENT_TEXT_PROMPT_JA,
)
from logging import getLogger


logger = getLogger(__name__)


class DemoRunner:
    """
    デモ動画のための実況を生成する
    """

    def __init__(
        self,
        game: str,
        half: int,
        comment_csv: str,
        video_csv: str,
        label_csv: str,
        lang: str = "en",
        seed: int = 100,
        **kwargs,
    ):
        # スポッティングモジュール関連
        rng = np.random.RandomState(seed)

        spotting_params: SpottingArgment = SpottingArgment().parse_known_args()[0]
        spotting_params.default_rate = kwargs.get(
            "default_rate", spotting_params.default_rate
        )
        spotting_params.force_rate = kwargs.get(
            "force_rate", spotting_params.force_rate
        )

        spotting_model = SpottingModule(spotting_params, rng=rng)

        # 選手同定+ragモジュール関連
        instruction = INSTRUCTION_JA if lang == "ja" else INSTRUCTION
        game_metadata = SpottingDataList.extract_data_from_game(game)
        gold_comment_data_list: CommentDataList = CommentDataList.read_csv(
            comment_csv, game
        )
        video_data = VideoData(video_csv, label_csv=label_csv)
        llm = LangChainOpenAI(**MODEL_CONFIG)
        retriever = get_retriever(
            "openai-embedding",
            langchain_store_dir=PERSIST_LANGCHAIN_DIR,
            embedding_config=EMBEDDING_CONFIG,
            search_config=SEARCH_CONFIG,
        )
        rag_chain, _, _ = get_rag_chain(
            retriever=retriever,
            llm=llm,
            log_documents=log_documents,
            log_prompt=log_prompt,
            format_docs=format_docs,
            instruction=instruction,
        )
        # メンバ変数
        self.llm = llm
        self.default_text_threshold = kwargs.get("default_text_threshold", 1.0)
        self.lang = lang
        self.rng = rng
        self.seed = seed
        self.game = game
        self.half = half
        self.gold_comment_data_list = (
            gold_comment_data_list  # 映像の説明のモック用・検索クエリを補強する用
        )
        self.video_data = video_data
        self.spotting_model = spotting_model
        self.game_metadata = game_metadata
        self.rag_chain = rag_chain
        self.func_utterance_length = (
            get_utterance_length_ja if lang == "ja" else get_utterance_length
        )

    def build_extended_query(self, comment_data_list: CommentDataList, time: float):
        """game, half, game_metadata, video_data は関数内で定義された変数を用いる"""
        filtered_comment_list = CommentDataList.filter_by_half_and_time(
            comment_data_list, self.half, time
        )
        video_data_dict = self.video_data.get_data(self.game, self.half, time)

        query_args = {
            "comments": filtered_comment_list,
            "game_metadata": self.game_metadata,
            "players": video_data_dict["players"],
            "actions": video_data_dict["actions"],
        }

        query = build_query(**query_args)
        return query

    def get_pbp_alternative_commentary(self, comment: str) -> str:
        # COMMENT_TO_EVENT_TEXT_PROMPT
        template = (
            COMMENT_TO_EVENT_TEXT_PROMPT_JA
            if self.lang == "ja"
            else COMMENT_TO_EVENT_TEXT_PROMPT
        )
        prompt = template.format(comment=comment)
        response = self.llm.generate([[prompt]])
        return response.generations[0][0].text

    def run(
        self,
        start: float,
        end: float,
        save_jsonl: str,
        save_srt: str,
        play_by_play_jsonl: Optional[str] = None,
    ):
        finish_time = end
        prev_end = start
        comment_data_list = self.initialize_comment_data_list(start)

        play_by_play_func = self.setup_play_by_play(play_by_play_jsonl, start)

        while True:
            next_ts, next_label = self.spotting_model(
                previous_t=prev_end, game=self.game, half=self.half
            )
            comment = self.generate_comment(
                next_ts, next_label, comment_data_list, play_by_play_func
            )

            if comment is None or comment_data_list.is_duplicate(comment):
                prev_end = self.adjust_prev_end(prev_end)
                continue

            next_start, next_end = self.calculate_comment_timing(next_ts, comment)
            self.add_comment_to_list(
                comment_data_list, next_start, next_end, comment, next_label
            )

            logger.info(
                f"s: {next_start:.02f}, e: {next_end:.02f}, label: {next_label}, comment: {comment}"
            )

            if finish_time <= next_end:
                break

            prev_end = next_end

        self.save_comments(comment_data_list, save_jsonl, save_srt, start)

    def initialize_comment_data_list(self, start: float) -> CommentDataList:
        """検索クエリ増強のため、指定した時間範囲より過去のコメントでコメント履歴を初期化する"""
        comment_data_list = CommentDataList([])
        for comment in self.gold_comment_data_list.comments:
            if comment.half == self.half and comment.end_time < start:
                comment_data_list.comments.append(comment)
        return comment_data_list

    def setup_play_by_play(self, play_by_play_jsonl: Optional[str], start: float):
        if play_by_play_jsonl and os.path.exists(play_by_play_jsonl):
            """映像の説明を生成する関数を設定
            play_by_play_jsonl が指定されている場合、田中さんのシステムによる映像の説明を利用する
            play_by_play_jsonl が指定されていない場合、現実の実況を参考にしつつ、映像の説明を生成する
            """
            self.play_by_play_generator = PlayByPlayGenerator(
                pbp_jsonl=play_by_play_jsonl,
                lang=self.lang,
                rng=self.rng,
                base_time=start,
                default_text_threshold=self.default_text_threshold,
            )
            return self.play_by_play_generator.generate
        else:
            return lambda next_ts: self.get_pbp_alternative_commentary(
                self.gold_comment_data_list.get_comment_nearest_time(next_ts)
            )

    def generate_comment(
        self,
        next_ts: float,
        next_label: int,
        comment_data_list: CommentDataList,
        play_by_play_func,
    ) -> Optional[str]:
        """次のタイムスタンプとラベルに基づいてコメントを生成する関数

        引数:
        next_ts (float): 次のタイムスタンプ
        next_label (int): 次のラベル (0または1)
        comment_data_list (CommentDataList): コメント履歴
        play_by_play_func (function): 映像の説明を生成する関数

        戻り値:
        Optional[str]: 生成されたコメント。ラベルが0の場合は映像の説明、ラベルが1の場合は付加的情報。
        """
        if next_label == 0:
            return play_by_play_func(next_ts)
        elif next_label == 1:
            query = self.build_extended_query(comment_data_list, next_ts)
            spot = SpottingData(
                game=self.game,
                half=self.half,
                category=next_label,
                game_time=next_ts,
                query=query,
                position=int(next_ts) * 1000,
                confidence=1.0,
            )
            return self.rag_chain.invoke(spot)
        else:
            raise RuntimeError(f"無効な発話ラベルです: {next_label}")

    def adjust_prev_end(self, prev_end: float) -> float:
        return prev_end + 0.5

    def calculate_comment_timing(
        self, next_ts: float, comment: str
    ) -> Tuple[float, float]:
        next_start = next_ts
        next_end = next_ts + self.func_utterance_length(comment)
        return next_start, next_end

    def add_comment_to_list(
        self,
        comment_data_list: CommentDataList,
        next_start: float,
        next_end: float,
        comment: str,
        next_label: int,
    ):
        comment_data_list.comments.append(
            CommentData(
                half=int(self.half),
                start_time=next_start,
                end_time=next_end,
                text=comment,
                category=int(next_label),
            )
        )

    def save_comments(
        self,
        comment_data_list: CommentDataList,
        save_jsonl: str,
        save_srt: str,
        start: float,
    ):
        for comment in self.gold_comment_data_list.comments:
            if comment in comment_data_list.comments:
                comment_data_list.comments.remove(comment)
        comment_data_list.to_jsonline(save_jsonl)
        comment_data_list.to_srt(save_srt, base_time=start)

    def reference(self, start: float, end: float, save_jsonl: str, save_srt: str):
        """現実の実況（参照例）を出力する"""
        comment_data_list = CommentDataList([])
        for comment in self.gold_comment_data_list.comments:
            if (
                comment.half == self.half
                and comment.start_time >= start
                and comment.start_time < end
            ):
                comment_data_list.comments.append(comment)
        comment_data_list.to_jsonline(save_jsonl)
        comment_data_list.to_srt(save_srt, base_time=start)
