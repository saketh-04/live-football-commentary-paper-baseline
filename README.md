# soccer-bg-commentary

## Preliminary

[uv](https://github.com/astral-sh/uv)でパッケージ管理しています。

## Tree

```bash
.
├── archives
│   └── system-data.zip
├── data
│   └── ...
├── scripts
│   ├── demo
│   │   ├── run.sh
│   │   ...
│   │   └── submission.sh
│   ├── preparation
│   │   └── setup-data.sh
│   ├── rag
│   │    └── ...
│   └── spotting
│        └── ...
├── src
│   └── soccer_bg_commentary
│       ├── addinfo_retrieval.py
│      ...
│       └── main.py
├── pyproject.toml
...
└── README.md
```

## data

archives/system-data.zip には以下のデータが含まれています。

```bash
├── addinfo_retrieval  [3249 entries exceeds filelimit, not opening dir]
├── commentary
│   └── scbi-v2.csv
├── demo
│   ├── pbp
│   │   └── `%04d`
│   │       ├── play-by-play-en.jsonl
│   │       └── play-by-play-ja.jsonl
│   ├── Action_and_Rates_Data.csv
│   ├── Additional_Info_Ratios__Before_and_After.csv
│   ├── Extracted_Action_Rates.csv
│   ├── sample_metadata.csv
│   └── silence_distribution.csv
├── from_video
│   ├── players_in_frames.csv
│   ├── players_in_frames_sn_gamestate.csv
│   └── soccernet_spotting_labels.csv
├── reference_documents
│   └── evaluation-samples.yaml
└── spotting
    ├── test.csv
    ├── train.csv
    └── valid.csv
```

このうち、重要なファイルを紹介します

- ラベル付き実況コメントデータ (commentary/)
- 外部知識 (addinfo_retrieval/)
  - [wikipediaから選手情報を収集](https://github.com/zaemon1251-hesty/sn-script/blob/dev/src/sn_script/download_articles.py)
- 実況生成デモに用いる情報群 (from_video/, demo/)
  - `players_in_frames_sn_gamestate.csv`は、[tracklab](https://github.com/zaemon1251-hesty/tracklab)で生成した選手追跡結果を、[soccer-bg-script](https://github.com/zaemon1251-hesty/soccer-bg-script)で選手名およびボール座標を付与する後処理を施したもの
  - `soccernet_spotting_labels.csv`は[soccer-bg-script](https://github.com/zaemon1251-hesty/soccer-bg-script)で作成したAction Spottingのラベルcsv
  - `pbp`はGoogle Drive経由で田中さんから受け取った、Play-by-Playのjsonlファイル
  - `sample_metadata.csv`は、評価およびデモ動画作成に使った映像のメタデータ

## 使い方

```bash
# setup data
scripts/setup-data.sh

# generate srt format commentary
scripts/demo/run.sh

# Integrated execution from live generation to demo video creation for multiple videos.
scripts/demo/submission.sh

# run APIserver
uv run uvicorn src.app:app --reload

# run frontend
cd front
npm run start
```
