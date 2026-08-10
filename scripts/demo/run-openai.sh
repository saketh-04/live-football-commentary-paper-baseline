#!/bin/bash

base_dir=outputs/demo-step2

# base_dir 以下の任意のサブディレクトリから commentary-ja.srt を再帰的に探す
srt_files=("$base_dir"/**/commentary-full-ja.srt)

# 各 commentary.srt に対して処理を実行
for srt_file in "${srt_files[@]}"; do
    echo "Processing SRT: $srt_file"
    # SRT ファイルと同じディレクトリに WAV ファイルを生成（ファイル名は commentary.wav）
    srt_dir=$(dirname "$srt_file")
    wav_file="$srt_dir/commentary-full-openai-ja.wav"
    
    # Python スクリプトを呼び出して音声合成を実行
    uv run python src/soccer_bg_commentary/srt_to_wav.py \
        --input_srt "$srt_file" \
        --output_wav "$wav_file" \
        --lang openai
done


# base_dir 以下の任意のサブディレクトリから commentary-en.srt を再帰的に探す
srt_files=("$base_dir"/**/commentary-full-en.srt)

# 各 commentary.srt に対して処理を実行
for srt_file in "${srt_files[@]}"; do
    echo "Processing SRT: $srt_file"
    # SRT ファイルと同じディレクトリに WAV ファイルを生成
    srt_dir=$(dirname "$srt_file")
    wav_file="$srt_dir/commentary-full-openai-en.wav"
    
    # Python スクリプトを呼び出して音声合成を実行
    uv run python src/soccer_bg_commentary/srt_to_wav.py \
        --input_srt "$srt_file" \
        --output_wav "$wav_file" \
        --lang openai
done

echo "All done."