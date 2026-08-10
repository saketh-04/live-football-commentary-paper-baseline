#!/bin/bash
# 異なるseedで10回平均
# 1秒未満を含めたデータでの 発話間隔の平均は 2.14 s
# 結果
# タイミング 提案法がベースラインより良い (前の実験設定との差分: (1)4秒以内への強制が効いている & (2)データのフィルタリングがなくなって、データの特徴が変わった)
# ラベル 提案法がベースラインより良い

path="./data/spotting"

echo "System:"
echo "empirical, action_spotting, Separate, Addinfo force"
uv run python src/soccer_bg_commentary/spotting_module.py --split test --path $path --seed 100 \
    --timing_algo empirical \
        --empirical_dist_csv  "data/demo/silence_distribution.csv" \
        --mean_silence_sec 2.14 \
    --label_algo action_spotting \
        --action_window_size 15 \
        --action_rate_csv "data/demo/Additional_Info_Ratios__Before_and_After.csv" \
        --default_rate 0.18 \
        --addinfo_force \
        --only_offplay

<< COMMENTOUT
2025-03-23 18:39:57.084 | INFO     | __main__:<module>:528 - diff_average: 15.503
2025-03-23 18:39:57.084 | INFO     | __main__:<module>:528 - label_accuracy_system: 61.001%
2025-03-23 18:39:57.084 | INFO     | __main__:<module>:528 - label_precision_system: 20.238%
2025-03-23 18:39:57.084 | INFO     | __main__:<module>:528 - label_recall_system: 38.346%
2025-03-23 18:39:57.084 | INFO     | __main__:<module>:528 - label_f1_system: 26.493%
2025-03-23 18:39:57.084 | INFO     | __main__:<module>:528 - label_accuracy_gold: 61.001%
2025-03-23 18:39:57.084 | INFO     | __main__:<module>:528 - label_precision_gold: 20.689%
2025-03-23 18:39:57.084 | INFO     | __main__:<module>:528 - label_recall_gold: 38.958%
2025-03-23 18:39:57.084 | INFO     | __main__:<module>:528 - label_f1_gold: 27.025%
COMMENTOUT

# echo "----------------------------------------"

# echo "Action Distribution:"
# echo "empirical, action_spotting, Separate, Addinfo force"
# uv run python src/soccer_bg_commentary/spotting_module.py --split test --path $path --seed 100 \
#     --timing_algo empirical \
#         --empirical_dist_csv  "data/demo/silence_distribution.csv" \
#         --mean_silence_sec 2.14 \
#     --label_algo action_spotting \
#         --action_window_size 15 \
#         --action_rate_csv "data/demo/Additional_Info_Ratios__Before_and_After.csv" \
#         --default_rate 0.18 \
#         --only_offplay
# << COMMENTOUT
# 2025-03-23 18:53:02.731 | INFO     | __main__:<module>:524 - diff_average: 15.503
# 2025-03-23 18:53:02.731 | INFO     | __main__:<module>:524 - label_accuracy_system: 69.170%
# 2025-03-23 18:53:02.731 | INFO     | __main__:<module>:524 - label_precision_system: 19.224%
# 2025-03-23 18:53:02.731 | INFO     | __main__:<module>:524 - label_recall_system: 20.766%
# 2025-03-23 18:53:02.731 | INFO     | __main__:<module>:524 - label_f1_system: 19.965%
# 2025-03-23 18:53:02.731 | INFO     | __main__:<module>:524 - label_accuracy_gold: 69.170%
# 2025-03-23 18:53:02.731 | INFO     | __main__:<module>:524 - label_precision_gold: 19.336%
# 2025-03-23 18:53:02.731 | INFO     | __main__:<module>:524 - label_recall_gold: 20.910%
# 2025-03-23 18:53:02.732 | INFO     | __main__:<module>:524 - label_f1_gold: 20.092%
# COMMENTOUT


# echo "----------------------------------------"


# echo "Baseline:"
# echo "constant, constant,,"
# uv run python src/soccer_bg_commentary/spotting_module.py --split test --path $path --seed 100 \
#     --timing_algo constant \
#         --mean_silence_sec 2.14 \
#     --label_algo constant
# << COMMENTOUT
# 2025-03-23 18:53:18.518 | INFO     | __main__:<module>:524 - diff_average: 19.864
# 2025-03-23 18:53:18.518 | INFO     | __main__:<module>:524 - label_accuracy_system: 69.934%
# 2025-03-23 18:53:18.518 | INFO     | __main__:<module>:524 - label_precision_system: 18.631%
# 2025-03-23 18:53:18.518 | INFO     | __main__:<module>:524 - label_recall_system: 18.548%
# 2025-03-23 18:53:18.518 | INFO     | __main__:<module>:524 - label_f1_system: 18.589%
# 2025-03-23 18:53:18.518 | INFO     | __main__:<module>:524 - label_accuracy_gold: 69.934%
# 2025-03-23 18:53:18.519 | INFO     | __main__:<module>:524 - label_precision_gold: 18.598%
# 2025-03-23 18:53:18.519 | INFO     | __main__:<module>:524 - label_recall_gold: 18.420%
# 2025-03-23 18:53:18.519 | INFO     | __main__:<module>:524 - label_f1_gold: 18.508%
# COMMENTOUT


<< COMMENTOUT
オフプレー　0.5 の結果

2025-03-25 17:56:52.799 | INFO     | __main__:<module>:519 - diff_average: 15.503
2025-03-25 17:56:52.799 | INFO     | __main__:<module>:519 - label_accuracy_system: 66.260%
2025-03-25 17:56:52.799 | INFO     | __main__:<module>:519 - label_precision_system: 19.640%
2025-03-25 17:56:52.799 | INFO     | __main__:<module>:519 - label_recall_system: 26.831%
2025-03-25 17:56:52.799 | INFO     | __main__:<module>:519 - label_f1_system: 22.679%
2025-03-25 17:56:52.799 | INFO     | __main__:<module>:519 - label_accuracy_gold: 66.260%
2025-03-25 17:56:52.799 | INFO     | __main__:<module>:519 - label_precision_gold: 19.886%
2025-03-25 17:56:52.799 | INFO     | __main__:<module>:519 - label_recall_gold: 27.080%
2025-03-25 17:56:52.800 | INFO     | __main__:<module>:519 - label_f1_gold: 22.932%

デモ動画を見ての定性評価
サンプル番号 15,25,28　は、0.5にすることで、ちょうどいい感じの割合になっている。

それ以外の8,10などは、依然付加的情報が多い。
というのも、実装上は、play event textが用意できない場所は付加的情報を入れることになっている。
そのため、8,10のようなプレーしていない時間が長いような例では、付加的情報が多くなってしまう。

COMMENTOUT
