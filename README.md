## プロジェクト概要
本リポジトリでは、歩行パターンとしてtrotとpaceを学習させたAIモデルをそれぞれ用意し、坂道ではtrotを平地ではpaceを使用させることで単一の歩行パターンよりも効率良く歩行を行うことを目的としている。


## ファイルについて
基本的に、Genesis/examples/locomotion以下にある"go2_"から始まるファイル群によって訓練と推論を行っている。 \\

#### 1.go2_env.py
このファイルで環境を定義している。　

#### 2.go2_train.py
このファイルは訓練を行うためのファイルである。
使用方法については
"""
python go2_train.py  --imitations pace, --exp_name go2
"""
のように歩行パターンの名前と実験名を指定すればよい。

#### 3.go2_eval.py
このファイルは推論を行う物である。
"""
python go2_eval.py --imitations pace -e go2-walking-pace
"""
のように歩行パターンと使用したいモデルを学習させた時の実験名を指定する。

#### 4.go2_switcher_eval.py
このファイルは自立切り替えロボットを実装したものである
"""
python go2_switcher_eval.py --trot_model_path "logs/go2-walking-trot_slope" --pace_model_path "logs/go2-walking-pace" --exp_name "go2-trot_slope"
"""
のように、trotのモデルパスとpaceのモデルパスをしていすれば良い
