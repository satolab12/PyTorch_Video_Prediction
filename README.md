# PyTorch_Video_Prediction
## Overview
動画予測タスク向きの各種モデルをPyTorchで実装しています。
性能比較等の検証結果や詳細は[Qiita](https://qiita.com/satolab/items/489fde0a8452080bf7e5)をご確認ください。
main.pyを実行すれば学習と検証をします。config.pyにて各種パラメータとデータセットの指定をします。データセットは動画形式ではなくフレーム画像に変換してください。（Dataloaderの仕様上）
dataset.pyで変換できます。モデルを変えて実行したい場合はmain.pyのモデル定義部を変更してください。

Various models for video prediction tasks are implemented in PyTorch. 
Run main.py to train and validate the models, and config.py to specify parameters and datasets. 
The dataset should be converted to frame images, not video format (due to Dataloader specifications).  
You can use dataset.py to convert them. If you want to run with different models,
change the model definition part of main.py.

## Model
### CNN+GRU
![図形 (3)](https://user-images.githubusercontent.com/56526560/167286624-91a052a0-f10e-4725-aa95-5efe909ce82e.jpg)
### CNN+ConvLSTM
![図形 (4)](https://user-images.githubusercontent.com/56526560/167286627-985efea0-5018-4687-8478-63b075cad64e.jpg)


## Qiita
### CNN+GRU(Part1)
[https://qiita.com/satolab/items/489fde0a8452080bf7e5](https://qiita.com/satolab/items/489fde0a8452080bf7e5)
### CNN+ConvLSTM(Part2)
https://qiita.com/satolab/items/bac43905f3427910d057

## References 
https://github.com/ndrplz/ConvLSTM_pytorch
https://note.nkmk.me/python-opencv-video-to-still-image/
