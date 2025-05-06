---
title: "複数のデータセットを結合したデータセットを自作（PyTorch）"
emoji: "📌"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Python", "PyTorch", "深層学習", ]
published: true
---

## なんで作りたいの？

　卒業研究でPyTorchを利用した深層学習をやっていた時に、1つの音声データを細かく区切ってデータセットを作っていました。
　データセットに入れる音声データを増やすとなったときに、データセットを一から作り直すのがひたすら面倒だったので…
「音声データごとにデータセット作って後からくっつければ良くね？？？？？？」
と思った次第でございます。
　後々よく調べてみると、`torch.utils.data.ConcatDataset`という全く同じ名前のクラスがありました。みんな考えることって同じなんですね…。ただ、自分で実装していろいろ理解が深まりました。
　初心者研究生なおかつZennも初心者なので~~高速道路の合流もままなりませんが~~アドバイスなどありましたら気軽にくださると非常に感謝です。

## コードの説明
以下、実装したコードです。下に補足をしています。

```py
from torch.utils.data import Dataset
import numpy as np

class ConcatDataset(Dataset):
    # データセットのリストを受け取る
    def __init__(self,datasets:list):
        super().__init__()
        self.datasets = datasets
        # 一つ一つのデータセットの長さを格納
        datalen = [len(data) for data in self.datasets]
        # 各データセットの開始位置を格納
        self.startindex = [sum(datalen[:i]) for i in range(len(datalen))]
        # 結合データセットの大きさ
        self.len = sum(datalen)

    def __getitem__(self, index):
        # indexがデータ数よりも多いときに終了
        if index >= self.len:
            raise(StopIteration)
        # indexを対応する正のindexに変換（indexが-1などのとき）
        index = index % len(self)
        # 呼び出すべきデータセットの位置
        dataset_index = np.searchsorted(self.startindex, index, side = "right") - 1
        # データセット内のデータの位置
        data_index = index - self.startindex[dataset_index]
        return self.datasets[dataset_index][data_index]

    def __len__(self):
        return self.len
```

　このConcatDatasetはデータセットのリストを引数として受け取り、リストの先頭のデータから順に吐き出していきます。
　試行錯誤していて見つけた`np.searchsorted()`が非常に便利で、第一引数のリストを昇順にソートしたのち、第二引数が入るべきindexを返してくれます。
　`self.startindex`は各データセットの０番目と結合したデータセットの開始地点を対応付けたリストで、昇順に並んでいます。このリストと`__getitem__`で指定された`index`を引数に渡すことで、何番目のデータセットにアクセスすればよいかを一発で求められるのに感動しました…！！

## MNISTで確認してみる
作成したコードをMNISTデータセットで確認してみます。

```py
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

dataset_1 = MNIST(root = "./data",download = True,train = True,transform = transforms.ToTensor())
print(len(dataset_1))
dataset = ConcatDataset([dataset_1 for _ in range(2)])
print(len(dataset))
print((dataset_1[0][0] == dataset[60000][0]).all())
print((dataset_1[-1][0] == dataset[-1][0]).all())

```
```powershell:出力
60000
120000
tensor(True)
tensor(True)
```

　やってることは単純で、MNISTの訓練データを2個くっつけたのちに、まずデータの数は正しいかどうか、そのあとに1個分のデータセットの最初のデータとくっつけた方の60000個目のデータが一致しているか（訓練データは60000個ある）、1個分のデータセットの最後のデータとくっつけた方の最後のデータが一致しているかを調べているだけです。
　正しく結合できていることがわかります。これが間違っていた場合は僕の卒業研究が根幹から崩れ去ります。
　でも全く同じ挙動のクラスがすでに公式で存在していたとはね…

## 読んでくれてありがとう
　こんな感じで研究中にいろいろ感じたこととか見つけたことを備忘録形式で投稿していこうかなと思います。
　間違ったことを言っていた場合は沢山訂正お願いします！

　