---
title: "モデルの学習速度を上げる小手先のテクニック集（PyTorch）"
emoji: "🤖"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Python","pytorch","デバッグ"]
published: true
---

## なんか学習時間長くないか？？？
って思う瞬間、ありますよね。このあとパラメータチューニングもしなきゃいけないと考えると、気が遠くなります。
今回は、私がいろいろ調べた中で実際に学習速度が向上した方法をいくつか紹介します。（なんとな～く上がったかも、的なプラシーボかもしれませんが。）
最初の方法に関しては、完全に私の勉強不足で、おそらく常識レベルだと思いますが同じ悩みを抱えている人がいれば力になれればと思います。Zennや〇iitaなどのテック系ブログでPytorchのチュートリアルを調べてみると、最初の方法をやっていない記事もいくつか見つかったのでもしかしたら誰かを救えるかもしれません。なお、学習はすべてGPU上で行われているとします。

## 小手先のテクニックたち

### 1. `model.eval()`は`torch.no_grad()`ではない
~~いきなりレジ袋有料化しそうなことを言ってしまいましたが、~~ これ知らなかったです…
検証フェーズで`model.eval()`を実行すれば、自動的に勾配計算が無効になると思っていました。実際はこれら二つは全く別物で、`model.eval()`は`BatchNorm`や`Dropout`の挙動を検証フェーズ用に変更することが主な機能であり、勾配計算は事実上行うことができます。そのため、誤った用法ですが、`model.eval()`の後に`loss.backward()`や`optimizer.step()`を実行しても正しく（？）逆伝播が行われます。一方で`torch.no_grad()`はご存じの通り勾配計算を無効化するもので、それゆえに検証フェーズでの計算効率が向上します。
検証フェーズで`torch.no_grad()`を利用することでモデルの学習時間が短縮されます。また、誤った逆伝播が行われないためより安全といえます。

### 2. エポックごとの損失の保持は、`loss.item()`の代わりに`loss.detach()`を使う
おそらく多くの方がバッチ内の処理を以下に準ずる形で記述していると思います。

```python:train_改良前
current_loss = 0
for j, (x, t) in enumerate(train_loader):
    optimizer.zero_grad()
    x, t = x.to(device), t.to(device)
    y = model(x)
    loss = criterion(y,t)
    loss.backward()
    optimizer.step()
    current_loss += loss.item()
current_loss /= (j+1)
```

この処理では損失の計算後にスカラー値が`current_loss`に加算されていますが、ミニバッチ内で加算されるたびに`.item()`が実行されています。
`.item()`は要素が1つの`Tensor`に対して数値型に変換し、GPU上にある場合はCPUに移動する処理となっています。このデバイスの移動がメモリアクセスの観点で効率が悪いため、実行時間の増大につながります。コードの一部を改良してみます。

```diff python:train_改良後
+current_loss = torch.tensor([0.0], device=device)
for j, (x, t) in enumerate(train_loader):
    optimizer.zero_grad()
    x, t = x.to(device), t.to(device)
    y = model(x)
    loss = criterion(y,t)
    loss.backward()
    optimizer.step()
+   current_loss += loss.detach()
current_loss = (current_loss / (j+1)).item()
```

この処理では、`current_loss`をあらかじめGPU上の値として保持します。そして、ミニバッチ内では`loss.detach()`により損失を加算していきます。
`.detach()`は`Tensor`を計算グラフから切り離し、値のみをコピーして返す非破壊的メソッドになります。よって、元の数値データのみをもつGPU上の値が返されるため、for文で処理が繰り返されても高効率に損失の値を足していくことができます。for文から抜けたのち、初めて`.item()`を使用することで一度のみCPUに移動することでモデルの学習時間を若干ながら短縮できます。

### 3. `del`文 & `torch.cuda.empty_cache()`で明示的にメモリリリース
まず実装例を示します。

```diff python:train
current_loss = torch.tensor([0.0], device=device)
for j, (x, t) in enumerate(train_loader):
    optimizer.zero_grad()
    x, t = x.to(device), t.to(device)
    y = model(x)
    loss = criterion(y,t)
    loss.backward()
    optimizer.step()
+   del x, y, t
+   torch.cuda.empty_cache()
    current_loss += loss.detach()
current_loss = (current_loss / (j+1)).item()
```

増えた部分は`del x, y, t`と`torch.cuda.empty_cache()`です。これらはどちらもメモリリリースに関する命令で、`del`は与えられた変数を強制的に削除してメモリを開放します。
`torch.cuda.empty_cache()`はGPU上に残存している、計算に関係のない部分のメモリを開放します。メモリリリースはそれ自体が実行時間を短縮させるわけではありませんが、メモリが圧迫されているとデータを断片化させて保持しなければならず、これらのアクセスに時間を要します。
メモリを広く開けておくことでデータを保持しやすくし、学習時間の短縮に期待できます。

### 4. model定義においても、積極的に`torch.no_grad()`を使用する
modelの定義部分って、何回もやっているうちに作業ゲー感が強くなってしまいます。
`def __init__(self,...), super(...).__init__(...), def forward(self,x), ...`と毎回同じような書き方でガチャガチャやるので実装が固定観念化してしまいますよね。
`forward`内での順伝播処理では、多くの場合は勾配計算を必要としますがごく稀に学習可能なパラメータがない処理を書くことがあります（`Flatten`や`Pooling`などは除く）。このような処理は、勾配情報を保持したまま計算を行うとたいていの場合はとんでもない時間のロスになります。データ量の多い訓練フェーズでも無駄な勾配情報を保持するので、検証フェーズでの`torch.no_grad()`よりも影響が大きいといえます。
したがって、自作のモデルを定義する際に勾配計算が必要ない部分を記述する場合は、`torch.no_grad()`を積極的に使用するとよいでしょう。

:::message alert
2025/05/26 この部分の表現に間違いが含まれていましたので削除しました。
:::

## 世界が変わるほど劇的に高速になるわけではない

これまでいろいろ小手先のテクニックを紹介しましたが、劇的に速度が向上するわけではありません。（1つ目は劇的に向上しますが常識でした…）[以前紹介した`profiler`](https://zenn.dev/kita_no_in/articles/0b1bcc759c10c3)を利用してわかるレベルの差ですが、何エポックも繰り返して学習するとなると差は生まれてきます。
この悪あがきで誰かの時間をセーブ出来たら私としては非常にうれしいです。

## 2025/05/26 追記

テクニック4に関して、近似操作には勾配追跡が必要だということがわかりましたので、該当部分を削除しました。
混乱させてしまい申し訳ございません。
代わりにもう1つテクニックを紹介するので許してください。

### 5. `torch.cuda.amp`を使って半精度小数で学習

`torch.cuda.amp`には`GradScaler()`と呼ばれるクラスがあり、これを用いると通常の学習で使用される32 bitや64 bitの浮動小数よりも精度が低いがメモリ効率の良い16 bit小数に自動でキャストして学習を行うことができます。非常に高精細な計算を必要とするモデルでなければ、通常のニューラルネットワークの学習では性能の低下は無視できるほどです。
`torch.cuda.amp`は以下のように使用します。
```diff python
import torch.cuda.amp as amp
# モデルの学習ループの前に
+ scaler = amp.GradScaler()
# モデルの学習ループ
for epoch in (1, n_epochs + 1):
    current_loss = 0
    for j, (x, t) in enumerate(train_loader):
        optimizer.zero_grad()
        # 半精度に自動キャスト
+       with amp.autocast():
            x, t = x.to(device), t.to(device)
            y = model(x)
            loss = criterion(y,t)
        # scaler経由で逆伝播
+       scaler.scale(loss).backward()
        # scaler経由で最適化
+       scaler.step(optimizer)
        # scalerの更新
+       scaler.update()

```

また、半精度での学習を行ったのちに通常の32 bit、64 bitのデータを入力しても問題なく推論できます。
前回編集後にこの方法を知って実行してみましたが、2. ~ 5.の中ではこれが一番速度向上に寄与しました。

小手先のテクニック、みんなも試してみてね～