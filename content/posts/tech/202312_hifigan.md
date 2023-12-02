+++
title = 'Python 3.11.x で HiFi-GAN の学習を動かす'
date = 2023-12-03T00:43:29+09:00
draft = false
categories = [ "posts" ]
tags = [ "Tech", "Python" ]
+++

# TL;DR
公式リポジトリの実装が古すぎて最近の librosa や PyTorch で動かないので、影響があった2箇所を変えてあげましょう。
```diff python
    # meldataset.py:56-59
    if fmax not in mel_basis:
-       mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
+       mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
```

```diff python
    # meldataset.py:64-65
-   spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
-                     center=center, pad_mode='reflect', normalized=False, onesided=True)
+   spec = torch.view_as_real(
+           torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
+                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True))
```

# これはなんの記事
メルスペクトログラムから高品質な音声波形を生成する HiFi-GAN[^1] ですが、公式リポジトリの実装を clone するだけでは最近のバージョンの Python で動いてくれません。しかし、最近のバージョンの Python で公式リポジトリの環境を再現しようとすると、使用されている PyTorch が古すぎるために環境作成が失敗します。

librosa や PyTorch などのライブラリの仕様変更に伴って、いくつかコードを書き換える必要があるようです。

本記事では、執筆段階で最新となる Python と PyTorch の組み合わせで HiFi-GAN の学習を動かすことを目標とします。  
inference のほうは本記事で取り上げません。動かなかったらまた記事にします。

# 環境
執筆段階 (2023/12/03) で最新の version は次のとおりです。

```plain
Python: 3.11.6
PyTorch: 2.1.1
```
v3.12.x は まだ PyTorchが対応していないようなので v3.11.x 系列を対象とします。
HiFi-GAN の公式実装は 2023/12/03 時点での master ブランチに上がっているものを使用しています[^2]。

# 実装を変える
とりあえず公式実装のまま動かしてみましょう。動かし方は公式リポジトリのREADMEに書いてあるので、必要なファイルをDLしたあと、えいやでコマンドを叩きます。  
もちろんエラーが出ます。エラーメッセージはこんな感じ。

```plain
  File "/home/xxx/workspace/python/hifi-gan/meldataset.py", line 57, in mel_spectrogram
    mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: mel() takes 0 positional arguments but 5 were given

```

librosaの主要な関数 は v0.9.0 からキーワード引数で引数を渡すことを必須とするようになった[^3]ので、変数名から意図を読み取って以下のように修正してあげます。
```diff python
    # meldataset.py:56-59
    if fmax not in mel_basis:
-       mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
+       mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
```

もう一度叩いてみます。またエラーが出ます。

<details>
<summary>エラーメッセージ全文</summary>

```plain
Original Traceback (most recent call last):
  File "/home/xxx/.local/share/virtualenvs/hifi-gan-xxx/lib/python3.11/site-packages/torch/utils/data/_utils/worker.py", line 308, in _worker_loop
    data = fetcher.fetch(index)
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/xxx/.local/share/virtualenvs/hifi-gan-xxx/lib/python3.11/site-packages/torch/utils/data/_utils/fetch.py", line 51, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xxx/.local/share/virtualenvs/hifi-gan-xxx/lib/python3.11/site-packages/torch/utils/data/_utils/fetch.py", line 51, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/home/xxx/workspace/python/hifi-gan/meldataset.py", line 139, in __getitem__
    mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xxx/workspace/python/hifi-gan/meldataset.py", line 64, in mel_spectrogram
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xxx/.local/share/virtualenvs/hifi-gan-xxx/lib/python3.11/site-packages/torch/functional.py", line 650, in stft
    return _VF.stft(input, n_fft, hop_length, win_length, window,  # type: ignore[attr-defined]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: stft requires the return_complex parameter be given for real inputs, and will further require that return_complex=True in a future PyTorch release.
```
<!-- TODO: アコーディオンメニューにする -->
</details>

PyTorch の `stft` 関数が `return_complex` 引数を要求しているようです。 

PyTorch v2.1 の公式ドキュメント[^4] によると、`return_complex=True` のとき、complex tensor が `(* x N x T)` のサイズで、
`return_complex=False` のとき、 real tensor が `(* x N x T x 2)` のサイズで返ってくるようです。なるほど。

PyTorch v1.4.0 の公式ドキュメントを確認すると、HiFi-GANの実装が意図している挙動は `return_complex=False` を指定したときの挙動のようなので[^5]、 `return_complex=False` を渡してあげたいところですが、v2.1 では `return_complex` に `False` を入力するのは deprecated のようです[^4]。幸い前と同じ挙動をさせる方法が書いてあるので、それに従って修正しましょう。

```diff python
    # meldataset.py:64-65
-   spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
-                     center=center, pad_mode='reflect', normalized=False, onesided=True)
+   spec = torch.view_as_real(
+           torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
+                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True))
```

もう一度叩くと、無事実行できます。正しくデータが配置されていれば、学習が回り始めるはずです。


# まとめ
友人からの要望で書きました。PR、出してみようかな…


[^1]: HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis, Jungil Kong, Jaehyeon Kim, Jaekyoung Bae, https://arxiv.org/abs/2010.05646
[^2]: https://github.com/jik876/hifi-gan/tree/4769534d45265d52a904b850da5a622601885777
[^3]: https://librosa.org/doc/main/changelog.html#v0-9-0
[^4]: https://pytorch.org/docs/2.1/generated/torch.stft.html
[^5]: https://pytorch.org/docs/1.4.0/torch.html?highlight=stft#torch.stft