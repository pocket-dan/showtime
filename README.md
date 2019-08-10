# ShowTime



〜 あなたのプレゼンをもっと面白くする 〜



ShowTimeは発表者のポーズに応じてスライドを進めたり，効果音を鳴らしてくれるツールです．カメラに写った人の骨格をOpenPose(on CPU)で高速に認識してアクションを実行します．各ポーズに紐づくアクションパターンは設定画面でお好みに設定することができます．

[Hack U 2019 TOKYO Student Hackathon](https://hacku.yahoo.co.jp/hacku2019tokyo/) にて最優秀賞を受賞しました．



### Demo:


<a href="https://vimeo.com/353065781"><img src="https://user-images.githubusercontent.com/13511520/62817977-dbd3b600-bb7a-11e9-9c79-aa5dd1110bdc.png"></a>



### Slide:



<a href="https://speakerdeck.com/taigamikami/showtime-hack-u"><img src="https://user-images.githubusercontent.com/13511520/62817953-41737280-bb7a-11e9-9fee-d6d4424ac101.png"></a>



### Architecture

- カメラクライアント: `/camera_client`
  - カメラから画像を読み込み，MLサーバーに送信します．返ってきたポーズ認識結果を元に，対応するアクションを実行します．
  - 使用技術: Python, Flask, Apple Script
  
- 機械学習サーバ: `/ml`
  - クライアントから受け取った画像を元にポーズを分類し，結果を返します．
    1. ポーズ認識で首・肩・肘・手首の位置を特定
    2. それらのパーツの座標 `(x, y)` を連結したベクトルを入力にポーズを分類
  - 使用技術: Python, OpenPose, Tensorflow, PyTorch, Flask
  
- 設定ページ: `/settings_client`
  - ポーズとアクションのパターンをGUIで設定します．
  - 使用技術: Nuxt.js



###  Creators:

- [Taiga Mikami](https://taigamikami.netlify.com/): `frontend`, `presentation`
- [Yuki Nakahira](https://raahii.github.io/about/): `backend`, `machine learning`
