# ShowTime



〜 あなたのLTやプレゼンを面白くする 〜



ShowTimeは発表者のポーズに応じてスライドを進めたり，効果音を鳴らしてくれるツールです．カメラに写った人の骨格をOpenPose(on CPU)で高速に認識してアクションを実行します．各ポーズに紐づくアクションパターンは設定画面でお好みに設定することができます．



### Creators:

- [Taiga Miakmi](https://taigamikami.netlify.com/): `frontend`, `presentation`
- [Yuki Nakahira](https://raahii.github.io/about/): `machine learning`, `server side`



### Demo:

<iframe src="https://player.vimeo.com/video/353065781" width="640" height="358" frameborder="0" allow="autoplay; fullscreen" allowfullscreen></iframe>
<p><a href="https://vimeo.com/353065781">showtime demo</a> from <a href="https://vimeo.com/user101681771">56 piyo</a> on <a href="https://vimeo.com">Vimeo</a>.</p>



### Slide:

これは [Hack U 2019 TOKYO Student Hackathon](https://hacku.yahoo.co.jp/hacku2019tokyo/) で発表した作品です．

<script async class="speakerdeck-embed" data-id="97f75e1a8c4a43d199dd8b43623ec655" data-ratio="1.33333333333333" src="//speakerdeck.com/assets/embed.js"></script>



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



