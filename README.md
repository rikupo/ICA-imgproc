# ICAを用いた移動物体の認識
## ICA 独立成分分析プログラム
独立成分分析ICAを行って2枚の画像間で差分画像を作成する．  
RGB画像をグレイスケールに白色化，標準化した後に自然勾配法を使ってICAを行う．    
差分画像へ閾値処理を施し移動物体の領域を推定する．


# 実行例
### 入力画像
<img src="image/input1.png" width=15%>
<img src="image/input2.png" width=15%>

### 出力画像
2枚を足したら元の画像に戻りそうないい感じの結果になった．
<img src="image/out1.png" width=15%>
<img src="image/out2.png" width=15%>

### 閾値処理後
移動前と移動後でピクセルの輝度値が上下に分かれるため適切な閾値で分離ができる
<img src="image/Threshed.png" width=15%>
