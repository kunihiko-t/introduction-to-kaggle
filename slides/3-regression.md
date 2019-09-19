<!-- sectionTitle: 機械学習の基礎:回帰 -->

## 回帰と分類

機械学習のアルゴリズムは2種類に分けられる。

値を予測する*回帰*と、入力値を識別する*分類*


---

##  線形回帰

一番簡単な回帰のアルゴリズム

入力された値から予測値を返す

例えば間取りを入力して住宅の予測価格を出力など

---

|  平均部屋数  |  住宅価格 単位：$1,000  |
| ---- | ---- |
| 6.575 |24.0 |
| 6.421 | 21.6 |
| 7.185 | 34.7 |
| 6.998 | 33.4 |
| 7.147 | 36.2 |
| 6.430 | 28.7 |
| 6.012	| 22.9 |
| 6.172 | 27.1 |
| 5.631	| 16.5 |
| 6.004 | 18.9 |

---

<!-- classes: fullscreen -->

![グラフ](https://cl.ly/83558118d6fc/%2525E3%252582%2525B9%2525E3%252582%2525AF%2525E3%252583%2525AA%2525E3%252583%2525BC%2525E3%252583%2525B3%2525E3%252582%2525B7%2525E3%252583%2525A7%2525E3%252583%252583%2525E3%252583%252588%2525202019-09-08%25252012.37.03.png)

---

## どうやって予測する？

---

こうやって線引けばなんとなく予測できそう


![グラフ](https://cl.ly/811ea9643683/%2525E3%252582%2525B9%2525E3%252582%2525AF%2525E3%252583%2525AA%2525E3%252583%2525BC%2525E3%252583%2525B3%2525E3%252582%2525B7%2525E3%252583%2525A7%2525E3%252583%252583%2525E3%252583%252588%2525202019-09-08%25252013.02.20.png)


---

## どうやって線を引くか
直線なので切片と傾きがあれば描画できそう

---

$$
x: 平均部屋数
$$

$$
y: ~~~~~~~~~~~~ 価格
$$
<br/>
$$
\huge h_\theta(x) = \theta_0 + \theta_1x
$$
<br/>

$$
h_\theta(x) ~~~~~~ 予測値
$$


$$
\theta_0 ~~~~~~~~~~~~~~~ 切片
$$

$$
\theta_1x ~~~~~~~~~~~~~ 傾き
$$
<br/>

### 仮説関数と呼ぶ

---

### つまり
<br/>
$$
\huge h_\theta(x) - y
$$
<br/>

### を最小化する
<br/>

$$
\huge \theta_0,\theta_1
$$

<br/>

### を求めれば良さそう

---

### m個のサンプルがあるのでこうなる

<br/><br/>

$$
\huge \frac{1}{2m}\sum_{i = 1}^{m}(h_\theta(x^{(i)}) - y^{(i)} )^2
$$

<br/><br/>

$$
x^{(i)} ~~ i番目のサンプルの値
$$

$$
\frac{1}{2} ~~ 計算しやすくするため掛けてるだけ
$$
<br/>

### 誤差を二乗しているので二乗誤差という(平均二乗誤差)

---

## これを目的関数(cost function)と呼ぶ

<br/>

$$
\huge J(\theta_0, \theta_1) = \frac{1}{2m}\sum_{i = 1}^{m}(h_\theta(x^{(i)}) - y^{(i)} )^2
$$

<br/>
<br/>

### 二乗誤差は線形回帰問題で効果的なのでよく使われる

---

$$
Hypothesis:
$$

<br/>

$$
\large h_\theta(x) = \theta_0 + \theta_1x
$$

<br/>
<br/>

$$
Parameters: 
$$

<br/>

$$
\large \theta_0,\theta_1
$$

<br/>
<br/>

$$
Const Function:
$$

<br/>

$$
\large J(\theta_0, \theta_1) = \frac{1}{2m}\sum_{i = 1}^{m}(h_\theta(x^{(i)}) - y^{(i)} )^2
$$

<br/>
<br/>

$$
Goal:
$$

<br/>

$$
\large minimize~J(\theta_0, \theta_1)
$$


---


## 目的関数を最小化する


---


## 最急降下法(Gradient Decent)


関数を最小化するのによく使われる方法


他には確率的勾配降下法(stochastic gradient descent, SGD)とかもある

---

$$
\large J(\theta_0, \theta_1) に対してそれを最小化する
\\
\large \theta_0,\theta_1 が欲しい。
\\
最急降下法では
\\
\large \theta_0,\theta_1 に対して初期値を設定し（何でも良いが0が一般的)
\\
\large \theta_0,\theta_1 を変化させ続け最終的に
\\
\large J(\theta_0, \theta_1) を最小化させる \large \theta_0,\theta_1 を見つける
$$

---

ある点からスタートして、ひたすら下に進むのが最急降下法。局所解に陥ることもある


![Gradient Decent](https://cl.ly/45f649e8921b/%2525E3%252582%2525B9%2525E3%252582%2525AF%2525E3%252583%2525AA%2525E3%252583%2525BC%2525E3%252583%2525B3%2525E3%252582%2525B7%2525E3%252583%2525A7%2525E3%252583%252583%2525E3%252583%252588%2525202019-09-08%25252014.27.17.png)

---

線形回帰の場合はこんな形になるので局所最適解が何個も現れることはない

![Gradient Decent](https://cl.ly/5dbf61f2b0ce/%2525E3%252582%2525B9%2525E3%252582%2525AF%2525E3%252583%2525AA%2525E3%252583%2525BC%2525E3%252583%2525B3%2525E3%252582%2525B7%2525E3%252583%2525A7%2525E3%252583%252583%2525E3%252583%252588%2525202019-09-08%25252017.17.37.png)


---


## 最急降下法のアルゴリズム


$$
\large \theta_j が収束するまで
$$

<br/>

$$
\huge \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta_0 , \theta_1)
$$

<br/>

$$
\large  (for~j=0~and~j=1)
$$

<br/>
<br/>

$$
\large \alpha : 学習率(どれだけ大きく坂を降りるか)
$$

<br/>

$$
\large \alpha \frac{\partial}{\partial \theta_j}J(\theta_0 , \theta_1) : 導関数項
$$


<br/>

$$
\large temp0 := \theta_0 - \alpha \frac{\partial}{\partial \theta_0}J(\theta_0 , \theta_1)
$$

<br/>

$$
\large  temp1 := \theta_1 - \alpha \frac{\partial}{\partial \theta_1}J(\theta_0 , \theta_1)
$$

<br/>

$$
\large  \theta_0 := temp0
$$

<br/>

$$
\large  \theta_1 := temp1
$$

<br/>

$$
\large  \theta_0, \theta_1は同時に更新すること
$$

---

## 動作イメージ

$$
\large  \theta_0 = 0 として考えてみると下図のようになる
$$

<br/>

![Gradient Decent](https://cl.ly/e68d5bfc10d0/%2525E3%252582%2525B9%2525E3%252582%2525AF%2525E3%252583%2525AA%2525E3%252583%2525BC%2525E3%252583%2525B3%2525E3%252582%2525B7%2525E3%252583%2525A7%2525E3%252583%252583%2525E3%252583%252588%2525202019-09-08%25252015.06.20.png)

<br/>

導関数は二次関数の接線になるのでθ1が二次関数の右側のほうにあると導関数は正の関数となり、左側にあると負の関数となる。

そのため学習率を小さくすると小さいステップで正解に向かって進み、大きいとステップも大きくなる。

---

学習率が小さすぎると学習に時間がかかり、大きすぎると局所解まで辿り着けない

![Gradient Decent](https://cl.ly/945f0840d47b/%2525E3%252582%2525B9%2525E3%252582%2525AF%2525E3%252583%2525AA%2525E3%252583%2525BC%2525E3%252583%2525B3%2525E3%252582%2525B7%2525E3%252583%2525A7%2525E3%252583%252583%2525E3%252583%252588%2525202019-09-08%25252015.14.34.png)

---


# 線形回帰のモデルに最急降下法を適用する

---


### 1. 最急降下法のアルゴリズム

<br/>

$$
\large \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta_0 , \theta_1)
$$

<br/>
<br/>

### 2. 線形回帰のモデル

<br/>

仮説関数

<br/>

$$
\large h_\theta(x) = \theta_0 + \theta_1x
$$

<br/>

目的関数（最急降下法の導関数の部分はこれを使う 3を参照）

<br/>

$$
\large J(\theta_0, \theta_1) = \frac{1}{2m}\sum_{i = 1}^{m}(h_\theta(x^{(i)}) - y^{(i)} )^2
$$

<br/>
<br/>

### 3. 導関数はこうなる

<br/>


$$
\large \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1) = \frac{\partial}{\partial \theta_j} \frac{1}{2m}\sum_{i = 1}^{m}(h_\theta(x^{(i)}) - y^{(i)} )^2
$$

<br/>

$$
\large = \frac{\partial}{\partial \theta_j} \frac{1}{2m}\sum_{i = 1}^{m}(\theta_0 + \theta_1x^{(i)} - y^{(i)} )^2
$$

### 4. j=0のケースとj=1のケースで偏微分する

<br/>

$$
\large \theta_0 =  \frac{\partial}{\partial \theta_0} =  \frac{1}{m}\sum_{i = 1}^{m}(h_\theta(x^{(i)}) - y^{(i)} )
$$

<br/>

$$
\large \theta_1 =  \frac{\partial}{\partial \theta_1} =  \frac{1}{m}\sum_{i = 1}^{m}(h_\theta(x^{(i)}) - y^{(i)} )x^{(i)}
$$

---


## あとはこれを解けばいいけど、さすがに１から書くのもしんどいのでscikit-learnを使う。

---

## Google Colaboratoryを使おう

---

## 転置について

---

## 線形回帰で特徴量が複数あるとき


例えば身長、体重、年齢みたいな３つの特徴量を線形回帰に使いたいとすると

i番目のサンプルのj番目の特徴量をどう表すのか？

<br/>

$$

x^{(i)}= i番目のサンプル

$$

<br/>


$$
x^{(i)}_j= i番目のサンプルのj番目の特徴量
$$

--- 

## 仮説関数の変更

今まで使っていたのは

<br/>

$$
h_\theta(x) = \theta_0 + \theta_1x
$$

<br/>

これを複数の特徴量に対応させるとこうなる

<br/>

$$
h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \theta_3x_3 + ..... + \theta_nx_n
$$

<br/>

ここでx0を1とすると

$$
  x =
  \left[
    \begin{array}{rrr}
      x_0 \\
      x_1 \\
      x_2 \\
      ... \\
      x_n \\
    \end{array}
  \right]
  
  \in\mathbb{R}^{n+1}
  ,
  \theta =
  \left[
    \begin{array}{rrr}
      \theta_0 \\
      \theta_1 \\
      \theta_2 \\
      ... \\
      \theta_n \\
    \end{array}
  \right]
  
  \in\mathbb{R}^{n+1}
$$

<br/>

$$
h_\theta(x) = \theta_0x_0 + \theta_1x_1 + \theta_2x_2 + \theta_3x_3 + ..... + \theta_nx_n
$$

<br/>

$$
= \theta^Tx
$$

<br/>

Tは「転置(transpose)」と呼ぶ

