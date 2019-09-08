<!-- sectionTitle: 機械学習の基礎:分類 -->

## 分類

- メールのスパム判定
- オンライン取引が詐欺かどうか
- 腫瘍が悪性か良性かどうか

など

---


## バイナリ分類(２項分類)

結果がYesかNoかなど二種類になるもの

<br/>

$$
\large y \in \{0,1\}
$$

<br/>

0: Negative Class


1: Positive Class  
 
<br/>

例えば

<br/>

$$
h_\theta(x) = \theta^Tx
$$

<br/>

とすると

<br/>

$$
If~h_\theta(x) \geq 0.5,~predict~"y=1"
$$

<br/>

$$
If~h_\theta(x) < 0.5,~predict~"y=0"
$$

<br/>
という感じにすれば二値に分類できる

---

## ロジスティック回帰分析

$$
\large 0 \leq h_\theta(x) \leq 1
$$

<br/>

$$
\large h_\theta (x) = g(\theta ^Tx)
$$

<br/>

$$
\large g(z) = \frac{1}{1+e^{-z}} : Sigmoid~function (Logistic~function)
$$

---

Sigmoid関数の形(必ず0から1の間に値が収まる)


![Sigmoid](https://cl.ly/40fdddaa949e/%2525E3%252582%2525B9%2525E3%252582%2525AF%2525E3%252583%2525AA%2525E3%252583%2525BC%2525E3%252583%2525B3%2525E3%252582%2525B7%2525E3%252583%2525A7%2525E3%252583%252583%2525E3%252583%252588%2525202019-09-08%25252020.59.04.png)
