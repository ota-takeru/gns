# SPH による流体力学基礎方程式の離散化

## 1. 流体力学の基礎方程式（ラグランジュ形式）

- $\mathbf{v}$ : 流体の速度ベクトル
- $\rho$ : 流体の密度

**連続の式（質量保存）**

$$
\frac{D\rho}{Dt} = -\rho\,(\nabla \cdot \mathbf{v}) 
$$
- $\nabla \cdot \mathbf{v} = \left(\frac{ \partial v_x}{\partial x} + \frac{\partial v_y}{\partial y} + \frac{\partial v_z}{\partial z}\right)$ : 体積が膨張/収縮する速さと考えられる。
1次元で考えると、左から右に進むとき右の面が左の面より速度が速いとき体積が増える。
$$
\frac{ \partial v_x}{\partial x} \approx \frac{v_x(x+ \Delta x) - v_x(x)}{\Delta x}
$$
つまり、密度の変化率が、流体の体積変化率に比例することを表している。

**運動方程式（オイラー方程式）**

$$
\frac{D \mathbf{v}}{Dt} = -\frac{1}{\rho} \nabla p
$$
$\nabla p = \left(\frac{ \partial p}{\partial x}, \frac{\partial p}{\partial y}, \frac{\partial p}{\partial z}\right)$
左右の面に働く圧力による合力は、以下の様に表される。
$$
f_x = p(x)A - p(x+ \Delta x)A = -\left(p(x+ \Delta x) - p(x)\right)A \approx -\frac{\partial p}{\partial x} \Delta x A = -\frac{\partial p}{\partial x} \Delta V = -\frac{1}{\rho} \frac{\partial p}{\partial x} \Delta m
$$

圧力の定義は扱う流体によって異なる。例えば、水などの非圧縮性流体を近似する際(弱圧縮)は以下の式を用いる。
Tait方程式:
$$
p = B \left[ \left( \frac{\rho}{\rho_0} \right)^\gamma - 1 \right]
$$
- $B$, $\gamma$ : 物質ごとに決める定数
- $\rho_0$ : 基準密度
  
小さい密度変化に対して大きな圧力変化が生じるようにすることで、非圧縮性を近似的に表現している。


**内部エネルギーの時間発展方程式**

$$
\frac{Du}{Dt} = -\frac{p}{\rho} (\nabla \cdot \mathbf{v})
$$
圧力(力)×体積変化率(速度)＝仕事率（エネルギー変化率）を表している。

## 2. カーネル近似と粒子和による離散化

スカラー場 $A(\mathbf{r})$ について、デルタ関数を用いると

$$
A(\mathbf{r}) = \int A(\mathbf{r}')\,\delta(\mathbf{r} - \mathbf{r}')\,d\mathbf{r}'
$$

と表せる。デルタ関数の代わりにカーネル関数 $W$ を用いて近似すると、

$$
A(\mathbf{r})
\approx \int A(\mathbf{r}')\,W(\mathbf{r} - \mathbf{r}', h)\,d\mathbf{r}'
\quad
\int W(\mathbf{r} - \mathbf{r}', h)\,d\mathbf{r}' = 1
$$  

となる。カーネル関数 $W$ は次の性質を持つ。

- $W(\mathbf{r}) = W(-\mathbf{r})$（対称性）
- $W(\mathbf{r}) \ge 0$（非負性）
- $|\mathbf{r}| > kh$ のとき $W(\mathbf{r}) = 0$（有限支持）

このように、$\mathbf{r}$ 周辺の重み付き平均として関数値を近似できる。

連続体の積分を粒子の和に離散化すると

$$
A(\mathbf{r})
\approx \sum_b A(\mathbf{r}_b)\,W(\mathbf{r} - \mathbf{r}_b, h)\,\Delta V_b
\approx \sum_b A_b\, W(\mathbf{r} - \mathbf{r}_b, h)\,\frac{m_b}{\rho_b}
$$

となる。ここで $m_b$ は粒子 $b$ の質量、$\rho_b$ はその密度。

**スカラー場の勾配**

$$
\nabla A(\mathbf{r})
= \int A(\mathbf{r}')\,\nabla W(\mathbf{r} - \mathbf{r}', h)\, d\mathbf{r}'
\approx \sum_b m_b \frac{A_b}{\rho_b}\,\nabla W(\mathbf{r} - \mathbf{r}_b, h).
$$

**ベクトル場の発散**

$$
\nabla \cdot \mathbf{A}(\mathbf{r})
= \int \mathbf{A}(\mathbf{r}')\cdot \nabla W(\mathbf{r} - \mathbf{r}', h)\, d\mathbf{r}'
\approx \sum_b m_b \frac{\mathbf{A}_b}{\rho_b}\cdot \nabla W(\mathbf{r} - \mathbf{r}_b, h).
$$

これらの近似を連続体の基礎方程式に適用すると SPHの離散式と一致することを次に示す。

## 3. 基礎方程式の SPH 離散化
SPHの離散式が流体の基礎方程式の離散化であることを示す。

### 3.1 連続の式の SPH 離散化

SPH において粒子$a$における密度は以下の様に定義される。

$$
\rho_a = \sum_b m_b W(\mathbf{r}_a - \mathbf{r}_b, h_a)
$$
$m_b$は一様に定められる質量

この式を時間で微分する。

$$
\frac{d\rho_a}{dt}
= \sum_b m_b \frac{d}{dt} W(\mathbf{r}_a - \mathbf{r}_b, h_a).
$$

連鎖律（チェーンルール）より

$$
\begin{aligned}
\frac{d}{dt} W(\mathbf{r}_a - \mathbf{r}_b, h_a) &= \frac{\partial W_{ab}}{\partial \mathbf{r}_a} \cdot \frac{d\mathbf{r}_a}{dt} + \frac{\partial W_{ab}}{\partial \mathbf{r}_b} \cdot \frac{d\mathbf{r}_b}{dt} \\
&= \nabla_a W_{ab} \cdot \mathbf{v}_a - \nabla_a W_{ab} \cdot \mathbf{v}_b
\quad\left(対称性より\frac{\partial W_{ab}}{\partial \mathbf{r}_b} = -\nabla_a W_{ab}\right) \\
&= (\mathbf{v}_a - \mathbf{v}_b)\cdot \nabla_a W_{ab}.
\end{aligned}
$$

したがって

$$
\frac{d\rho_a}{dt}
= \sum_b m_b (\mathbf{v}_a - \mathbf{v}_b)\cdot \nabla_a W_{ab}
$$

この式が連続の式

$$
\frac{D\rho}{Dt} = -\rho\,(\nabla \cdot \mathbf{v})
$$

の離散化になっていることを示す。
連続体の恒等式

$$
\nabla \cdot (\rho \mathbf{v})
= \rho\,(\nabla \cdot \mathbf{v}) + \mathbf{v}\cdot \nabla \rho
$$

を用いると、

$$
\frac{D\rho}{Dt} = -\rho\,(\nabla \cdot \mathbf{v}) = \mathbf{v}\cdot \nabla \rho - \nabla \cdot (\rho \mathbf{v})
$$

スカラー場の勾配の近似を用いると、

$$
\nabla \rho(\mathbf{r}_a)
\approx \sum_b m_b \frac{\rho_b}{\rho_b}\,\nabla_a W_{ab}
= \sum_b m_b \nabla_a W_{ab}
$$

ベクトル場の発散の近似を用いると、

$$
\nabla \cdot (\rho \mathbf{v})(\mathbf{r}_a)
\approx \sum_b m_b \frac{\rho_b \mathbf{v}_b}{\rho_b}\cdot \nabla_a W_{ab}
= \sum_b m_b \mathbf{v}_b\cdot \nabla_a W_{ab}
$$

したがって
$$
\begin{aligned}
\frac{D\rho}{Dt} &= -\rho\,(\nabla \cdot \mathbf{v}) \\
&= \mathbf{v}\cdot \nabla \rho - \nabla \cdot (\rho \mathbf{v}) \\
&\approx \mathbf{v}_a \cdot \sum_b m_b \nabla_a W_{ab} - \sum_b m_b \mathbf{v}_b \cdot \nabla_a W_{ab} \\
&\approx \sum_b m_b \mathbf{v}_a \cdot \nabla_a W_{ab} - \sum_b m_b \mathbf{v}_b \cdot \nabla_a W_{ab} \\
&= \sum_b m_b (\mathbf{v}_a - \mathbf{v}_b)\cdot \nabla_a W_{ab}
\end{aligned}
$$

### 3.2 運動方程式（オイラー方程式）の SPH 離散化

SPH では運動方程式は以下のように表される。

$$
\frac{d\mathbf{v}_a}{dt}
= - \sum_b m_b
\left(
\frac{P_a}{\rho_a^2} + \frac{P_b}{\rho_b^2}
\right) \nabla_a W_{ab}
$$

これが、連続体のオイラー方程式

$$
\frac{D\mathbf{v}}{Dt} = -\frac{1}{\rho}\nabla p
$$

の離散化になっていることを示す。

ベクトル解析の恒等式

$$
\frac{1}{\rho}\nabla p
= \nabla\left(\frac{p}{\rho}\right) + \frac{p}{\rho^2}\nabla \rho
$$

を用いると、

$$
\frac{D\mathbf{v}}{Dt} = -\frac{1}{\rho}\nabla p
= -\nabla\left(\frac{p}{\rho}\right) - \frac{p}{\rho^2}\nabla \rho.
$$

ここでスカラー場の勾配の近似を用いると、

$$
\nabla\rho(\mathbf{r}_a)
\approx \sum_b m_b \nabla_a W_{ab},
$$

$$
\nabla\left(\frac{p}{\rho}\right)(\mathbf{r}_a)
\approx \sum_b m_b \frac{P_b}{\rho_b^2}\,\nabla_a W_{ab} \quad (P_b = p(\mathbf{r}_b))
$$

これらを代入すると、

$$
\begin{aligned}
\frac{D\mathbf{v}_a}{Dt}
&= -\nabla\left(\frac{p}{\rho}\right)_a - \frac{P_a}{\rho_a^2} (\nabla \rho)_a \\
&\approx - \sum_b m_b \frac{P_b}{\rho_b^2}\,\nabla_a W_{ab}
        - \frac{P_a}{\rho_a^2} \sum_b m_b \nabla_a W_{ab} \\
&\approx - \sum_b m_b
\left(
\frac{P_a}{\rho_a^2} + \frac{P_b}{\rho_b^2}
\right) \nabla_a W_{ab}
\end{aligned}
$$

### 3.3 内部エネルギー方程式の SPH 離散化

SPH において粒子$a$における内部エネルギーの時間発展は以下の様に表される。

$$
\frac{du_a}{dt} = \frac{P_a}{\rho_a^2} \sum_b m_b (\mathbf{v}_a - \mathbf{v}_b) \cdot \nabla_a W_{ab}
$$

これが、連続体の内部エネルギーの式

$$
\frac{Du}{Dt} = -\frac{p}{\rho} (\nabla \cdot \mathbf{v})
$$

の離散化になっていることを示す。
連続の式の離散化で示したように、

$$
-\rho\,(\nabla \cdot \mathbf{v}) \approx \sum_b m_b (\mathbf{v}_a - \mathbf{v}_b)\cdot \nabla_a W_{ab}
$$

であるので、

$$
\begin{aligned}
\frac{Du_a}{Dt}
&= \frac{P_a}{\rho_a^2} (-\rho_a\,(\nabla \cdot \mathbf{v})_a) \\
&\approx \frac{P_a}{\rho_a^2} \sum_b m_b (\mathbf{v}_a - \mathbf{v}_b)\cdot \nabla_a W_{ab}
\end{aligned}
$$

## 4. 粘性項

粘性が考慮された流体では、以下の運動方程式が用いられる。
**ナビエ・ストークス方程式（非圧縮）**

$$
\frac{D \mathbf{v}}{Dt} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{v}
$$

- $\nu$ : 動粘性係数

これは先ほどのオイラーの方程式に粘性項 $\nu \nabla^2 \mathbf{v}$ が追加されたものである。

$\nu \nabla^2 \mathbf{v} = \nu \frac{\partial^2 \mathbf{v}}{\partial x^2} + \nu \frac{\partial^2 \mathbf{v}}{\partial y^2} + \nu \frac{\partial^2 \mathbf{v}}{\partial z^2}$ 

２枚の板の間に流体がある(クエット流れ)を考える。
この時、x軸方向に流体が流れるとするとせん断応力はx軸方向に働く力で以下の式で書ける。
$$
\tau = \mu \frac{du(y)}{dy}
$$
- $\mu$ : 粘性係数
- $u(y)$ : 流体の速度
つまり、速度の空間的な変化率が大きいほどせん断応力が大きくなる。

次に単位体積当たりに働く力を考える。
これは上面と下面のせん断応力の差を体積で割ったものになる。
$$
f_x \approx \frac{\tau(y+\Delta y) - \tau(y)}{\Delta y} = \frac{\partial \tau}{\partial y} = \mu \frac{\partial^2 u(y)}{\partial y^2}
$$
これは粘性項の各成分と対応している。


### 物理粘性

SPHにおける粘性項の離散化は以下の様に表される。

$$
\frac{d\mathbf{v}_a}{dt}
= 2 \nu \sum_b \frac{m_b}{\rho_b} \frac{(\mathbf{v}_b - \mathbf{v}_a)}{|\mathbf{r}_b - \mathbf{r}_a|^2}(\mathbf{r}_b - \mathbf{r}_a) \cdot \nabla_a W_{ab}
$$

これが、ナビエ・ストークス方程式の粘性項 $\nu \nabla^2 \mathbf{v}$ の離散化になっていることを示す。

$$
\begin{aligned}
\nabla^2 \mathbf{v}
&= \nabla \cdot (\nabla \mathbf{v})\\
&= \int \nabla' \mathbf{v}(\mathbf{r}') \cdot \nabla' W(\mathbf{r} - \mathbf{r}', h)\, d\mathbf{r}'
\end{aligned}
$$

$\nabla' \mathbf{v}(\mathbf{r}')$ が線形に変化すると仮定して以下のように近似する。

$$
\begin{aligned}
\nabla' \mathbf{v}(\mathbf{r}')
&\approx \frac{\mathbf{v}(\mathbf{r}') - \mathbf{v}(\mathbf{r})}{|\mathbf{r}' - \mathbf{r}|} \frac{\mathbf{r}' - \mathbf{r}}{|\mathbf{r}' - \mathbf{r}|} \\
&= \frac{\mathbf{v}(\mathbf{r}') - \mathbf{v}(\mathbf{r})}{|\mathbf{r}' - \mathbf{r}|^2} (\mathbf{r}' - \mathbf{r})
\end{aligned}
$$

元の式に代入して、

$$
\begin{aligned}
\nabla^2 \mathbf{v}(\mathbf{r})
&\approx \int \frac{\mathbf{v}(\mathbf{r}') - \mathbf{v}(\mathbf{r})}{|\mathbf{r}' - \mathbf{r}|^2} (\mathbf{r}' - \mathbf{r}) \cdot \nabla' W(\mathbf{r} - \mathbf{r}', h)\, d\mathbf{r}' \\
&\approx \sum_b m_b \frac{\mathbf{v}_b - \mathbf{v}_a}{|\mathbf{r}_b - \mathbf{r        }_a|^2} (\mathbf{r}_b - \mathbf{r}_a) \cdot \nabla_a W_{ab} \frac{1}{\rho_b}
\end{aligned}
$$

実際のラプラシアンの定義と一致するように係数を調整する。これは$\mathbf{v}_b$のテイラー展開を右辺に代入して求められる。

$$
\nu \nabla^2 \mathbf{v}(\mathbf{r}_a)
\approx 2 \nu \sum_b \frac{m_b}{\rho_b} \frac{(\mathbf{v}_b - \mathbf{v}_a)}{|\mathbf{r}_b - \mathbf{r}_a|^2}(\mathbf{r}_b - \mathbf{r}_a) \cdot \nabla_a W_{ab}
$$

### 人工粘性
人工粘性は、数値的に安定したシミュレーションを行うために導入される。
SPHにおける人工粘性項の離散化は以下の様に表される。これは２点の距離が近づいているときにのみ働く。
$$
\frac{d\mathbf{v}_a}{dt}
= - \sum_b m_b \Pi_{ab} \nabla_a W_{ab}
$$
ここで、$\Pi_{ab}$は粘性係数。圧縮の度合いを表す$\mu_{ab}$の一次項と二次項から構成される。
$$
\Pi_{ab} =
\begin{cases}
\displaystyle
\frac{-\alpha \bar c_{ab} \mu_{ab} + \beta \mu_{ab}^2}{\bar \rho_{ab}} & \text{if } (\mathbf{v}_{ab}) \cdot (\mathbf{r}_{ab}) < 0(２点の距離が近づいているとき) \\
0 & \text{otherwise}
\end{cases}
$$

- $\alpha, \beta$ : 人工粘性の強さを調整する定数  
- $\bar c_{ab}$ : 粒子$a$と$b$の音速(圧力が伝わるはやさ)        の平均、単位を合わせるために用いる
- $\bar \rho_{ab}$ : 粒子$a$と$b$の密度の平均

$\mathbf{v}_{ab} \cdot \mathbf{r}_{ab}$は以下の様に計算でき、２粒子の距離が近づいているときに負の値を取る。
$$
\frac{d|\mathbf{r}_{ab}|^2}{dt} = 2 \mathbf{r}_{ab} \cdot \frac{d\mathbf{r}_{ab}}{dt} = 2 \mathbf{r}_{ab} \cdot \mathbf{v}_{ab} 
$$

$\mu_{ab}$は以下の様に定義され、粒子がどれくらいの勢いで近づいているかを表す。
$$
\mu_{ab} = \frac{h (\mathbf{v}_{ab}) \cdot (\mathbf{r}_{ab})}{|\mathbf{r}_{ab}|^2 + \epsilon h^2} 
$$
- $\epsilon h^2$ : 分母がゼロになるのを防ぐための微小量

つぎに、人工粘性項がナビエ・ストークス方程式の粘性項の離散化であることを示す。
$\alpha$に関する項だけを考えると、
$$
\begin{aligned}
\sum_b m_b \Pi_{ab} \nabla_a W_{ab}
&\approx \sum_b m_b \frac{\alpha \bar c_{ab} \mu_{ab}}{\bar \rho_{ab}} \nabla_a W_{ab} \\
&\approx \sum_b m_b \frac{\alpha \bar c_{ab}h   }{\bar \rho_{ab}} \frac{\mathbf{v}_{ab}\cdot \mathbf{r}_{ab}}{|\mathbf{r}_{ab}|^2} \nabla_a W_{ab} \\
\end{aligned}
$$
これは物理粘性の離散化、
$$
\nu \nabla^2 \mathbf{v}(\mathbf{r}_a)
\approx 2 \nu \sum_b \frac{m_b}{\rho_b} \frac{(\mathbf{v}_b - \mathbf{v}_a)}{|\mathbf{r}_b - \mathbf{r}_a|^2}(\mathbf{r}_b - \mathbf{r}_a) \cdot \nabla_a W_{ab}
$$
と同じ形をしている。
よって、人工粘性項は有効粘性として解釈できる。

