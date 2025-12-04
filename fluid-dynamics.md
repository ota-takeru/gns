# SPH による流体力学基礎方程式の離散化

## 1. 流体力学の基礎方程式（ラグランジュ形式）

- $\mathbf{v}$ : 流体の速度ベクトル  
- $\rho$ : 流体の密度  
- $p$ : 流体の圧力  
- $\nabla \cdot \mathbf{v}$ : 単位体積当たりの流出量  

**連続の式（質量保存）**
$$
\frac{D\rho}{Dt} = -\rho\,(\nabla \cdot \mathbf{v})
$$

**運動方程式（オイラー方程式）**
$$
\frac{D \mathbf{v}}{Dt} = -\frac{1}{\rho} \nabla p
$$

**内部エネルギーの時間発展方程式**
$$
\frac{Du}{Dt} = -\frac{p}{\rho} (\nabla \cdot \mathbf{v})
$$


## 2. カーネル近似と粒子和による離散化

スカラー場 $A(\mathbf{r})$ について、デルタ関数を用いると
$$
A(\mathbf{r}) = \int A(\mathbf{r}')\,\delta(\mathbf{r} - \mathbf{r}')\,d\mathbf{r}'
$$
と書ける。デルタ関数の代わりにカーネル関数 $W$ を用いて近似すると
$$
A(\mathbf{r})
\approx \int A(\mathbf{r}') W(\mathbf{r} - \mathbf{r}', h)\, d\mathbf{r}' + O(h^2),
\qquad
\int W(\mathbf{r} - \mathbf{r}', h)\, d\mathbf{r}' = 1
$$
となり、$\mathbf{r}$ 周りの重み付き平均として近似できる。

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

これらの近似を連続体の基礎方程式に適用するとSPH法の離散式と一致することを次に示す。

## 3. 基礎方程式の SPH 離散化

粒子 $a$ の位置を $\mathbf{r}_a$、質量を $m_a$ とし、  
カーネル $W_{ab} = W(\mathbf{r}_a - \mathbf{r}_b, h_a)$、  
その勾配を $\nabla_a W_{ab} = \nabla_{\mathbf{r}_a} W(\mathbf{r}_a - \mathbf{r}_b, h_a)$ と書く。

SPH法の離散式が流体の基礎方程式の離散化であることを示す。

### 3.1 連続の式の SPH 離散化
SPHにおいて粒子$a$における密度は以下の様に定義される。
$$
\rho_a = \sum_b m_b W(\mathbf{r}_a - \mathbf{r}_b, h_a)
$$
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
&\approx \sum_b m_b \mathbf{v}_a \cdot \nabla_a W_{ab} - \sum_b m_b \mathbf{v}_b \cdot \nabla_a W_{ab}
\end{aligned}
$$


### 3.2 運動方程式（オイラー方程式）の SPH 離散化
SPHでは運動方程式は以下のように表される。
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
SPHにおいて粒子$a$における内部エネルギーの時間発展は以下の様に表される。
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
**ナビエ・ストークス方程式**
$$
\frac{D \mathbf{v}}{Dt} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{v}
$$
- $\nu$ : 動粘性係数

これは先ほどのオイラーの方程式に粘性項 $\nu \nabla^2 \mathbf{v}$ が追加されたものである。
### 物理粘性
SPH法における粘性項の離散化は以下の様に表される。
$$
\left(\frac{d\mathbf{v}_a}{dt}\right)_{visc}
= 2 \nu \sum_b \frac{m_b}{\rho_b} \frac{(\mathbf{v}_b - \mathbf{v}_a)}{|\mathbf{r}_b - \mathbf{r}_a|^2}(\mathbf{r}_b - \mathbf{r}_a) \cdot \nabla_a W_{ab}
$$

これが、ナビエ・ストークス方程式の粘性項 $\nu \nabla^2 \mathbf{v}$ の離散化になっていることを示す。

$$
\begin{aligned}
\nabla^2 \mathbf{v}
&= \nabla \cdot (\nabla \mathbf{v}) \quad 気になる \\ 
&= \int \nabla' \mathbf{v}(\mathbf{r}') \cdot \nabla' W(\mathbf{r} - \mathbf{r}', h)\, d\mathbf{r}'
\end{aligned}
$$

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
係数を調整して、
$$
\nu \nabla^2 \mathbf{v}(\mathbf{r}_a)
\approx 2 \nu \sum_b \frac{m_b}{\rho_b} \frac{(\mathbf{v}_b - \mathbf{v}_a)}{|\mathbf{r}_b - \mathbf{r}_a|^2}(\mathbf{r}_b - \mathbf{r}_a) \cdot \nabla_a W_{ab}                
$$


