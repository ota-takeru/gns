# WCSPH法

## 1. 近傍探索
- 近傍 $N_f(i), N_b(i)$ を得る

## 2. 密度計算
- $W_{ij}$を用いて距離に応じて重みをつける
$$
\rho_i = \sum_{j \in N_f(i)} m_j W_{ij}
$$

## 3. 圧力計算
$$
p_i = \kappa \rho_0^\gamma \left[ \left( \frac{\rho_i}{\rho_0} \right)^\gamma - 1 \right]
$$

## 4. 圧力勾配と外力による加速度
- $a = -\frac{1}{\rho} \nabla p + g$ を離散化した式
- $\frac{m_j}{\rho_j}$：粒子 $j$ の体積  
  $\nabla W_{ij}$：粒子間の距離と向きによる重み  
  $\left( \frac{p_i}{\rho_i^2} + \frac{p_j}{\rho_j^2} \right)$は対称化
$$\nabla p(\mathbf{x}_i) \approx \sum_j \frac{m_j}{\rho_j} p_j \nabla W_{ij}$$
$$
\mathbf{a}_i = \mathbf{g} - m_i\sum_{j \in N_f(i)} m_j \left( \frac{p_i}{\rho_i^2} + \frac{p_j}{\rho_j^2} \right) \nabla W_{ij}
$$

## 5. 速度更新
- 上で求めた加速度を用いて速度を更新
$$
\mathbf{v}_i^{n+1} = \mathbf{v}_i^n + \Delta t \, \mathbf{a}_i
$$

## 6. 粘性による速度補正
- 上で求めた速度に対して粘性による速度を加える
- $\alpha$は粘性係数  
  $\frac{m_j}{\rho_j}$は体積  
  $W_{ij}$は距離に応じた重み  
  近傍粒子に対する速度の差を、距離と体積の重みで調整して加える
$$
\hat{\mathbf{v}}_i = \mathbf{v}_i^{n+1} + \alpha \sum_{j \in N_f(i)} \frac{m_j}{\rho_j} \left( \mathbf{v}_j^{n+1} - \mathbf{v}_i^{n+1} \right) W_{ij}
$$

## 7. 位置更新
- 上で求めた速度を用いて位置を更新
$$
\mathbf{x}_i^{n+1} = \mathbf{x}_i^n + \Delta t \, \hat{\mathbf{v}}_i
$$

## 8. 境界条件処理
- $\psi_k$は境界粒子 $k$ の仮想密度
$$
\rho_i = \sum_{j \in N_f(i)} m_j W_{ij} + \sum_{k \in N_b(i)} \psi_k W_{ik}
$$
$$
\psi_k = \rho_0 V_k = \rho_0 \frac{m_k}{\rho_k} = \rho_0 \frac{m_k}{\sum_l m_k W_{kl}} = \rho_0 \frac{1}{\sum_l W_{kl}}
$$