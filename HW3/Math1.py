import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 步驟 1: 定義矩陣 ---

# W (鄰接矩陣，根據問題 1)
# 修正： W[1, 2] 和 W[2, 1] (Python 索引) 應為 0。
W = np.array([
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # x1
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # x2 (已修正)
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # x3 (已修正)
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # x4
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # x5
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # x6
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],  # x7
    [0, 1, 1, 0, 0, 0, 0, 0, 1, 0],  # x8
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  # x9
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0]   # x10
])

# D (對角矩陣)
# d_i 是 W 矩陣第 i 行的總和
# 這裡的計算會自動根據修正後的 W 得到正確的 D
degrees = W.sum(axis=1)
D = np.diag(degrees)

# L (拉普拉斯矩陣)
L = D - W

# --- 步驟 2: 求解廣義特徵值問題 ---
# 求解 L * v = lambda * D * v
try:
    eigenvalues, eigenvectors = linalg.eigh(L, D)
except linalg.LinAlgError:
    print("矩陣求解出錯。嘗試使用偽逆（pinv）處理 D。")
    # 處理 D 中可能為 0 的對角線元素（雖然此例中 d_6=1 最小）
    # 為了數值穩定性，檢查度數
    safe_degrees = np.where(degrees == 0, 1e-6, degrees)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(safe_degrees))
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt
    eigenvalues, eigenvectors_sym = linalg.eigh(L_sym)
    eigenvectors = D_inv_sqrt @ eigenvectors_sym


# --- 步驟 3: 提取用於嵌入的特徵向量 ---

# 我們需要對應於第 2、3、4 小的特徵值的特徵向量
# Python 索引 1, 2, 3
Psi = eigenvectors[:, 1:4]  # Psi 是 10x3 矩陣

# 嵌入的 3D 座標 Z 就是 Psi 的行
Z = Psi

print("--- 修正後的度 (Degrees) ---")
print(degrees)
print("\n--- 廣義特徵值 (前 5 個) ---")
print(eigenvalues[:5])
print("\n--- 嵌入座標矩陣 Psi (Z = Psi) ---")
print(Z)

# --- 步驟 3.5: 驗證優化結果 (新增功能) ---

# 1. 驗證 tr(Ψ^T L Ψ)
# 理論上，這等於所選特徵值的總和 (λ_1 + λ_2 + λ_3)
trace_val = np.trace(Psi.T @ L @ Psi)
sum_eigenvals = eigenvalues[1] + eigenvalues[2] + eigenvalues[3]

print("\n--- 驗證 tr(Ψ^T L Ψ) ---")
print(f"tr(Ψ^T L Ψ) 的計算結果: {trace_val}")
print(f"特徵值之和 (λ_1+λ_2+λ_3): {sum_eigenvals}")
print(f"此值是否約等於 1.098 (1.09779...): {np.isclose(trace_val, 1.09779)}")

# 2. 驗證 Ψ^T D Ψ = I_3
# 這是廣義特徵值問題的 D-正交歸一化條件
identity_check = Psi.T @ D @ Psi

print("\n--- 驗證 Ψ^T D Ψ = I_3 ---")
print("矩陣 Ψ^T D Ψ (應為 3x3 單位矩陣):")
print(identity_check)
print("\n(注意：對角線元素應接近 1，非對角線元素應接近 0)")

# --- 步驟 4: 繪製 3D 散點圖 ---

# 設置字體以支援中文
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 嘗試微軟正黑體
    plt.rcParams['axes.unicode_minus'] = False # 正常顯示負號
except:
    try:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # 嘗試 Arial Unicode
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("未找到中文字體，繪圖標籤可能顯示為方框。")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 繪製散點 (顏色 c 現在基於修正後的 degrees)
ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], s=100, c=degrees, cmap='viridis', alpha=0.8)

# 為每個點添加標籤 (z_1 到 z_10)
labels = [f'$z_{i+1}$' for i in range(Z.shape[0])]
for i, label in enumerate(labels):
    ax.text(Z[i, 0], Z[i, 1], Z[i, 2], label, size=12,
            zorder=1, color='k', ha='center', va='center')

# 設置座標軸標籤和標題
ax.set_xlabel('Eigenvector 1', fontsize=12)
ax.set_ylabel('Eigenvector 2', fontsize=12)
ax.set_zlabel('Eigenvector 3', fontsize=12)
ax.set_title('Laplacian Eigenmaps', fontsize=16)

plt.grid(True)
plt.show()

