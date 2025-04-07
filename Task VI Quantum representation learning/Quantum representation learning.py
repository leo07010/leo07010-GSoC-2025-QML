import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import zoom

# ==================================
# 1. 載入及抽樣 MNIST 資料
# ==================================
from tensorflow.keras.datasets import mnist

(x_train, y_train), _ = mnist.load_data()

# 只示範抽 200 筆
num_samples = 200
indices = np.random.choice(len(x_train), num_samples, replace=False)
x_train = x_train[indices]
y_train = y_train[indices]

# 簡單歸一化
x_train = x_train / 255.0

# ==================================
# 2. 通用嵌入函式
# ==================================
def embed_data(data_vec, params_vec, start_wire):
    """
    將 data_vec (長度D) 與 params_vec (長度D) 嵌入到 qubit。
    每 2 維用同一 qubit 的 RY, RX；若 D 為奇數，最後一維只用 RY。
    """
    D = len(data_vec)
    # qubit 數 = ceil(D/2)
    M = (D + 1) // 2

    # 第一層：資料編碼
    for i in range(M):
        idx1 = 2*i
        idx2 = 2*i + 1

        # 第 idx1 維存在 → RY
        if idx1 < D:
            angle_ry = data_vec[idx1] * np.pi + params_vec[idx1]
            qml.RY(angle_ry, wires=start_wire + i)

        # 第 idx2 維存在 → RX
        if idx2 < D:
            angle_rx = data_vec[idx2] * np.pi + params_vec[idx2]
            qml.RX(angle_rx, wires=start_wire + i)
    
    # 第二層：增加糾纏
    for i in range(M-1):
        qml.CNOT(wires=[start_wire + i, start_wire + i + 1])
    
    # 第三層：再次旋轉
    for i in range(M):
        if 2*i < D:
            qml.RY(params_vec[idx1], wires=start_wire + i)
        if 2*i+1 < D:
            qml.RX(params_vec[idx2], wires=start_wire + i)

# ==================================
# 3. 建立通用 SWAP Test 電路
# ==================================
def create_swap_test_circuit(D):
    """
    建立一個通用的 SWAP Test QNode，用於比較 2 張 (長度 D) 的資料。
    配置：
      - ancilla: wire=0
      - 第一張資料 qubits: wires=1..M
      - 第二張資料 qubits: wires=(1+M)..(1+2M-1)
    其中 M = ceil(D/2)

    QNode 輸入：
      img1, img2: 各長度 D
      params: 長度 2*D (前 D 給第一張、後 D 給第二張)
    回傳 ancilla=0 的機率。
    """
    M = (D + 1) // 2
    num_qubits = 1 + 2 * M  # ancilla + 2*M
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def swap_test(img1, img2, params):
        # 拆分參數
        half = len(params) // 2
        params1 = params[:half]
        params2 = params[half:]

        # 第一張資料放 wires=1..M
        embed_data(img1, params1, start_wire=1)
        # 第二張資料放 wires=(1+M)..(1+2M-1)
        embed_data(img2, params2, start_wire=(1 + M))

        # SWAP Test
        qml.Hadamard(wires=0)
        for i in range(M):
            qml.ctrl(qml.SWAP, control=0)([1 + i, 1 + M + i])
        qml.Hadamard(wires=0)
        
        return qml.probs(wires=0)

    return swap_test

def fidelity_from_probs(prob0):
    """
    Fidelity = 2 * P(ancilla=0) - 1
    """
    return 2 * prob0 - 1

# ==================================
# 4. 對比損失函式
# ==================================
def contrastive_loss(img1, lbl1, img2, lbl2, params, swap_test_func):
    """
    簡化的對比損失函數：
    1. 直接使用 Fidelity
    2. 使用 hinge loss 形式
    3. 避免複雜的數學運算
    """
    probs = swap_test_func(img1, img2, params)
    p0 = probs[0]
    F = fidelity_from_probs(p0)
    
    margin = 0.5
    
    if lbl1 == lbl2:
        # 相同類別：希望 F > margin
        loss = pnp.maximum(0, margin - F)
    else:
        # 不同類別：希望 F < -margin
        loss = pnp.maximum(0, F + margin)
    
    return loss

# ==================================
# 5. 訓練示範
# ==================================
D = 16  # 使用 16 維（考慮到 SWAP Test 需要的額外 qubit）

# 建立一個針對 D 維資料的 SWAP Test QNode
swap_test_circuit = create_swap_test_circuit(D)

def preprocess_image(img):
    """
    使用更多像素信息，並進行簡單的降維
    """
    # 將圖像壓縮到較小尺寸
    reshaped = img.reshape(28, 28)
    # 取中心區域（因為數字通常在中心）
    center = reshaped[7:21, 7:21]
    # 縮放到 4x4
    scaled = zoom(center, 4/14)
    # 展平並取前 D 個值
    flattened = scaled.flatten()[:D]
    # 確保值在 0-1 之間
    normalized = (flattened - flattened.min()) / (flattened.max() - flattened.min() + 1e-8)
    return normalized

# 初始化 2*D 維度參數：前 D 給第一張圖、後 D 給第二張圖
params = pnp.random.uniform(-0.1, 0.1, size=(2*D,), requires_grad=True)

# 使用 Nesterov Momentum 優化器
opt = qml.NesterovMomentumOptimizer(stepsize=0.005, momentum=0.95)  # 降低學習率，增加momentum

# 訓練超參數
num_epochs = 20  # 保持20輪
batch_size = 16   # 保持較大的批次大小

print(f"Start Training with D={D} (embedding dimension) ...\n")

for epoch in range(num_epochs):
    # 隨機抽 batch_size*2 張圖
    batch_indices = np.random.choice(num_samples, batch_size*2, replace=False)
    imgs = x_train[batch_indices]
    lbls = y_train[batch_indices]

    total_loss = 0
    for i in range(0, len(imgs), 2):
        img1 = preprocess_image(imgs[i])
        img2 = preprocess_image(imgs[i+1])
        lbl1 = lbls[i]
        lbl2 = lbls[i+1]

        def closure_fn(par_):
            return contrastive_loss(img1, lbl1, img2, lbl2, par_, swap_test_circuit)

        params, loss_val = opt.step_and_cost(closure_fn, params)
        total_loss += loss_val

    avg_loss = total_loss / batch_size
    print(f"Epoch {epoch+1}/{num_epochs}, Loss = {avg_loss:.4f}")

print("\nTraining done!")
print("Final params:", params)

# ==================================
# 6. 測試：比較多組圖片
# ==================================
def test_multiple_pairs():
    print("\n測試多組圖片對：")
    
    # 測試相同數字
    same_digit = np.where(y_train == y_train[0])[0][:2]
    img1 = preprocess_image(x_train[same_digit[0]])
    img2 = preprocess_image(x_train[same_digit[1]])
    probs = swap_test_circuit(img1, img2, params)
    f_same = fidelity_from_probs(probs[0])
    print(f"相同數字 ({y_train[same_digit[0]]}) 的 Fidelity: {f_same:.4f}")
    
    # 測試不同數字
    diff_digit_idx = np.where(y_train != y_train[0])[0][0]
    img3 = preprocess_image(x_train[diff_digit_idx])
    probs = swap_test_circuit(img1, img3, params)
    f_diff = fidelity_from_probs(probs[0])
    print(f"不同數字 ({y_train[same_digit[0]]} vs {y_train[diff_digit_idx]}) 的 Fidelity: {f_diff:.4f}")
    
    # 測試同一張圖片
    probs = swap_test_circuit(img1, img1, params)
    f_identical = fidelity_from_probs(probs[0])
    print(f"完全相同的圖片的 Fidelity: {f_identical:.4f}")

# 在訓練結束後進行多組測試
test_multiple_pairs()

# 原有的隨機測試
idx1, idx2 = np.random.choice(num_samples, 2, replace=False)
imgA = preprocess_image(x_train[idx1])
lblA = y_train[idx1]
imgB = preprocess_image(x_train[idx2])
lblB = y_train[idx2]

test_probs = swap_test_circuit(imgA, imgB, params)
p0_test = test_probs[0]
F_test = fidelity_from_probs(p0_test)

print(f"\nTest on random pair:")
print(f"  Image A idx={idx1}, label={lblA}")
print(f"  Image B idx={idx2}, label={lblB}")
print(f"  Fidelity = {F_test:.4f}")
if lblA == lblB:
    print("  (Same class -> ideally high Fidelity)")
else:
    print("  (Different class -> ideally low Fidelity)")

# 顯示實際圖像
fig, axes = plt.subplots(1, 2, figsize=(6,3))
axes[0].imshow(x_train[idx1], cmap='gray')
axes[0].set_title(f"idx={idx1}, label={lblA}")
axes[0].axis('off')
axes[1].imshow(x_train[idx2], cmap='gray')
axes[1].set_title(f"idx={idx2}, label={lblB}")
axes[1].axis('off')
plt.tight_layout()
plt.show()