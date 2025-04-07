import pennylane as qml
from pennylane import numpy as np
import numpy as np_std
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ======================================
# 1. 生成滿足 Z2 x Z2 對稱性的分類資料
# ======================================
def generate_symmetric_dataset(N=500, seed=42):
    np_std.random.seed(seed)
    X = np_std.random.uniform(-1, 1, (N, 2))
    # 定義標籤：若 x1*x2 > 0 則 label = 0 (同號)，否則 label = 1 (異號)
    y = np.array([0 if x[0]*x[1] > 0 else 1 for x in X])
    return X, y

# 生成資料
X, y = generate_symmetric_dataset(500)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 畫出資料分布（方便檢查對稱性）
plt.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm", edgecolor="k")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Symmetric Dataset (Z2 x Z2)")
plt.show()

# ======================================
# 2. 定義輔助函式：從 QNode 輸出值轉換為預測概率
# ======================================
def predict_prob(expval):
    # 我們採用： p = (1 + expval) / 2 ，將 expval ∈ [-1,1] 映射至 [0,1]
    return (1 + expval) / 2

def cross_entropy_loss(y_true, y_pred):
    # 二元交叉熵損失
    return - (y_true * np.log(y_pred + 1e-8) + (1-y_true) * np.log(1 - y_pred + 1e-8))

# ======================================
# 3. 定義標準 QNN (non-equivariant)
# ======================================
dev_std = qml.device("default.qubit", wires=2)

@qml.qnode(dev_std)
def qnn_standard(x, theta):
    # 特徵編碼：將 x[0]、x[1] 分別用 RY 旋轉編碼到 qubit 0 和 qubit 1
    qml.RY(np.pi * x[0], wires=0)
    qml.RY(np.pi * x[1], wires=1)
    
    # 變分層（不強制對稱）
    qml.RY(theta[0], wires=0)
    qml.RY(theta[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(theta[2], wires=0)
    qml.RY(theta[3], wires=1)
    return qml.expval(qml.PauliZ(0))

def loss_std(theta, X, y):
    loss = 0
    for xi, yi in zip(X, y):
        expval = qnn_standard(xi, theta)
        p = predict_prob(expval)
        loss += cross_entropy_loss(yi, p)
    return loss / len(X)

# ======================================
# 4. 定義 Z2 x Z2 等變 QNN
# ======================================
# 此處設計的等變電路採用參數共享，使得對於兩個 qubit執行相同的旋轉操作
dev_eq = qml.device("default.qubit", wires=2)

@qml.qnode(dev_eq)
def qnn_equivariant(x, theta):
    # 先進行特徵編碼，直接以原始輸入編碼，但後續變分層則強制參數共享
    qml.RY(np.pi * x[0], wires=0)
    qml.RY(np.pi * x[1], wires=1)
    
    # 變分層：採用相同參數對兩個 qubit進行旋轉
    qml.RY(theta[0], wires=0)
    qml.RY(theta[0], wires=1)  # 參數共享
    qml.CNOT(wires=[0, 1])
    qml.RY(theta[1], wires=0)
    qml.RY(theta[1], wires=1)
    return qml.expval(qml.PauliZ(0))

def loss_eq(theta, X, y):
    loss = 0
    for xi, yi in zip(X, y):
        expval = qnn_equivariant(xi, theta)
        p = predict_prob(expval)
        loss += cross_entropy_loss(yi, p)
    return loss / len(X)

# ======================================
# 5. 訓練流程
# ======================================
# 初始參數：標準 QNN 有 4 個參數；等變 QNN 有 2 個參數（因為參數共享）
np_std.random.seed(42)
theta_std = np_std.random.uniform(0, np.pi, 4, requires_grad=True)
theta_eq = np_std.random.uniform(0, np.pi, 2, requires_grad=True)

opt_std = qml.GradientDescentOptimizer(stepsize=0.1)
opt_eq = qml.GradientDescentOptimizer(stepsize=0.1)

num_epochs = 50

loss_history_std = []
loss_history_eq = []

print("Training standard QNN...")
for epoch in range(num_epochs):
    theta_std, loss_val = opt_std.step_and_cost(lambda th: loss_std(th, X_train, y_train), theta_std)
    loss_history_std.append(loss_val)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss_val:.4f}")

print("\nTraining equivariant QNN...")
for epoch in range(num_epochs):
    theta_eq, loss_val = opt_eq.step_and_cost(lambda th: loss_eq(th, X_train, y_train), theta_eq)
    loss_history_eq.append(loss_val)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss_val:.4f}")

# ======================================
# 6. 測試與比較
# ======================================
def predict_std(x, theta):
    expval = qnn_standard(x, theta)
    p = predict_prob(expval)
    return 1 if p >= 0.5 else 0

def predict_eq(x, theta):
    expval = qnn_equivariant(x, theta)
    p = predict_prob(expval)
    return 1 if p >= 0.5 else 0

y_pred_std = [predict_std(x, theta_std) for x in X_test]
y_pred_eq  = [predict_eq(x, theta_eq) for x in X_test]

acc_std = accuracy_score(y_test, y_pred_std)
acc_eq = accuracy_score(y_test, y_pred_eq)

print("\nTest accuracy:")
print(f"  Standard QNN: {acc_std*100:.2f}%")
print(f"  Equivariant QNN: {acc_eq*100:.2f}%")

# 畫出訓練損失並存檔
plt.plot(loss_history_std, label="Standard QNN")
plt.plot(loss_history_eq, label="Equivariant QNN")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss Comparison")
plt.savefig("training_loss_comparison.png")  # 存成 PNG，檔案會在同一個資料夾下產生
plt.show()
