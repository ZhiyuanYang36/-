import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import HypergraphConv
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"#降低可靠

# ---------------------------
# 1. 数据准备与超图构建
# ---------------------------

def generate_synthetic_data():
    """
    模拟涉农企业数据，生成节点特征、超边索引和节点标签。
    返回:
        - node_features: 节点特征矩阵 (num_nodes x num_features)
        - hyperedge_index: 超边索引 (2 x num_connections)
        - labels: 节点的分类标签 (num_nodes)
    """
    num_nodes = 100  # 节点数量（企业数）
    num_features = 16  # 每个节点的特征维度
    num_hyperedges = 10  # 超边数量

    # 随机生成节点特征
    node_features = torch.rand((num_nodes, num_features))

    # 构建超边，每个超边连接 10 个节点
    hyperedges = []
    for i in range(num_hyperedges):
        edge_nodes = torch.randint(0, num_nodes, (10,))  # 随机选择 10 个节点连接成超边
        hyperedges.append(edge_nodes)

    # 转换为稀疏矩阵表示的超边索引
    hyperedge_index = []
    for e_id, nodes in enumerate(hyperedges):
        for node in nodes:
            hyperedge_index.append([e_id, node])
    hyperedge_index = torch.tensor(hyperedge_index).T  # 转置为 [2, num_connections]

    # 随机生成节点标签（0：未破产，1：破产）
    labels = torch.randint(0, 2, (num_nodes,))

    return node_features, hyperedge_index, labels

# 生成模拟数据
node_features, hyperedge_index, labels = generate_synthetic_data()

# 使用 PyG 的 Data 对象封装图数据
data = Data(x=node_features, edge_index=hyperedge_index, y=labels)

# ---------------------------
# 2. 超图神经网络模型定义
# ---------------------------

class HGNNModel(nn.Module):
    """
    超图神经网络模型，包含两层超图卷积。
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HGNNModel, self).__init__()
        self.conv1 = HypergraphConv(input_dim, hidden_dim)  # 第一层超图卷积
        self.conv2 = HypergraphConv(hidden_dim, output_dim)  # 第二层超图卷积
        self.dropout = nn.Dropout(0.5)  # Dropout 防止过拟合

    def forward(self, x, edge_index):
        """
        模型前向传播。
        参数:
            - x: 节点特征矩阵
            - edge_index: 超边索引
        返回:
            - x: 输出特征
        """
        x = self.conv1(x, edge_index)  # 第一层超图卷积
        x = F.relu(x)  # 激活函数
        x = self.dropout(x)  # Dropout
        x = self.conv2(x, edge_index)  # 第二层超图卷积
        return x

# 初始化模型
input_dim = node_features.shape[1]  # 输入特征维度
hidden_dim = 32  # 隐藏层维度
output_dim = 2  # 分类输出维度（0 或 1）
model = HGNNModel(input_dim, hidden_dim, output_dim)

# ---------------------------
# 3. 模型训练与可视化
# ---------------------------

def plot_training_progress(losses, accuracies):
    """
    绘制训练过程中的损失曲线和准确率曲线。
    """
    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(12, 6))
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Training Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Curve')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def visualize_predictions(data, predictions):
    """
    可视化预测结果的混淆矩阵。
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        data=[[sum((data.y == 0) & (predictions == 0)), sum((data.y == 0) & (predictions == 1))],
              [sum((data.y == 1) & (predictions == 0)), sum((data.y == 1) & (predictions == 1))]],
        annot=True, fmt='d', cmap='Blues',
        xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def train_model_with_logging(data, model, epochs=100, lr=0.01):
    """
    训练模型并记录损失和准确率。
    参数:
        - data: 图数据对象
        - model: HGNN 模型
        - epochs: 训练轮次
        - lr: 学习率
    返回:
        - model: 训练后的模型
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    losses = []  # 记录损失
    accuracies = []  # 记录准确率

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 前向传播
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        # 记录损失和准确率
        losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            acc = (pred == data.y).sum().item() / len(data.y)
        accuracies.append(acc)

        # 每 10 轮打印一次
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")

    # 绘制训练曲线
    plot_training_progress(losses, accuracies)

    return model

# 训练模型
trained_model = train_model_with_logging(data, model, epochs=100, lr=0.01)

# 评估模型并可视化预测结果
trained_model.eval()
with torch.no_grad():
    predictions = trained_model(data.x, data.edge_index).argmax(dim=1)
visualize_predictions(data, predictions)
