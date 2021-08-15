import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def get_positional_encoding(max_seq_len,embed_dim):
    # 初始化一个position encoding
    # Embed_dim: 嵌入的维度
    # max_seq_len:最大的序列维度
    position_encoding = np.array([
        [pos/np.power(10000,2*i/embed_dim) for i in range(embed_dim)]
        if pos != 0 else np.zeros(embed_dim) for pos in range(max_seq_len)
    ])
    print("positional_encoding=",position_encoding)
    position_encoding[1:,0::2] = np.sin(position_encoding[1:,0::2])     # 从第1个元素起，步长为2取元素(奇数序列)，映射到sin函数上
    position_encoding[1:,1::2] = np.cos(position_encoding[1:,1::2])     # 从第2个元素起，步长为2取元素(偶数序列)，映射到cos函数上
    
    return position_encoding

positional_encoding = get_positional_encoding(max_seq_len=100,embed_dim=16)
print("positional_encoding=",positional_encoding)

#######################################################
## 热力图可视化
#######################################################
# plt.figure(figsize=(10,10))
# sns.heatmap(positional_encoding)
# plt.title("Sinusoidal Function")
# plt.xlabel("hidden dimension")
# plt.ylabel("sequence length")
# plt.show()
###########################################################
## 正弦函数可视化
###########################################################
plt.figure(figsize=(8, 5))
plt.plot(positional_encoding[1:, 1], label="dimension 1")
plt.plot(positional_encoding[1:, 2], label="dimension 2")
plt.plot(positional_encoding[1:, 3], label="dimension 3")
plt.legend()
plt.xlabel("Sequence length")
plt.ylabel("Period of Positional Encoding")
plt.show()