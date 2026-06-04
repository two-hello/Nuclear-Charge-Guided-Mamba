import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# -------------------------------
# 数据集与路径配置
# -------------------------------
datasets = [
    "GPS+BigBird", "GPS+Performer", "GPS+Transformer", "Ours"
]
base_dir = "roc"  # 每个数据集都有 results/predictions_for_roc.csv

colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
plt.figure(figsize=(8, 7))

# -------------------------------
# 遍历各个数据集
# -------------------------------
for i, dataset in enumerate(datasets):
    file_path = os.path.join(base_dir, dataset, "predictions_test.csv")
    if not os.path.exists(file_path):
        print(f"[Warning] 文件不存在: {file_path}")
        continue

    df = pd.read_csv(file_path)

    # 清理 NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how="any")
    if df.empty:
        print(f"[Warning] {dataset} 文件为空或全是 NaN，跳过。")
        continue

    # 自动判断任务类型
    if "pred" in df.columns and "true" in df.columns:
        # 单标签二分类任务
        y_true = df["true"].values
        y_prob = df["pred"].values

        # 若标签中出现 NaN 或常数列，跳过
        if len(np.unique(y_true)) < 2:
            print(f"[Warning] {dataset} 标签无变化（全为0或全为1），跳过。")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

    else:
        # 多标签任务
        true_cols = [c for c in df.columns if "true" in c]
        prob_cols = [c for c in df.columns if "pred" in c]

        if len(true_cols) == 0 or len(prob_cols) == 0:
            print(f"[Warning] {dataset} 缺少 true/prob 列，跳过。")
            continue

        auc_list = []
        mean_fpr = np.linspace(0, 1, 100)
        tpr_list = []

        for t_col, p_col in zip(true_cols, prob_cols):
            y_true = df[t_col].values
            y_prob = df[p_col].values

            # 去除 NaN
            mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
            y_true, y_prob = y_true[mask], y_prob[mask]

            if len(np.unique(y_true)) < 2:
                continue

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            auc_list.append(roc_auc)
            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tpr_list.append(tpr_interp)

        if len(tpr_list) == 0:
            print(f"[Warning] {dataset} 无有效标签用于ROC计算")
            continue

        mean_tpr = np.mean(tpr_list, axis=0)
        mean_auc = np.mean(auc_list)
        fpr, tpr, roc_auc = mean_fpr, mean_tpr, mean_auc

    # 绘制曲线
    plt.plot(
        fpr, tpr,
        color=colors[i],
        lw=2,
        label=f"{dataset.upper()} (AUC = {roc_auc:.3f})"
    )

# -------------------------------
# 图像美化与保存
# -------------------------------
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves across Multiple Datasets', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()

os.makedirs("roc", exist_ok=True)
plt.savefig("roc/all_datasets_roc.png", dpi=300)
plt.show()

print("✅ 所有数据集ROC曲线绘制完成，结果已保存到 roc/all_datasets_roc.png")
