import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def debug_plot():
    # 1. 读取数据 (使用你提供的路径)
    file_path = r"Z:\python_projects\map_entropy\SymbolGeneration\Agent\outputs\images\生成\experiment_convergence_zh21.01.csv"

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("❌ 找不到文件，请确认路径。")
        return

    # 2. 打印平均分，先看看数值到底是多少
    print("=== 数据统计 (Mean Scores) ===")
    print(df.groupby("round")[["Structural Accuracy", "Style Consistency"]].mean())
    print("\n=== 数据变化量 (Delta) ===")
    means = df.groupby("round")[["Structural Accuracy", "Style Consistency"]].mean()
    try:
        diff = means.diff().iloc[2]  # Round 2 - Round 1 这种逻辑需要更严谨计算，这里简单看
        print(diff)
    except:
        pass

    # 3. 数据转换
    df_long = pd.melt(df, id_vars=["landmark", "round"],
                      value_vars=["Structural Accuracy", "Style Consistency"],
                      var_name="Metric", value_name="Score")

    # 4. 绘制箱线图 (Boxplot) - 更能看出分布的变化
    sns.set_theme(style="whitegrid", font_scale=1.3)
    plt.figure(figsize=(10, 6))

    # Boxplot 能显示中位数、四分位数和异常值
    sns.boxplot(x="round", y="Score", hue="Metric", data=df_long,
                palette={"Structural Accuracy": "#D62728", "Style Consistency": "#1F77B4"})

    # 加上散点 (Stripplot) 让审稿人看到真实样本点
    sns.stripplot(x="round", y="Score", hue="Metric", data=df_long,
                  dodge=True, color="black", alpha=0.3, size=4, legend=False)

    plt.title("Distribution of Quality Scores per Iteration", fontsize=16, weight='bold')
    plt.ylim(0, 10.5)

    save_path = os.path.join(os.path.dirname(file_path), "Debug_Boxplot.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ 诊断图已保存至: {save_path}")
    print("请查看这张图，箱子的'中横线'(中位数)是否在上升？箱子的整体位置是否在变高？")


if __name__ == "__main__":
    debug_plot()