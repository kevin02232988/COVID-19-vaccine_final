import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# IMPORTANT: Please ensure these file paths are correct in your local environment.
binary_file = "predicted_binary_weighted_final.csv" # 최신 파일명으로 변경 (smote_final 제외)
three_class_file = "predicted_three_weighted_final.csv" # 최신 파일명으로 변경

# Label mappings
binary_map = {0: '부정', 1: '긍정'}
three_class_map = {0: '부정', 1: '중립', 2: '긍정'}

# ------------------------------------------------
# 1. Binary Classification Analysis (긍/부정)
# ------------------------------------------------
try:
    # 파일을 로컬 환경에서 다시 읽도록 시도합니다.
    df_binary = pd.read_csv(binary_file, encoding='utf-8')
except FileNotFoundError:
    print(f"Error: {binary_file} 파일을 찾을 수 없습니다. 파일명을 확인해 주세요.")
    df_binary = None
except UnicodeDecodeError:
    df_binary = pd.read_csv(binary_file, encoding='cp949')

if df_binary is not None:
    # Map predicted labels to text
    df_binary['predicted_sentiment'] = df_binary['predicted_label'].map(binary_map)

    # Calculate counts and proportions
    binary_counts = df_binary['predicted_sentiment'].value_counts().reindex(['긍정', '부정'])
    binary_proportions = (binary_counts / binary_counts.sum()) * 100

    print("--- Final Binary Classification Results (긍/부정) ---")
    print("Count:")
    print(binary_counts.to_string())
    print("\nProportion (%):")
    print(binary_proportions.map('{:.2f}%'.format).to_string())

    # Generate bar chart for Binary
    plt.figure(figsize=(7, 5))
    sns.barplot(x=binary_counts.index, y=binary_counts.values, palette=['green', 'red'])
    plt.title('Final Binary Classification Distribution (긍정/부정)', fontsize=14)
    plt.ylabel('Count')
    plt.xlabel('Sentiment')
    plt.tight_layout()
    plt.savefig('final_predicted_binary_distribution.png')
    plt.close()

# ------------------------------------------------
# 2. Three-Class Classification Analysis (긍/부정/중립)
# ------------------------------------------------
try:
    df_three = pd.read_csv(three_class_file, encoding='utf-8')
except FileNotFoundError:
    print(f"Error: {three_class_file} 파일을 찾을 수 없습니다. 파일명을 확인해 주세요.")
    df_three = None
except UnicodeDecodeError:
    df_three = pd.read_csv(three_class_file, encoding='cp949')


if df_three is not None:
    # Map predicted labels to text
    df_three['predicted_sentiment'] = df_three['predicted_label'].map(three_class_map)

    # Calculate counts and proportions
    three_counts = df_three['predicted_sentiment'].value_counts().reindex(['긍정', '중립', '부정'])
    three_proportions = (three_counts / three_counts.sum()) * 100

    print("\n--- Final Three-Class Classification Results (긍/부정/중립) ---")
    print("Count:")
    print(three_counts.to_string())
    print("\nProportion (%):")
    print(three_proportions.map('{:.2f}%'.format).to_string())

    # Generate bar chart for Three-Class
    plt.figure(figsize=(7, 5))
    sns.barplot(x=three_counts.index, y=three_counts.values, palette=['green', 'grey', 'red'])
    plt.title('Final Three-Class Classification Distribution (긍정/중립/부정)', fontsize=14)
    plt.ylabel('Count')
    plt.xlabel('Sentiment')
    plt.tight_layout()
    plt.savefig('final_predicted_three_class_distribution.png')
    plt.close()