import pandas as pd
import re
from collections import Counter
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

# Fallback: A basic list of common English stopwords
basic_stopwords = set([
    'the', 'a', 'an', 'is', 'it', 'and', 'or', 'but', 'if', 'then', 'else', 'for', 'with', 'on', 'at', 'by', 'of', 'to',
    'in', 'that', 'this', 'you', 'i', 'we', 'they', 'he', 'she', 'was', 'were', 'be', 'are', 'not', 'as', 's', 'can',
    'will', 'would', 'should', 'could', 'get', 'like', 'know', 'go', 'from', 'have', 'do', 'don', 'just', 'more', 'one',
    'use', 'out', 'up', 'down', 'about', 'how', 'what', 'when', 'where', 'why', 'them', 'their', 'has', 'had', 'been',
    'my', 'your', 'her', 'his', 'its', 'our', 'their', 'also', 'many', 'much', 'see', 'may', 'new', 'time', 'first',
    'even', 'make', 'say', 'think', 'look', 'people', 'wouldnt', 'im', 'ive', 'didnt', 'doesnt', 'cant', 'couldnt'
])

# 불용어 목록 설정 (NLTK 로드 실패 시 대체)
try:
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words('english'))
    print("NLTK stopwords loaded successfully.")
except (LookupError, ImportError, NameError):
    stop_words = basic_stopwords
    print("Using basic custom stopwords list.")

# 데이터 로드 (파일 경로 수정 필요 시 여기에 적용)
# 현재는 실행 경로에 파일이 있다고 가정합니다.
df = pd.read_csv("FINAL_DATA_FILTERED_#TRUE.csv")


# ----------------------------------------------------------------------
# 1. 단어 빈도수 분석 및 그래프 (Word Frequency Analysis and Chart)
# ----------------------------------------------------------------------

# 전처리 함수
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return [word for word in tokens if word not in stop_words and len(word) > 2]


# 전처리 적용 및 단어 목록 생성
df['tokens'] = df['text'].apply(preprocess_text)
all_words = [word for tokens_list in df['tokens'] for word in tokens_list]
word_counts = Counter(all_words)
top_n = 20
top_words = pd.DataFrame(word_counts.most_common(top_n), columns=['word', 'count'])

# 막대 그래프 생성 (Seaborn Warning 수정)
plt.figure(figsize=(12, 6))
# 'y' 변수('word')를 'hue'로 지정하고 'legend=False' 추가
sns.barplot(x='count', y='word', data=top_words, palette='viridis', hue='word', legend=False)
plt.title(f'Top {top_n} Frequently Appearing Words in Comments (Excluding Stopwords)', fontsize=15)
plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Word', fontsize=12)
plt.tight_layout()
plt.savefig('top_words_frequency.png')
plt.close()

# ----------------------------------------------------------------------
# 2. 시간에 따른 단어 빈도 변화 분석 (Word Frequency Change Over Time)
# ----------------------------------------------------------------------

df['created_at'] = pd.to_datetime(df['created_at'])
df_time = df.set_index('created_at')

words_to_track = top_words['word'].head(5).tolist()
time_series_data = {}

for word in words_to_track:
    def count_word_monthly(group):
        return sum(tokens.count(word) for tokens in group)


    # Resample Warning 수정: 'M' 대신 'ME' 사용
    monthly_counts = df_time['tokens'].resample('ME').apply(count_word_monthly)
    monthly_total_words = df_time['tokens'].resample('ME').apply(
        lambda x: sum(len(tokens) for tokens in x))  # Resample Warning 수정: 'M' 대신 'ME' 사용

    monthly_relative_frequency = (monthly_counts / (monthly_total_words + 1e-6)) * 1000
    time_series_data[word] = monthly_relative_frequency

# 결과를 DataFrame으로 변환
ts_df = pd.DataFrame(time_series_data)
ts_df.index.name = 'Month'
ts_df.columns.name = 'Word'

# 꺾은선 그래프 생성
plt.figure(figsize=(14, 7))
for word in words_to_track:
    data_to_plot = ts_df[ts_df[word] > 0][word]
    plt.plot(data_to_plot.index, data_to_plot.values, label=word, marker='o', markersize=3)

plt.title(f'Relative Frequency of Top 5 Words Over Time (Per 1000 Words)', fontsize=15)
plt.xlabel('Date (Month)', fontsize=12)
plt.ylabel('Relative Frequency (Count per 1000 words)', fontsize=12)
plt.legend(title='Word', loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('word_frequency_over_time.png')
plt.close()

# 월별 시계열 데이터 CSV 저장
ts_df.to_csv('monthly_word_frequency_ts.csv')