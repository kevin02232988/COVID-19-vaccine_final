import pandas as pd
import matplotlib.pyplot as plt

# Load the binary and three-class files
df_binary = pd.read_csv('predicted_binary_weighted_smote_final.csv')
df_three = pd.read_csv('predicted_three_weighted_smote_final.csv')

def analyze_and_plot_sentiment(df, file_name, label_map):
    """
    Convert 'created_at' to datetime, resample by month, calculate sentiment proportions, and plot.
    """
    # 1. Convert 'created_at' to datetime and set as index
    df['created_at'] = pd.to_datetime(df['created_at'])
    df = df.set_index('created_at')

    # 2. Resample by month and count the occurrences of each predicted_label
    sentiment_counts = df.groupby([df.index.to_period('M'), 'predicted_label']).size().unstack(fill_value=0)

    # Convert PeriodIndex back to DatetimeIndex for plotting
    sentiment_counts.index = sentiment_counts.index.to_timestamp()

    # 3. Calculate the proportion of each label per month
    sentiment_proportions = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0)

    # 4. Map the integer labels to their descriptive names
    sentiment_proportions = sentiment_proportions.rename(columns=label_map)

    # 5. Plot the results
    plt.figure(figsize=(12, 6))
    sentiment_proportions.plot(
        kind='line',
        marker='o',
        linewidth=2,
        ax=plt.gca()
    )

    # Set plot title and labels
    title = f'Sentiment Proportion Over Time ({file_name})'
    plt.title(title, fontsize=16)
    plt.xlabel('Time (Month)', fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the figure as PNG
    plot_filename = f'sentiment_over_time_{file_name}.png'
    plt.savefig(plot_filename, format='png') # Explicitly setting format to 'png'
    plt.close()

    return plot_filename

# --- Analysis for Binary File ---
binary_label_map = {0: 'Negative', 1: 'Positive'}
# The previous step already used df_binary.copy() so it's safe to use df_binary again
binary_plot_file = analyze_and_plot_sentiment(pd.read_csv('predicted_binary_weighted_smote_final.csv'), 'Binary', binary_label_map)

# --- Analysis for Three-Class File ---
three_label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
# The previous step already used df_three.copy() so it's safe to use df_three again
three_plot_file = analyze_and_plot_sentiment(pd.read_csv('predicted_three_weighted_smote_final.csv'), 'Three-Class', three_label_map)

print(f"Binary Plot saved to: {binary_plot_file}")
print(f"Three-Class Plot saved to: {three_plot_file}")