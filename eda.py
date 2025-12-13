import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def perform_eda(df):
    # Drop redundant column
    df = df.drop(columns=['Unnamed: 0'])
    
    # 1. Summary Statistics for key numerical features
    summary_stats = df[['Innings Runs', 'Innings Wickets', 'Target Score', 'Runs to Get', 'Balls Remaining']].describe()
    print("1. Summary Statistics:")
    # Using to_string() to avoid the 'tabulate' dependency error
    print(summary_stats.to_string()) 
    
    # 2. Target Variable Distribution (Chased Successfully - for the whole match outcome)
    match_outcomes = df.drop_duplicates(subset=['Match ID'])
    successful_chase_count = match_outcomes['Chased Successfully'].value_counts()
    print("\n2. Successful Chase Distribution (0=Failed, 1=Succeeded):")
    print(successful_chase_count.to_string())
    
    # Create a bar plot for successful chase
    plt.figure(figsize=(6, 5))
    successful_chase_count.plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Distribution of Match Outcomes (Successful Chase)')
    plt.xlabel('Chased Successfully (0: No, 1: Yes)')
    plt.ylabel('Number of Matches')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig('eda_chase_outcome.png')
    plt.close()
    print("Saved eda_chase_outcome.png")
    
    # 3. Venue Analysis: Average First Innings Score by Venue
    first_innings_scores = df[df['Innings'] == 1].groupby('Match ID').agg({
        'Venue': 'first',
        'Innings Runs': 'max'
    }).rename(columns={'Innings Runs': 'First Innings Score'})
    
    venue_avg_score = first_innings_scores.groupby('Venue')['First Innings Score'].mean().sort_values(ascending=False).head(10)
    print("\n3. Top 10 Venues by Average First Innings Score:")
    print(venue_avg_score.to_string())
    
    # Create a bar plot for top 10 venues
    plt.figure(figsize=(10, 6))
    venue_avg_score.plot(kind='bar', color='darkgreen')
    plt.title('Top 10 Venues by Average First Innings Score')
    plt.xlabel('Venue')
    plt.ylabel('Average Score')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig('eda_venue_avg_score.png')
    plt.close()
    print("Saved eda_venue_avg_score.png")
    
    # 4. Feature Distribution: Target Score Distribution
    plt.figure(figsize=(8, 5))
    # Filter out matches where Target Score is 0 
    valid_targets = df[(df['Target Score'] > 0) & (df['Innings'] == 2)]['Target Score'].drop_duplicates()
    sns.histplot(valid_targets, bins=20, kde=True, color='purple')
    plt.title('Distribution of Target Scores in T20 Matches')
    plt.xlabel('Target Score')
    plt.ylabel('Number of Matches')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig('eda_target_score_distribution.png')
    plt.close()
    print("Saved eda_target_score_distribution.png")
    
    # 5. Correlation Analysis (Focusing on 2nd Innings features)
    # The 'Chased Successfully' column already exists in the 2nd innings data.
    df_2nd_innings = df[df['Innings'] == 2].copy()
    
    # Removed the problematic merge logic as it created a duplicate column 'Chased Successfully_final'.
    
    correlation_features = [
        'Innings Runs', 
        'Innings Wickets', 
        'Target Score', 
        'Runs to Get', 
        'Balls Remaining', 
        'Runs From Ball',
        'Bowler Runs Conceded',
        'Chased Successfully' # The target variable
    ]
    
    corr_matrix = df_2nd_innings[correlation_features].corr()
    
    # Create a heatmap for correlation
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black')
    plt.title('Correlation Matrix of Key 2nd Innings Features')
    plt.tight_layout()
    plt.savefig('eda_correlation_heatmap.png')
    plt.close()
    print("Saved eda_correlation_heatmap.png")
    
    # 6. Trend Analysis: Win Probability vs. Overs Completed (Approximation using average win rate)
    df_2nd_innings['Overs Completed'] = df_2nd_innings['Over'] + (df_2nd_innings['Ball'] / 6)
    
    win_prob_over = df_2nd_innings.groupby(pd.cut(df_2nd_innings['Overs Completed'], bins=np.arange(0, 21, 1), include_lowest=True))['Chased Successfully'].mean() * 100
    win_prob_over.index = [f'Over {int(interval.right)}' for interval in win_prob_over.index]
    
    plt.figure(figsize=(12, 6))
    win_prob_over.plot(kind='line', marker='o', color='red')
    plt.title('Approximate Win Probability for Chasing Team vs. Overs Completed')
    plt.xlabel('Over Completed')
    plt.ylabel('Average Win Percentage (%)')
    plt.xticks(rotation=45)
    plt.grid(axis='both', linestyle='--')
    plt.tight_layout()
    plt.savefig('eda_win_prob_trend.png')
    plt.close()
    print("Saved eda_win_prob_trend.png")
    
    # 7. Grouped Aggregation: Wickets fallen per over
    wickets_per_over = df[df['Wicket'] == 1].groupby('Over')['Wicket'].count()
    
    plt.figure(figsize=(10, 5))
    wickets_per_over.plot(kind='bar', color='darkred')
    plt.title('Total Wickets Fallen Per Over Across All Matches')
    plt.xlabel('Over Number')
    plt.ylabel('Total Wickets')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig('eda_wickets_per_over.png')
    plt.close()
    print("Saved eda_wickets_per_over.png")
    
    # 8. Outlier Detection: Runs from Ball (Box Plot)
    plt.figure(figsize=(8, 5))
    sns.boxplot(y=df['Runs From Ball'])
    plt.title('Box Plot of Runs From Ball')
    plt.ylabel('Runs From Ball')
    plt.tight_layout()
    plt.savefig('eda_runs_from_ball_boxplot.png')
    plt.close()
    print("Saved eda_runs_from_ball_boxplot.png")
    
    # 9. Data Types and Unique Value Counts
    print("\n9. Data Types:")
    print(df.dtypes.to_string())
    
    print("\n10. Unique Value Counts:")
    unique_counts = df.nunique()
    print(unique_counts.to_string())

if __name__ == '__main__':
    # File loading for local execution test
    try:
        df = pd.read_csv('ball_by_ball_it20.csv')
        perform_eda(df)
    except FileNotFoundError:
        print("Dataset 'ball_by_ball_it20.csv' not found.")