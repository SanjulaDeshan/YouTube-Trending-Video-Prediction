import pandas as pd

# Read the CSV file
df = pd.read_csv('../data/enriched_youtube_data.csv')

# Filter for only Entertainment category
entertainment_df = df[df['video_category'] == 'Entertainment']

# Display basic information about the filtered dataset
print(f"Original dataset shape: {df.shape}")
print(f"Entertainment category rows: {entertainment_df.shape}")
print(f"Percentage of Entertainment videos: {(len(entertainment_df)/len(df))*100:.2f}%")

# Display first few rows of the filtered data
print("\nFirst 5 rows of Entertainment category data:")
print(entertainment_df.head())

# Save the filtered data to a new CSV file
entertainment_df.to_csv('entertainment_videos.csv', index=False)
print(f"\nFiltered data saved to 'entertainment_videos.csv'")

# Optional: Display some basic statistics about Entertainment videos
print(f"\nBasic statistics for Entertainment videos:")
print(f"Average views: {entertainment_df['views'].mean():,.0f}")
print(f"Average likes: {entertainment_df['likes'].mean():,.0f}")
print(f"Average comments: {entertainment_df['comments'].mean():,.0f}")
print(f"Total Entertainment videos: {len(entertainment_df)}")