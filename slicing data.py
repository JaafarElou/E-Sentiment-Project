import pandas as pd


file_path = 'training.1600000.processed.noemoticon.csv'
df = pd.read_csv(file_path)


print(f"Original dataset size: {df.shape}")


reduced_df = df.sample(n=1000000, random_state=42)


print(f"Reduced dataset size: {reduced_df.shape}")


reduced_file_path = 'Projet E-sentiment.csv'
reduced_df.to_csv(reduced_file_path, index=False)

print(f"Reduced dataset saved to {reduced_file_path}")
