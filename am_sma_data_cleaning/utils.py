import re
import pandas as pd
from fuzzywuzzy import fuzz

def normalize_post(text):
    """
    Convert the text to lowercase and remove URLs, hashtags, mentions, punctuation,
    and extra whitespace.
    """
    text = text.lower()
    text = re.sub(r'http\S+', '', text)   # Remove URLs
    text = re.sub(r'[@#]\S+', '', text)     # Remove hashtags and mentions
    text = re.sub(r'[^\w\s]', '', text)     # Remove punctuation
    return text.strip()

def remove_spammers(df, spammer_list):
    """
    Remove rows where the 'Author ID' is in the provided spammer_list.
    """
    return df[~df["Author ID"].isin(spammer_list)]

def remove_duplicates(df, similarity_threshold=90):
    """
    Remove near-duplicate posts based on fuzzy matching of the normalized text.
    Assumes that the DataFrame has a 'Normalized' column.
    Returns a DataFrame with only the highest-engagement post kept from duplicates.
    """
    unique_posts = []
    final_rows = []
    
    # Iterate through the DataFrame rows (assumed sorted by engagement descending)
    for idx, row in df.iterrows():
        text_norm = row["Normalized"]
        is_duplicate = False
        # Compare with already kept unique posts
        for unique_text in unique_posts:
            if fuzz.ratio(text_norm, unique_text) >= similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_posts.append(text_norm)
            final_rows.append(row)
    return pd.DataFrame(final_rows)
