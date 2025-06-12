import os
import json
import pandas as pd
from typing import Set
from datetime import datetime
import math
from utils import count_tokens 
import re
def get_processed_ids(dir_path: str) -> Set[str]:
    processed = set()
    for fname in os.listdir(dir_path):
        if not (fname.endswith(".json") or fname.endswith(".jsonl")):
            continue
        full = os.path.join(dir_path, fname)
        with open(full, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # 1) Top-level tweet_id?
                if isinstance(obj, dict) and "tweet_id" in obj:
                    processed.add(str(obj["tweet_id"]))
                    continue

                # 2) Nested in choices → message.content
                resp = obj.get("response", {}) \
                          .get("body", {}) \
                          .get("choices", [])
                if resp:
                    content = resp[0].get("message", {}).get("content", "")
                    # Try a quick find of tweet_id fields
                    for match in re.finditer(r'"tweet_id"\s*:\s*"([^"]+)"', content):
                        processed.add(match.group(1))
    return processed

def filter_unprocessed(df: pd.DataFrame,
                       processed_ids: Set[str],
                       id_col: str = "tweet_id") -> pd.DataFrame:
    """
    Return a subset of df where df[id_col] is NOT in processed_ids.
    """
    # Ensure tweet_id column is string for matching
    df[id_col] = df[id_col].astype(str)
    return df[~df[id_col].isin(processed_ids)].copy()


# 1) Load your dataset
df = pd.read_csv("random_sample_cleaned_dataset_20250520_144016.csv")  

#df = pd.read_csv("CLEANED_DATASET_20250526_174217.csv")


# 2) Scan the processed folder
processed = get_processed_ids("openai_processed_data")
print(f"Found {len(processed)} already-processed tweet IDs.")

# 3) Filter for the rest
to_process = filter_unprocessed(df, processed, id_col="tweet_id")
print(f"{len(to_process)} tweets remain to be processed.")


# 4) Build your combined prompt template
PROMPT_TEMPLATE = (
    "You are an AI that classifies tweets about 3D printing into one of these TOP-LEVEL categories:\n"
    "  • Business-relevant     (market trends, supply chain, investment, mergers)\n"
    "  • Technological         (innovations, materials, firmware, technical advancements)\n"
    "  • Use-case              (real-world applications)\n"
    "  • N/A                   (not related to 3D printing or additive manufacturing)\n\n"
    "Use the category 'N/A' only if the tweet:\n"
    "  - Is unrelated to 3D printing or additive manufacturing\n"
    "  - Mentions unrelated technologies or industries without clear connection to AM\n"
    "  - Is purely promotional (e.g., selling, marketing offers, product deals)\n"
    "  - Includes ads, job postings, or event announcements without substantive AM content\n"
    "  - Uses generic buzzwords without specific relevance to AM\n\n"
    "You MUST assign exactly ONE top-level category and exactly ONE subcategory per tweet.\n"
    "If the top-level category is 'N/A', then the subcategory must also be 'N/A'.\n\n"
    "Valid subcategories for each top-level category are:\n\n"
    "• Use-case:\n"
    "  - Motor Vehicles / Automotive\n"
    "  - Aerospace\n"
    "  - Industrial / Business Machines\n"
    "  - Consumer Products / Electronics\n"
    "  - Medical / Dental\n"
    "  - Academic Institutions\n"
    "  - Government / Military\n"
    "  - Architectural\n"
    "  - Power / Energy\n"
    "  - Home & DIY (Consumer / Hobbyist)\n"
    "  - Other\n\n"
    "• Business-relevant:\n"
    "  - Supply Chain, Manufacturing & Logistics\n"
    "  - Cost Models & Pricing\n"
    "  - Intellectual Property & Patents\n"
    "  - Mergers, Acquisitions & Partnerships\n"
    "  - Investment & Financing\n"
    "  - Business Models\n"
    "  - Customer Adoption & Demand Dynamics\n"
    "  - Sustainability & Circular Economy\n\n"
    "  - Other\n\n"
    "• Technological:\n"
    "  - Materials Development\n"
    "  - Printing Processes\n"
    "  - Hardware & Equipment\n"
    "  - Software & Design Tools\n"
    "  - Process Monitoring & Control\n"
    "  - Post-Processing Techniques\n"
    "  - AI & Digital Twin Integration\n\n"
    "  - Other\n\n"
    "Also detect:\n"
    "  • Sentiment: Positive, Negative, or Neutral\n"
    "  • Language: e.g. English, Spanish, French\n\n"
    "You will receive a JSON array of tweets:\n"
    "[\n"
    "  {\"tweet_id\": \"id123\", \"row_num\": 1, \"text\": \"This is a tweet\"},\n"
    "  {\"tweet_id\": \"id124\", \"row_num\": 2, \"text\": \"Another tweet\"},\n"
    "  …\n"
    "]\n\n"
    "Loop through each tweet and produce a JSON array of objects with these keys:\n"
    "  • tweet_id (string)\n"
    "  • row_num (integer)\n"
    "  • category (string): one of the TOP-LEVEL categories\n"
    "  • subcategory (string): required for all categories; must be 'N/A' if category is 'N/A'\n"
    "  • sentiment (string): Positive, Negative, or Neutral\n"
    "  • tweet_language (string): detected language\n\n"
    "Return ONLY the resulting JSON array. Do NOT include any explanations or extra text.\n\n"
    "Here is the JSON array of tweets:\n"
)

# 5) Prepare the JSON array string of all tweets
tweet_list = []
for idx, row in df.iterrows():
    tweet_list.append({
        "tweet_id": str(row.get("tweet_id", idx)),
        "row_num": int(idx),
        "text": row["Normalized Text"]
    })
tweet_list_str = json.dumps(tweet_list, ensure_ascii=False)


MODEL = "gpt-4o-mini"
MAX_TOKENS = 14000

# ── Constants ────────────────────────────────────────────────────────────────────────────────
MAX_INPUT_TOKENS = 2000
timestamp        = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path      = f"./openai_input_batches/openai_batch_input_{timestamp}.jsonl"
# ──────────────────────────────────────────────────────────────────────────────────────────────
# Pre‐compute the token cost of the fixed prompt prefix
prompt_prefix = PROMPT_TEMPLATE  # your multi‐line instruction
prefix_tokens = count_tokens(prompt_prefix)

# Prepare the raw tweet dicts
tweets = [
    {"tweet_id": str(row["tweet_id"]), "row_num": idx, "text": row["Normalized Text"]}
    for idx, row in to_process.iterrows()
]

batches = []
current_batch = []
current_tokens = prefix_tokens  # start with the prefix cost
processed_rows = 0

for tweet in tweets:
    # How many tokens will this tweet contribute (as JSON text)?
    tweet_json = json.dumps(tweet, ensure_ascii=False)
    tok = count_tokens(tweet_json)
    
    print(f"📝 Tweet {tweet['tweet_id']} adds {tok} tokens (text length: {len(tweet['text'])} chars)")

    # If adding it would overflow, seal off the current batch
    if current_tokens + tok > MAX_INPUT_TOKENS:
        print(f"🐦 Sealing batch #{len(batches)} at {current_tokens} tokens before overflow")
        batches.append(current_batch)
        current_batch = []
        current_tokens = prefix_tokens
    
    # Add tweet to current batch
    current_batch.append(tweet)
    current_tokens += tok
    processed_rows += 1


print(f"🔒 Batch #{len(batches)} complete!")
print(f"🧮  Tokens used: {current_tokens}")
print(f"🐥  Tweets in batch: {processed_rows}")

# Don’t forget the last batch
if current_batch:
    batches.append(current_batch)

# Write out one JSONL line per batch
with open(output_path, "w", encoding="utf-8") as fout:
    for batch_idx, batch in enumerate(batches):
        prompt = prompt_prefix + json.dumps(batch, ensure_ascii=False)
        envelope = {
            "custom_id": f"batch-{batch_idx}",
            "method":    "POST",
            "url":       "/v1/chat/completions",
            "body": {
                "model":      MODEL,
                "messages": [
                    {"role": "system",  "content": "You are a precise JSON-output assistant."},
                    {"role": "user",    "content": prompt}
                ],
                "max_tokens": MAX_TOKENS
            }
        }
        fout.write(json.dumps(envelope, ensure_ascii=False) + "\n")

print(f"Wrote {len(batches)} token-bounded batches to {output_path}")