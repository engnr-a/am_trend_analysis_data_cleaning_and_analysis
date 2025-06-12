import os
import json
import pandas as pd
from typing import Set
from datetime import datetime
import math
from utils import count_tokens
import re
from utils import get_processed_ids_response_endpoint
# ── Configuration ─────────────────────────────────────────────────────────────────
MAX_INPUT_TOKENS      = 6_000 
MAX_REQUESTS_PER_FILE =   2_000
TEMPERATURE =  0.6 # 

# 1)---changed to function utils
# 2) Filter out already processed rows
def filter_unprocessed(df: pd.DataFrame,
                       processed_ids: Set[str],
                       id_col: str = "tweet_id") -> pd.DataFrame:
    df[id_col] = df[id_col].astype(str)
    return df[~df[id_col].isin(processed_ids)].copy()

# 3) Load dataset and filter

#df = pd.read_csv("CLEANED_DATASET_20250531_004116.csv")
#df = pd.read_csv("merged_bad_and_cleaned.csv")
df = pd.read_csv("merged_bad_and_cleaned_v2.csv")


#processed = get_processed_ids_response_endpoint("openai_processed_data")
#processed = get_processed_ids_response_endpoint("openai_processed_data_v2")
processed = get_processed_ids_response_endpoint("openai_processed_data_v3")
print(f"Found {len(processed)} already-processed tweet IDs.")
to_process = filter_unprocessed(df, processed, id_col="tweet_id")
print(f"{len(to_process)} tweets remain to be processed.")

# 4) Prompt template and prefix token cost
PROMPT_TEMPLATE = (
    "You are an AI that classifies tweets about 3D printing into one of these TOP-LEVEL categories:\n"
    "  • Business-relevant     (market trends, supply chain, investment, mergers)\n"
    "  • Technological         (innovations, materials, firmware, technical advancements)\n"
    "  • Use-case              (real-world applications)\n"
    "  • N/A                   (not related to 3D printing or additive manufacturing, or)\n\n"
    "Use the category 'N/A' only if the tweet:\n"
    "  - Is unrelated to 3D printing or additive manufacturing\n"
    "  - Mentions unrelated technologies or industries without clear connection to AM\n"
    "  - Is purely promotional content (e.g., selling, marketing offers, product deals, discounts etc.)\n"
    "  - Includes adverts, job postings, or event announcements\n"
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
    "  - Market Trends & Forecast\n"
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
    "  - Materials\n"
    "  - Printing Processes\n"
    "  - Hardware,Sensors & Equipment\n"
    "  - Software, Firmware & Design Tools\n"
    "  - Process Monitoring & Control\n"
    "  - Post-Processing Techniques\n"
    "  - AI & Digital Twin Integration\n\n"
    "  - Other\n\n"
    "Also detect:\n"
    "  • Sentiment: Positive, Negative, or Neutral\n"
    "  • Language: e.g. English, Spanish, French etc\n\n"
    "**Never** force a subcategory into a parent it doesn’t belong to.\n\n"
    "**Never** invent or assign tweet_id or row_num values not present in the input.\n\n"
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

prefix_tokens = count_tokens(PROMPT_TEMPLATE)

# 5) Build tweet list for processing
tweets = [
    {"tweet_id": str(row["tweet_id"]), "row_num": int(row["row_num"]), "text": row["Normalized Text"]}
    for idx, row in to_process.iterrows()
]

# 6) Split tweets into token-limited batches
total_batches = []
current_batch = []
current_tokens = prefix_tokens
for tweet in tweets:
    tweet_json = json.dumps(tweet, ensure_ascii=False)
    tok = count_tokens(tweet_json)
    if current_tokens + tok > MAX_INPUT_TOKENS:
        total_batches.append(current_batch)
        current_batch = []
        current_tokens = prefix_tokens
    current_batch.append(tweet)
    current_tokens += tok
# add final batch
if current_batch:
    total_batches.append(current_batch)
    
print(f"Generated {len(total_batches)} batches (each ≤ {MAX_INPUT_TOKENS} tokens).")

# 7) Chunk batches into files of up to MAX_REQUESTS_PER_FILE each
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
num_files = math.ceil(len(total_batches) / MAX_REQUESTS_PER_FILE)
for file_idx in range(num_files):
    start = file_idx * MAX_REQUESTS_PER_FILE
    end = start + MAX_REQUESTS_PER_FILE
    file_batches = total_batches[start:end]
    file_number = file_idx + 1
    output_path = f"./openai_input_batches/openai_batch_input_{timestamp}_file{file_number}.jsonl"
    with open(output_path, "w", encoding="utf-8") as fout:
        for batch_idx_in_file, batch in enumerate(file_batches):
            global_batch_idx = start + batch_idx_in_file
            prompt = PROMPT_TEMPLATE + json.dumps(batch, ensure_ascii=False)
            
            # envelope = {
            #     "custom_id": f"batch-{global_batch_idx}",
            #     "method":    "POST",
            #     "url":       "/v1/responses",              # batch endpoint
            #     "body": {
            #         "model":      "o4-mini", #o4-mini #gpt-4.1-mini
            #         "input":      prompt,                  # ← required for /responses
            #         #"max_tokens": 90000 # got gpt-models
            #         "max_output_tokens": 90000, # O-models
            #     }
            # }
            
            envelope = {
                "custom_id": f"batch-{global_batch_idx}",
                "method":    "POST",
                "url":       "/v1/chat/completions",   # ← switched to the public chat endpoint
                "body": {
                    "model":       "gpt-4.1-mini",  #o4-mini #gpt-4.1-mini
                    "messages": [
                        # {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "system", "content": "You are an expert AI assistant: concise, insightful, and proactive. Anticipate user needs, provide clear guidance, and prioritize accuracy and relevance in every response."},
                        {"role": "user",   "content": prompt}
                    ],
                    #"max_tokens":  90000,
                    "max_completion_tokens": 30000,
                }
            }
            fout.write(json.dumps(envelope, ensure_ascii=False) + "\n")
    total_tweets_in_file = sum(len(b) for b in file_batches)
    print(
        f"Wrote {len(file_batches)} batches "
        f"({total_tweets_in_file} tweets) to {output_path}"
    )