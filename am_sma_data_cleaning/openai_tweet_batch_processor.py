import pandas as pd
import logging
import time
import os
import json
from tqdm import tqdm
from datetime import datetime
import openai

from am_sma_data_cleaning.utils import count_column_tokens

from am_sma_data_cleaning.API_tokens import openai_api_key
# === Configuration ===
MAX_TOKENS_PER_BATCH = 600_000  # Token context window per batch
#OPENAI_MODEL = "gpt-4.1-nano"
#PENAI_MODEL = "gpt-4.1"
OPENAI_MODEL = "gpt-4o-mini"
INPUT_CSV = "random_sample_cleaned_dataset_20250520_144016.csv"
OUTPUT_DIR = "batches"
PROCESSED_TWEETS_CSV = "processed_tweet_ids.csv"
SLEEP_BETWEEN_REQUESTS = 1.2  # seconds

# === Logging setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

openai.api_key = openai_api_key

if not openai.api_key or openai.api_key.startswith("your_api_key_here"):
    logging.error("❌ OPENAI_API_KEY is not set. Exiting.")
    exit()

# === Load Input ===
try:
    df = pd.read_csv(INPUT_CSV)
    # Expecting at least columns: tweet_id, row_num, Normalized Text
    required_cols = {"tweet_id", "row_num", "Normalized Text"}
    if not required_cols.issubset(df.columns):
        logging.error(f"❌ Input CSV must contain: {required_cols}")
        exit()
    df = df.dropna(subset=["Normalized Text"]).reset_index(drop=True)
except FileNotFoundError:
    logging.error(f"❌ File not found: {INPUT_CSV}")
    exit()

if df.empty:
    logging.error("❌ No tweets found in input file.")
    exit()

# === Load Processed IDs ===
if os.path.exists(PROCESSED_TWEETS_CSV):
    processed_ids = set(pd.read_csv(PROCESSED_TWEETS_CSV)["tweet_id"].astype(str))
    logging.info(f"Loaded {len(processed_ids)} processed tweet_ids.")
else:
    processed_ids = set()

# === Filter to Unprocessed ===
df = df[~df["tweet_id"].astype(str).isin(processed_ids)].reset_index(drop=True)
if df.empty:
    logging.info("All tweets have already been processed. Exiting.")
    exit()
logging.info(f"{len(df)} unprocessed tweets remain.")

# === Helper functions ===
def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def batch_by_token_limit(df, text_column, max_tokens, model_name):
    batches = []
    current_batch = []
    current_tokens = 0

    logging.info("Generating batches by token limit...")
    for idx, row in df.iterrows():
        row_dict = {
            "tweet_id": str(row["tweet_id"]),
            "row_num": int(row["row_num"]),
            "Normalized Text": row[text_column]
        }
        # Count tokens for this row
        row_df = pd.DataFrame([row])
        row_tokens = count_column_tokens(row_df, text_column, model_name)

        if row_tokens > max_tokens:
            logging.warning(
                f"Tweet {row['tweet_id']} (row {row['row_num']}) exceeds max token window alone ({row_tokens} > {max_tokens}). Will process as a single-item batch."
            )

        # If adding this row would exceed the token window, start new batch
        if current_tokens + row_tokens > max_tokens and current_batch:
            logging.info(f"Token window reached ({current_tokens} tokens). Starting new batch.")
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(row_dict)
        current_tokens += row_tokens

    if current_batch:
        batches.append(current_batch)
    logging.info(f"Generated {len(batches)} batches.")
    return batches

def classify_tweets(batch_rows):
    tweet_list = json.dumps([
        {"tweet_id": row["tweet_id"], "row_num": row["row_num"], "text": row["Normalized Text"]} for row in batch_rows
    ], ensure_ascii=False, indent=2)

    PROMPT_TEMPLATE = (
            "You are an AI that classifies tweets about 3D printing into one of these TOP-LEVEL categories:\n"
            "  • Business-relevant     (company news, market trends, commercial applications)\n"
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
            "• Technological:\n"
            "  - Materials Development\n"
            "  - Printing Processes\n"
            "  - Hardware & Equipment\n"
            "  - Software & Design Tools\n"
            "  - Process Monitoring & Control\n"
            "  - Post-Processing Techniques\n"
            "  - AI & Digital Twin Integration\n\n"
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


    try:
        start_time = time.time()
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        result = response.choices[0].message.content.strip()
        # pull the HTTP status code (new variable!)
        status = getattr(response, "status_code", None)
        end_time = time.time()
        logging.info("Received response from OpenAI.")
        return json.loads(result), end_time - start_time, status

    except json.JSONDecodeError as e:
        logging.error(f"❌ JSON Decode Error: {e}\nRaw Response: {result}")
        return [
            {
                "tweet_id": row["tweet_id"],
                "row_num": row["row_num"],
                "classification": "Unclassified",
                "tweet_language": "unknown"
            }
            for row in batch_rows
        ], None, None, status
    except Exception as e:
        logging.error(f"❌ Error during classification: {e}")
        return [
            {
                "tweet_id": row["tweet_id"],
                "row_num": row["row_num"],
                "classification": "Unclassified",
                "tweet_language": "unknown"
            }
            for row in batch_rows
        ], None, None, status

# === Batch Processing ===
batches = batch_by_token_limit(df, "Normalized Text", MAX_TOKENS_PER_BATCH, OPENAI_MODEL)

logging.info(f"Processing {len(df)} tweets in {len(batches)} token-limited batches...")

response_times = []
all_processed_ids = set(processed_ids)  # To accumulate new processed IDs

for batch_num, batch_rows in enumerate(tqdm(batches, desc="Classifying Batches"), start=1):
    timestamp = get_timestamp()
    
    try:
        ##----
        #model_tag = "NANO" if "nano" in OPENAI_MODEL.lower() else OPENAI_MODEL.replace(".", "_")
        model_tag = OPENAI_MODEL.replace(".", "_")
        stamp = f"{model_tag}_{timestamp}"
        

        logging.info(f"🚀 Sending batch {batch_num} ({len(batch_rows)} tweets) to OpenAI API...")
        batch_result, response_time,status = classify_tweets(batch_rows)
        
        print(f"==============status: {status}=============")

        # Save input batch to CSV (fault tolerance)
        batch_input_filename = os.path.join(OUTPUT_DIR, f"batch_{batch_num:05d}_input_{stamp}.csv")
        #----
        
        pd.DataFrame(batch_rows).to_csv(batch_input_filename, index=False)
        logging.info(f"📥 Saved input batch {batch_num} to {batch_input_filename}")


        # Save output batch to CSV (result)
        batch_output_filename = os.path.join(OUTPUT_DIR, f"batch_{batch_num:05d}_output_{stamp}.csv")
        if isinstance(batch_result, dict):
            batch_result = [batch_result]
        df_result = pd.DataFrame(batch_result)[["tweet_id", "row_num", "classification", "tweet_language"]]
        df_result.to_csv(batch_output_filename, index=False)
        logging.info(f"📤 Saved output batch {batch_num} to {batch_output_filename}")

        
        if status == 200:
        # Track processed tweet_ids
            all_processed_ids.update(df_result["tweet_id"].astype(str).tolist())
            pd.DataFrame({"tweet_id": list(all_processed_ids)}).to_csv(PROCESSED_TWEETS_CSV, index=False)
            logging.info(f"✅ Updated processed tweets CSV ({len(all_processed_ids)} processed so far)")
        else:
            logging.warning(f"⚠️⚠️⚠️⚠️⚠️⚠️ Batch {batch_num} had status {status}; not marking tweets as processed.")

        if response_time:
            response_times.append(response_time)
    except Exception as e:
         logging.error(f"Batch {batch_num} failed, will retry: {e}")

    time.sleep(SLEEP_BETWEEN_REQUESTS)

# === Summary ===
if response_times:
    avg_time = sum(response_times) / len(response_times)
    logging.info(f"✅ Done. Average response time per batch: {avg_time:.2f} seconds.")
else:
    logging.warning("⚠️ No successful batches to compute average response time.")
