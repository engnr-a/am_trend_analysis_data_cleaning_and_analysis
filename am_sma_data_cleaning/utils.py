import pandas as pd
from fuzzywuzzy import fuzz
from datasketch import MinHash, MinHashLSH
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import tiktoken
import os, json, re
from typing import Set, Any,List, Dict

#################### GENERAL DATA CLEANING FUNCTIONS ####################


def normalize_post(text):
    """
    Convert the text to lowercase and remove URLs, hashtags, mentions, punctuation,
    and extra whitespace.
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    # text = re.sub(r"[@#]\S+", "", text)  # Remove hashtags and mentions
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text.strip()


def remove_spammers(df, spammer_list):
    """
    Remove rows where the 'Author ID' is in the provided spammer_list.
    """
    return df[~df["Author ID"].isin(spammer_list)]


def tokenize(text):
    text = str(text).lower().strip()

    # Basic alphanumeric tokenization
    tokens = re.findall(r"\b\w+\b", text)

    # If it's a single token and mostly non-ASCII, fall back to character bigrams
    if len(tokens) <= 1 and sum(ord(c) > 127 for c in text) > len(text) // 2:
        return set([text[i : i + 2] for i in range(len(text) - 1)])  # bigrams

    return set(tokens)


def create_minhash(tokens, num_perm=128):
    mh = MinHash(num_perm=num_perm)
    for token in tokens:
        mh.update(token.encode("utf8"))
    return mh


def create_minhash_entry(index_text_pair, num_perm):
    idx, tokens = index_text_pair
    mh = MinHash(num_perm=num_perm)
    for token in tokens:
        mh.update(token.encode("utf8"))
    return idx, mh


def find_near_duplicates_parallel(
    df, text_column="Normalized Text", num_perm=512, threshold=0.7
):
    df = df.copy()
    df["tokens"] = df[text_column].apply(tokenize)

    # Prepare data for parallel processing
    index_token_pairs = list(zip(df.index, df["tokens"]))

    print("Building MinHash signatures in parallel...")
    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(
                    create_minhash_entry, index_token_pairs, [num_perm] * len(df)
                ),
                total=len(df),
            )
        )

    minhash_dict = dict(results)

    print("Inserting MinHash into LSH...")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, mh in tqdm(minhash_dict.items(), desc="LSH Insertion"):
        lsh.insert(str(i), mh)

    print("Finding near-duplicate pairs...")
    similar_pairs = []
    for i in tqdm(df.index, total=len(df), desc="Finding near-duplicate pairs"):
        mh = minhash_dict[i]
        result = lsh.query(mh)
        dup_indices = [int(x) for x in result if int(x) != int(i)]

        for j in dup_indices:
            similar_pairs.append(
                {
                    "Index_1": i,
                    "Tweet_1": df.loc[i, text_column],
                    "Index_2": j,
                    "Tweet_2": df.loc[j, text_column],
                    "Approx_Jaccard": threshold,
                }
            )

    duplicates_df = pd.DataFrame(similar_pairs)
    duplicates_df["Pair"] = duplicates_df.apply(
        lambda row: tuple(sorted([row["Index_1"], row["Index_2"]])), axis=1
    )
    duplicates_df.drop_duplicates("Pair", inplace=True)
    duplicates_df.drop(columns="Pair", inplace=True)

    print("Number of near-duplicate pairs found:", len(duplicates_df))
    return duplicates_df


def remove_duplicates_keep_highest_engagement(
    df, duplicates_df, engagement_column="engagement"
):
    """
    Memory-efficient function to remove near-duplicate entries from a DataFrame,
    keeping only the entry with the highest engagement value from each group.

    Works with large datasets under limited RAM constraints.
    """
    import pandas as pd
    import networkx as nx

    # Process duplicates in smaller chunks to reduce memory usage
    chunk_size = 10000

    # Create a graph to identify connected components (groups of duplicates)
    G = nx.Graph()

    # Add all unique indices as nodes
    all_indices = set(duplicates_df["Index_1"]).union(set(duplicates_df["Index_2"]))
    G.add_nodes_from(all_indices)

    # Add edges between duplicate pairs in chunks
    for i in range(0, len(duplicates_df), chunk_size):
        chunk = duplicates_df.iloc[i : i + chunk_size]
        for _, row in chunk.iterrows():
            G.add_edge(row["Index_1"], row["Index_2"])

    # Find connected components (groups of duplicates)
    duplicate_groups = list(nx.connected_components(G))
    print(f"Found {len(duplicate_groups)} groups of near-duplicates")

    # Process groups in batches to reduce memory usage
    indices_to_remove = []
    batch_size = 1000  # Process this many groups at once

    for i in range(0, len(duplicate_groups), batch_size):
        batch_groups = duplicate_groups[i : i + batch_size]

        for group in batch_groups:
            group_indices = list(group)

            # Instead of loading all group data at once, just extract engagement values
            group_engagements = df.loc[group_indices, engagement_column]

            # Find the index with highest engagement
            max_engagement_idx = group_engagements.idxmax()

            # Add other indices to removal list
            group_indices.remove(max_engagement_idx)
            indices_to_remove.extend(group_indices)

        # Explicitly clear variables to free memory
        del batch_groups
        del group_engagements

    print(f"Removing {len(indices_to_remove)} duplicates")

    # Create a new dataframe with duplicates removed, this is more memory efficient
    # than creating a new dataframe with only the rows to keep
    filtered_df = df.drop(indices_to_remove)

    return filtered_df


def plot_simplified_duplicate_network(
    duplicates_df, df=None, sample_size=None, save_path=None
):
    """
    Visualizes the network of near-duplicate clusters with group coloring and a discrete legend.

    Parameters:
    -----------
    duplicates_df : DataFrame
        Output from find_near_duplicates()
    df : DataFrame, optional
        Original dataset (only needed to resolve indices)
    sample_size : int, optional
        Number of duplicate pairs to sample for faster rendering

    Returns:
    --------
    None – displays the network graph
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import matplotlib.cm as cm
    import matplotlib.patches as mpatches

    # Sample if needed
    if sample_size and sample_size < len(duplicates_df):
        viz_df = duplicates_df.sample(sample_size, random_state=42)
        print(f"Sampled {sample_size} of {len(duplicates_df)} duplicate pairs")
    else:
        viz_df = duplicates_df

    # Build graph
    G = nx.Graph()
    for _, row in viz_df.iterrows():
        G.add_edge(row["Index_1"], row["Index_2"])

    # Layout
    pos = nx.spring_layout(G, seed=42)

    # Identify components (groups)
    components = list(nx.connected_components(G))
    node_to_group = {node: i for i, comp in enumerate(components) for node in comp}
    group_ids = sorted(set(node_to_group.values()))

    # Assign each group a unique color from tab20
    cmap = cm.get_cmap("tab20", len(group_ids))
    group_colors = {group_id: cmap(group_id) for group_id in group_ids}

    # Prepare plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw nodes per group (for custom legend)
    for group_id in group_ids:
        nodes_in_group = [n for n in G.nodes if node_to_group[n] == group_id]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes_in_group,
            node_color=[group_colors[group_id]] * len(nodes_in_group),
            node_size=300,
            alpha=0.9,
            ax=ax,
            label=f"Group {group_id}",
        )

    # Draw edges and labels
    nx.draw_networkx_edges(G, pos, edge_color="gray", ax=ax, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    # Add legend
    handles = [
        mpatches.Patch(color=group_colors[g], label=f"Group {g}") for g in group_ids
    ]
    ax.legend(
        handles=handles, title="Group ID", bbox_to_anchor=(1.02, 1), loc="upper left"
    )

    # Add prominent annotation about node labels
    ax.text(
        0.01,
        -0.08,
        "Each node number = index in the dataset",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color="darkslategray",
        ha="left",
    )

    # Restore plot border
    for spine in ax.spines.values():
        spine.set_visible(True)

    ax.set_title("Network of Near-Duplicate Pairs (Grouped)", fontsize=14)
    ax.axis("on")  # Show the border frame
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


def count_column_tokens(df, column_name, model_name="gpt-4o") -> int:
    """
    Count the total number of tokens in df[column_name] using the tokenizer for model_name.
    Suppresses errors for special tokens like <|endoftext|>.
    """
    import tiktoken

    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    texts = df[column_name].fillna("").astype(str)

    # Suppress special token errors by allowing all special tokens
    token_counts = texts.apply(lambda txt: len(enc.encode(txt, disallowed_special=())))

    return int(token_counts.sum())


def count_tokens(text: str, model_name: str = "gpt-4o-mini") -> int:
    """
    Count tokens for a single string using tiktoken.
    """
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    # allow all special tokens
    return len(enc.encode(text, disallowed_special=()))


def get_processed_ids_response_endpoint(dir_path: str) -> Set[str]:
    processed = set()

    for fname in os.listdir(dir_path):
        if not (fname.endswith(".json") or fname.endswith(".jsonl")):
            continue

        full_path = os.path.join(dir_path, fname)

        # 1) Load the file
        try:
            if fname.endswith(".json") and not fname.endswith(".jsonl"):
                # whole-file JSON (could be a list or a single dict)
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # normalize to a list of objects
                records = data if isinstance(data, list) else [data]

            else:
                # .jsonl — one JSON obj per line
                records = []
                with open(full_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            # skip malformed lines
                            continue

        except Exception as e:
            print(f"Warning: could not parse {full_path}: {e}")
            continue

        # 2) Walk each record
        for obj in records:
            # A) Top-level tweet_id?
            if isinstance(obj, dict) and "tweet_id" in obj:
                processed.add(str(obj["tweet_id"]))
                continue

            # B) Drill into response.body
            body = obj.get("response", {}).get("body", {})

            # try both choices and output
            for key in ("choices", "output"):
                items = body.get(key) or []
                for entry in items:
                    # If it’s a chat-style choice
                    if "message" in entry:
                        content = entry["message"].get("content", "")
                    else:
                        # If it’s the “output” style
                        # look for content → list of {text: ...}
                        parts = entry.get("content", [])
                        if isinstance(parts, list):
                            content = "".join(
                                p.get("text", "") for p in parts if isinstance(p, dict)
                            )
                        else:
                            content = str(parts)

                    # regex-scan for tweet_id inside the JSON blob
                    for m in re.finditer(r'"tweet_id"\s*:\s*"([^"]+)"', content):
                        processed.add(m.group(1))

    return processed

import os, json, re
from typing import List, Dict
import pandas as pd


def get_processed_metadata(dir_path: str) -> pd.DataFrame:
    """
    Walk `dir_path`, read every *.json/.jsonl file, and return a DataFrame with
    tweet-level metadata.

    Columns
    -------
    tweet_id      : str
    row_num       : int   (original value if supplied, otherwise running index)
    category      : str | None
    subcategory   : str | None
    sentiment     : str | None
    tweet_language: str | None
    """
    rows: List[Dict] = []
    auto_index = 0                     # only used when row_num is missing

    # ---------------- helper ----------------------------------------------
    def _append_row(rec: Dict) -> None:
        nonlocal auto_index
        if not rec.get("tweet_id"):
            return                     # tweet_id is mandatory

        rows.append(
            {
                "tweet_id": rec["tweet_id"],
                "row_num": rec.get("row_num", auto_index),
                "category": (
                    rec.get("category") or rec.get("classification")
                ),
                "subcategory": (
                    rec.get("subcategory") or rec.get("subclassification")
                ),
                "sentiment": rec.get("sentiment"),
                "tweet_language": rec.get("tweet_language"),
            }
        )
        auto_index += 1

    # ---------------- main directory walk ---------------------------------
    for fname in os.listdir(dir_path):
        if not fname.endswith((".json", ".jsonl")):
            continue
        fpath = os.path.join(dir_path, fname)

        try:
            # ---- load -----------------------------------------------------
            if fname.endswith(".json") and not fname.endswith(".jsonl"):
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                records = data if isinstance(data, list) else [data]
            else:                                     # .jsonl
                records = []
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                records.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            print(f"⚠️  could not parse {fpath}: {e}")
            continue

        # ---- walk every record -------------------------------------------
        for obj in records:
            if isinstance(obj, dict) and "tweet_id" in obj:
                _append_row(obj)
                continue

            body = obj.get("response", {}).get("body", {})
            for key in ("choices", "output"):
                for entry in body.get(key, []) or []:
                    # ── extract content ------------------------------------
                    if "message" in entry:            # chat-style
                        content = entry["message"].get("content", "")
                    else:                             # "output" style
                        parts = entry.get("content", [])
                        if isinstance(parts, list):
                            content = "".join(
                                p.get("text", "") for p in parts if isinstance(p, dict)
                            )
                        else:
                            content = str(parts)

                    # ── try structured JSON first -------------------------
                    try:
                        inner = json.loads(content)
                        inner_objs = inner if isinstance(inner, list) else [inner]
                    except Exception as e:
                        print(f"⚠️  JSON parse failed in {fname}: {e}")
                        print(f"--- Content snippet ---\n{content[:200]}...\n")
                        inner_objs = []

                    for rec in inner_objs:
                        if isinstance(rec, dict):
                            _append_row(rec)
                            
                            
                    if not inner_objs:
                        for match in re.finditer(
                            r'\{[^{}]*?"tweet_id"\s*:\s*"[^"]+"[^{}]*?\}',
                            content
                        ):
                            text = match.group(0)

                            # 🔧 Fix malformed key-value pairs (colon issues, comma issues)
                            text = re.sub(r'"(\w+)"\s+"([^"]+)"', r'"\1": "\2"', text)  # colon fixer
                            cleaned = re.sub(r'("\s*:\s*"[^"]*")(?=\s*"\w+"\s*:)', r'\1,', text)  # comma fixer

                            try:
                                obj = json.loads(cleaned)
                                _append_row(obj)
                            except Exception as e:
                                tweet_id_match = re.search(r'"tweet_id"\s*:\s*"([^"]+)"', text)
                                tweet_id_info = tweet_id_match.group(1) if tweet_id_match else "unknown"
                                print(f"⚠️  Final fallback skipped object for tweet_id {tweet_id_info} in {fname}: {e}")

                            
                    # if not inner_objs:
                    #     # fallback: extract all possible JSON-like tweet objects
                    #     for match in re.finditer(
                    #         r'\{[^{}]*?"tweet_id"\s*:\s*"[^"]+"[^{}]*?\}',
                    #         content
                    #     ):
                    #         text = match.group(0)

                    #         # Patch: Add missing commas between key-value pairs if needed
                    #         # (basic heuristic: ensure it's not one big unquoted blob)
                    #         cleaned = re.sub(r'"\s*:\s*("[^"]*?")(?=\s*[^,}\s])', r'\1,', text)

                    #         try:
                    #             obj = json.loads(cleaned)
                    #             _append_row(obj)
                    #         except Exception as e:
                    #             tweet_id_match = re.search(r'"tweet_id"\s*:\s*"([^"]+)"', text)
                    #             tweet_id_info = tweet_id_match.group(1) if tweet_id_match else "unknown"
                    #             print(f"⚠️  Final fallback skipped object for tweet_id {tweet_id_info} in {fname}: {e}")


                            
                    # if not inner_objs:
                    #     # fallback: extract all JSON-like dicts individually
                    #     for match in re.finditer(
                    #         r'\{[^{}]*"tweet_id"\s*:\s*"[^"]+"[^{}]*\}',
                    #         content
                    #     ):
                    #         try:
                    #             obj = json.loads(match.group(0))
                    #             _append_row(obj)
                    #         except Exception as e:
                    #             print(f"⚠️  Skipped broken fallback object in {fname}: {e}")
                            
    # ---- build DataFrame --------------------------------------------------
    return pd.DataFrame(
        rows,
        columns=[
            "tweet_id",
            "row_num",
            "category",
            "subcategory",
            "sentiment",
            "tweet_language",
        ],
    )
