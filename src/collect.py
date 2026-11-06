"""
Stage 1: corpus collection.

Requests 200 PubMed abstracts per topic (5 topics; 981 usable abstracts in this study) using the
Biopython Entrez API, deduplicates by PMID, and saves to data/pubmed_corpus.csv.

Usage:
    python src/collect.py
"""

import os
import time
import pandas as pd
from Bio import Entrez
from dotenv import load_dotenv

load_dotenv()

Entrez.email = os.getenv("ENTREZ_EMAIL", "researcher@example.com")
_ncbi_key = os.getenv("NCBI_API_KEY")
if _ncbi_key:
    Entrez.api_key = _ncbi_key

TOPICS = {
    "hypertension": "hypertension[MeSH] AND (treatment OR pathophysiology OR epidemiology)",
    "diabetes": "type 2 diabetes mellitus[MeSH] AND (management OR complications OR therapy)",
    "lung_cancer": "lung neoplasms[MeSH] AND (diagnosis OR treatment OR prognosis)",
    "covid19": "COVID-19[MeSH] AND (clinical features OR treatment OR outcomes)",
    "alzheimers": "Alzheimer disease[MeSH] AND (pathology OR biomarkers OR treatment)",
}

ABSTRACTS_PER_TOPIC = 200
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pubmed_corpus.csv")


def search_pubmed(query: str, max_results: int) -> list[str]:
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, usehistory="y")
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def fetch_abstracts(pmids: list[str]) -> list[dict]:
    records = []
    batch_size = 50
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        ids = ",".join(batch)
        handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="xml")
        data = Entrez.read(handle)
        handle.close()
        for article in data["PubmedArticle"]:
            try:
                medline = article["MedlineCitation"]
                pmid = str(medline["PMID"])
                art = medline["Article"]
                title = str(art.get("ArticleTitle", ""))
                if "Abstract" not in art:
                    continue
                abstract_texts = art["Abstract"].get("AbstractText", [])
                if isinstance(abstract_texts, list):
                    abstract = " ".join(str(t) for t in abstract_texts)
                else:
                    abstract = str(abstract_texts)
                if len(abstract.split()) < 50:
                    continue
                mesh_list = medline.get("MeshHeadingList", [])
                mesh_terms = "|".join(
                    str(m["DescriptorName"]) for m in mesh_list
                )
                records.append(
                    {
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "mesh_terms": mesh_terms,
                    }
                )
            except (KeyError, IndexError):
                continue
        time.sleep(0.4)
    return records


def main():
    all_records = []
    for topic, query in TOPICS.items():
        print(f"Fetching {ABSTRACTS_PER_TOPIC} abstracts for: {topic}")
        pmids = search_pubmed(query, ABSTRACTS_PER_TOPIC)
        print(f"  Found {len(pmids)} PMIDs")
        records = fetch_abstracts(pmids)
        for r in records:
            r["topic"] = topic
        all_records.extend(records)
        print(f"  Collected {len(records)} abstracts with sufficient length")
        time.sleep(1)

    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset="pmid").reset_index(drop=True)
    print(f"\nTotal after deduplication: {len(df)} abstracts")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
