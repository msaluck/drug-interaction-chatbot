import os
import re
import logging
import pandas as pd
from dotenv import load_dotenv
from rapidfuzz import process, fuzz
from langchain_community.llms.llamacpp import LlamaCpp as LlamaCppLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env if present
load_dotenv()

logger = logging.getLogger(__name__)

# --- 1. DATA LOADING AND RETRIEVAL CLASS ---


def format_retrieved_info(row):
    """Formats the retrieved DataFrame row into a readable string for the LLM."""
    if row is None:
        return "No interaction information found for this pair of drugs."

    info = (
        f"- Drug A: {row.get('drug_a', 'N/A')}\n"
        f"- Drug B: {row.get('drug_b', 'N/A')}\n"
        f"- Interaction Effect: {row.get('efek_interaksi', 'N/A')}\n"
        f"- Pharmacodynamic Interaction Mechanism: {row.get('mekanisme_interaksi_farmakodinamik', 'N/A')}\n"
        f"- Pharmacokinetic Interaction Mechanism: {row.get('mekanisme_interaksi_farmakokinetik', 'N/A')}\n"
        f"- Recommended Management: {row.get('manajemen_interaksi', 'N/A')}"
    )
    return info


class DrugRetriever:
    def __init__(self, source):
        """
        Initialize with a file path (str) or an existing DataFrame.
        """
        if isinstance(source, str):
            self.df = self._load_data(source)
        elif isinstance(source, pd.DataFrame):
            self.df = source
            # Ensure columns are lower case if passing raw DF
            self.df.columns = [col.lower() for col in self.df.columns]
        else:
            logger.error("Invalid source type. Must be filepath (str) or DataFrame.")
            self.df = None

        if self.df is not None:
            # Pre-compute the unique drug list ONCE during initialization
            self.all_drugs = list(
                set(self.df["drug_a"].unique()).union(set(self.df["drug_b"].unique()))
            )
            logger.info("Loaded %d unique drugs into vocabulary.", len(self.all_drugs))
        else:
            self.all_drugs = []

    def _load_data(self, filepath):
        """Loads interaction data and normalizes columns."""
        try:
            df = pd.read_csv(filepath)
            df.columns = [col.lower() for col in df.columns]
            logger.info(
                "Data loaded from %s. Columns: %s", filepath, df.columns.tolist()
            )
            return df
        except FileNotFoundError:
            logger.error("File not found: %s", filepath)
            return None

    def search(self, question: str):
        """
        Search for interaction data and return the raw dictionary or None/Error string.
        """
        if self.df is None:
            return None

        question_lower = question.lower()
        drug1_found = None
        drug2_found = None

        # --- STRATEGY 1: REGEX ---
        patterns = [
            r"between\s+(.+?)\s+and\s+(.+?)(?:\?|$| in | for )",
            r"interaction of\s+(.+?)\s+and\s+(.+?)(?:\?|$| in | for )",
            r"take\s+(.+?)\s+(?:with|and)\s+(.+?)(?:\?|$| in | for )",
            r"^(.+?)\s+(?:and|with|&|\+)\s+(.+?)(?:\?|$| in | for )",
        ]

        match = None
        for pat in patterns:
            match = re.search(pat, question_lower)
            if match:
                break

        if match:
            raw_drug1, raw_drug2 = match.groups()
            # Fuzzy match against CACHED all_drugs list
            m1 = process.extractOne(
                raw_drug1, self.all_drugs, scorer=fuzz.WRatio, score_cutoff=80
            )
            m2 = process.extractOne(
                raw_drug2, self.all_drugs, scorer=fuzz.WRatio, score_cutoff=80
            )
            if m1:
                drug1_found = m1[0]
            if m2:
                drug2_found = m2[0]

        # --- STRATEGY 2: VOCABULARY SCANNING ---
        if not drug1_found or not drug2_found:
            found_drugs = set()
            tokens = re.findall(r"\w+", question_lower)
            for token in tokens:
                if len(token) < 4:
                    continue
                # Fuzzy match token against CACHED list
                match = process.extractOne(
                    token, self.all_drugs, scorer=fuzz.WRatio, score_cutoff=85
                )
                if match:
                    found_drugs.add(match[0])

            if len(found_drugs) >= 2:
                d_list = list(found_drugs)
                drug1_found = d_list[0]
                drug2_found = d_list[1]

        # --- QUERY ---
        if not drug1_found or not drug2_found:
            return {
                "error": "No interaction information found: Could not identify two known drugs in your question."
            }

        result = self.df[
            ((self.df["drug_a"] == drug1_found) & (self.df["drug_b"] == drug2_found))
            | ((self.df["drug_a"] == drug2_found) & (self.df["drug_b"] == drug1_found))
        ]

        if not result.empty:
            return result.iloc[0].to_dict()
        else:
            return {
                "error": f"No interaction information found for {drug1_found} and {drug2_found}."
            }

    def retrieve(self, question: str):
        """
        Finds interaction info. Uses cached vocabulary for speed.
        Returns formatted string for LLM context.
        """
        data = self.search(question)
        if data is None:
            return "Database not loaded."
        if "error" in data:
            return data["error"]
        return format_retrieved_info(data)


# Helper to maintain backward compatibility for tests or other modules if needed
def load_interaction_data(filepath):
    # This function is deprecated in favor of DrugRetriever class
    retriever = DrugRetriever(filepath)
    return retriever.df


def retrieve_interaction_info(question, df):
    """
    Deprecated function wrapper. Uses DrugRetriever with the provided DataFrame.
    """
    retriever = DrugRetriever(df)
    return retriever.retrieve(question)


# --- 2. LANGCHAIN SETUP ---

# Define the prompt template
# This instructs the LLM on how to behave and structures the input.
TEMPLATE = """
You are an expert pharmacology assistant. Your role is to provide information about drug interactions based ONLY on the context provided.

AUDIENCE: {audience}
LANGUAGE: {language}
VERBOSITY: {verbosity}
FORMAT: {format}

INSTRUCTIONS:
- If the audience is 'pharmacist', use professional medical terminology, focus on mechanisms (pharmacokinetic/pharmacodynamic), and be precise.
- If the audience is 'patient' (normal person), explain in simple, easy-to-understand language. Avoid jargon or explain it if necessary. Focus on practical advice and management.
- If verbosity is 'concise', keep the answer short and to the point (max 2-3 sentences).
- If verbosity is 'detailed', provide a comprehensive explanation.
- If format is 'json', output ONLY a valid JSON object with keys: "severity", "summary", "recommendation". Do not add markdown formatting like ```json.
- Do not use any information outside of the provided context.
- If the context says "No information found", reply that you do not have information on that interaction.

CONTEXT:
{context}

QUESTION:
{question}

IMPORTANT: Answer the question in {language} language.
ANSWER:
"""

# Build a standard (string) prompt for LlamaCpp
prompt = PromptTemplate.from_template(TEMPLATE)

# Use a simple string output parser
output_parser = StrOutputParser()


# --- 3. MAIN EXECUTION ---

# if __name__ == "__main__":
#     # Load the data
#     db = load_interaction_data("data/interactions_seed.csv")

#     if db is not None:
#         # Create the RAG chain using LangChain Expression Language (LCEL)
#         # This is where we tie everything together.
#         chain = (
#             {
#                 "context": (lambda x: retrieve_interaction_info(x["question"], db)),
#                 "question": RunnablePassthrough(),
#             }
#             | prompt
#             | model
#             | output_parser
#         )

#         # --- ASK QUESTIONS ---
#         print("\n--- Drug Interaction Chatbot Ready ---")
#         print("Type 'exit' to quit.")

#         while True:
#             try:
#                 user_question = input(
#                     "\nAsk about a drug interaction (e.g., 'What is the interaction between captopril and zolpidem?'): "
#                 )
#                 if user_question.lower() == "exit":
#                     break

#                 # Stream the response for a better user experience
#                 print("\nAssistant:")
#                 for chunk in chain.stream({"question": user_question}):
#                     print(chunk, end="", flush=True)
#                 print("\n" + "=" * 50)

#             except KeyboardInterrupt:
#                 break
#             except Exception as e:
#                 print(f"An error occurred: {e}")

#     print("\nChatbot session ended.")
# ... (keep all the existing imports and functions like load_interaction_data, etc.)


def create_chatbot_chain():
    """Sets up and returns the RAG chain using llama.cpp (llama-cpp-python)."""
    # Initialize the Retriever Class
    retriever = DrugRetriever("data/interactions_seed.csv")
    if retriever.df is None:
        raise RuntimeError("Failed to load interaction data.")

    # Resolve llama.cpp model path and basic settings from env
    model_path = os.getenv("LLAMA_CPP_MODEL", os.path.join("models", "model.gguf"))

    n_ctx = int(os.getenv("LLAMA_CPP_CTX", "4096"))
    # Automatically clamp threads to available CPU count to prevent oversubscription
    max_threads = os.cpu_count() or 4
    env_threads = int(os.getenv("LLAMA_CPP_THREADS", str(max_threads)))
    n_threads = min(env_threads, max_threads)

    n_batch = int(os.getenv("LLAMA_CPP_BATCH", "512"))
    temperature = float(os.getenv("LLAMA_CPP_TEMPERATURE", "0"))

    # Optional GPU layers (only effective if llama-cpp-python built with cuBLAS)
    gpu_layers_env = os.getenv("LLAMA_CPP_GPU_LAYERS")
    n_gpu_layers = (
        int(gpu_layers_env) if gpu_layers_env and gpu_layers_env.isdigit() else 0
    )

    if not os.path.exists(model_path):
        raise RuntimeError(
            f"LLAMA_CPP_MODEL not found at '{model_path}'. Set LLAMA_CPP_MODEL in your .env to a valid GGUF file."
        )

    # Initialize llama.cpp LLM
    llm_kwargs = {
        "model_path": model_path,
        "n_ctx": n_ctx,
        "n_threads": n_threads,
        "n_batch": n_batch,
        "temperature": temperature,
        "streaming": True,  # Optimization: Hint to use streaming mode
    }
    if n_gpu_layers > 0:
        llm_kwargs["n_gpu_layers"] = n_gpu_layers
        logger.info("[llama.cpp] Using GPU offload for %d layers.", n_gpu_layers)
    else:
        logger.info(
            "[llama.cpp] Running fully on CPU (no GPU layers specified or value is 0)."
        )

    llm = LlamaCppLLM(**llm_kwargs)

    # Compose the chain
    chain = (
        {
            # Use the retriever instance method
            "context": (lambda x: retriever.retrieve(x["question"])),
            "question": (lambda x: x["question"]),
            "audience": (lambda x: x.get("audience", "patient")),
            "language": (lambda x: x.get("language", "English")),
            "verbosity": (lambda x: x.get("verbosity", "detailed")),
            "format": (lambda x: x.get("format", "text")),
        }
        | prompt
        | llm
        | output_parser
    )
    # Return both chain and the retriever instance (formerly db)
    return chain, retriever


# We remove the if __name__ == "__main__": block for now
# as the server will be run from main.py
