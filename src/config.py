"""
config.py - Centralna konfiguracija projekta.
Ključ rešenja: pravilno postavljanje OpenAI klijenta za Bosch Model Farm
sa proxy podrškom i Azure OpenAI endpointom.
"""

import os
import httpx
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()

# ── Osnovna podešavanja ────────────────────────────────────────────────────────
API_KEY        = os.getenv("MODEL_FARM_API_KEY", "")
_RAW_ENDPOINT  = os.getenv("MODEL_FARM_ENDPOINT", "").rstrip("/")
# Bosch Model Farm zahteva /api sufiks na base URL-u
# https://aoai-farm.bosch-temp.com  →  https://aoai-farm.bosch-temp.com/api
ENDPOINT       = _RAW_ENDPOINT + "/api" if not _RAW_ENDPOINT.endswith("/api") else _RAW_ENDPOINT
API_VERSION    = os.getenv("API_VERSION", "2024-08-01-preview")
CHAT_MODEL     = os.getenv("DEPLOYMENT_CHAT", "askbosch-prod-farm-openai-gpt-4o-mini-2024-07-18")
EMBED_MODEL    = os.getenv("DEPLOYMENT_EMBEDDING", "askbosch-prod-farm-openai-text-embedding-3-small")
HTTP_PROXY     = os.getenv("HTTP_PROXY", "http://127.0.0.1:3128")
HTTPS_PROXY    = os.getenv("HTTPS_PROXY", "http://127.0.0.1:3128")

# Proxy se koristi SAMO za spoljne API pozive — Gradio localhost mora biti izuzet
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"
os.environ["no_proxy"]  = "localhost,127.0.0.1,0.0.0.0"

# ── Proxy transport ───────────────────────────────────────────────────────────
def _make_http_client() -> httpx.Client:
    """Kreira httpx klijent sa proxy podešavanjima (kompatibilan sa httpx>=0.28)."""
    transport = httpx.HTTPTransport(proxy=HTTPS_PROXY, verify=False)
    return httpx.Client(transport=transport, verify=False, timeout=120.0)


def _make_async_http_client() -> httpx.AsyncClient:
    """Kreira async httpx klijent sa proxy podešavanjima (kompatibilan sa httpx>=0.28)."""
    transport = httpx.AsyncHTTPTransport(proxy=HTTPS_PROXY, verify=False)
    return httpx.AsyncClient(transport=transport, verify=False, timeout=120.0)


# ── Raw AzureOpenAI klijent (za direktne pozive) ──────────────────────────────
def get_azure_client() -> AzureOpenAI:
    """Vraća AzureOpenAI klijent sa proxy-jem."""
    return AzureOpenAI(
        api_key=API_KEY,
        azure_endpoint=ENDPOINT,
        api_version=API_VERSION,
        http_client=_make_http_client(),
    )


# ── LangChain LLM wrapper ─────────────────────────────────────────────────────
def get_llm(temperature: float = 0.0) -> AzureChatOpenAI:
    """Vraća LangChain AzureChatOpenAI sa proxy-jem."""
    return AzureChatOpenAI(
        azure_deployment=CHAT_MODEL,
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
        api_version=API_VERSION,
        temperature=temperature,
        http_client=_make_http_client(),
        http_async_client=_make_async_http_client(),
    )


# ── LangChain Embeddings wrapper ──────────────────────────────────────────────
def get_embeddings() -> AzureOpenAIEmbeddings:
    """Vraća LangChain AzureOpenAIEmbeddings sa proxy-jem."""
    return AzureOpenAIEmbeddings(
        azure_deployment=EMBED_MODEL,
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
        api_version=API_VERSION,
        http_client=_make_http_client(),
        http_async_client=_make_async_http_client(),
    )


# ── RAG podešavanja ───────────────────────────────────────────────────────────
CHUNK_SIZE         = 512
CHUNK_OVERLAP      = 64
TOP_K_VECTOR       = 5
TOP_K_BM25         = 5
TOP_K_HYBRID       = 5
HYBRID_ALPHA       = 0.5   # 0=samo BM25, 1=samo vektor
NUM_TEST_QUESTIONS = 10    # Broj pitanja koje agent generiše
DATA_DIR           = "data"
BOOK_PATH          = os.path.join(DATA_DIR, "book.pdf")