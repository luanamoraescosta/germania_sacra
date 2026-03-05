"""
config.py
---------
Application-wide constants, default values, and CSS styles.
"""

# Hugging Face token (replace with your own or load from env)
HF_TOKEN: str = "xxxxxxxxxx"

# FactGrid endpoints
FACTGRID_SPARQL_ENDPOINT: str = "https://database.factgrid.de/sparql"
FACTGRID_API_URL: str = "https://database.factgrid.de/w/api.php"
FACTGRID_USER_AGENT: str = "Mozilla/5.0"

# Available language models (ordered from lightest to heaviest)
AVAILABLE_MODELS: list[str] = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-2",
    "microsoft/Phi-3-mini-4k-instruct",
    "openlm-research/open_llama_3b",
]

# Example questions shown in the sidebar
EXAMPLE_QUERIES: list[str] = [
    "List all monasteries in Germany",
    "Show Benedictine monasteries founded before 1200",
    "What monasteries were part of the Germania Sacra project in Bavaria?",
    "Find Cistercian monasteries with their coordinates",
    "Show monasteries dissolved during the Reformation",
]

# FactGrid properties used in SPARQL generation
FACTGRID_PROPERTIES: list[tuple[str, str, str]] = [
    ("Instance of",               "P2",    "to state what the item is"),
    ("Territorial localisation",  "P297",  "national or political integration"),
    ("Coordinate location",       "P48",   "geographic coordinates"),
    ("Begin date",                "P49",   "start in time"),
    ("End date",                  "P50",   "final point of a period"),
    ("Klosterdatenbank ID",       "P471",  "Germania Sacra identifier"),
    ("Religious order",           "P746",  "membership in a religious order"),
    ("GND ID",                    "P76",   "German National Library identifier"),
    ("Administrative localisation","P1069","administrative hierarchy"),
    ("Naming",                    "P34",   "historical names (use qualifiers P49/P50 or P290/P291)"),
]

# Fallback terms used when the model is unavailable
FALLBACK_ENTITY_TERMS: list[str] = [
    "monastery", "abbey", "convent", "cloister", "kloster", "stift",
]

# Page CSS injected into Streamlit
PAGE_CSS: str = """
<style>
    .main {background-color:#9b1c1c;}
    .stButton>button{
        background-color:#9b1c1c;color:white;
        border-radius:5px;padding:10px 20px;
    }
    .stTextInput>div>div>input{border-radius:5px;}
    .query-box{
        background:#f8f9fa;border:1px solid #ddd;
        border-radius:5px;padding:10px;
        font-family:monospace;overflow-x:auto;
    }
</style>
"""
