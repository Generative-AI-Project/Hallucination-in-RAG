# app.py — RAG Hallucination Detector (Fixed, VS Code / Streamlit)
# Run: streamlit run app.py
# Requires: GROQ_API_KEY env variable  OR  enter it in the sidebar

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
import time
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Hallucination Detector",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
.chunk-box {
    background: #f9fafb;
    border-left: 4px solid #3b82f6;
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 8px;
    font-family: monospace;
    font-size: 13px;
    white-space: pre-wrap;
}
.disclaimer {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 12px 16px;
    margin: 10px 0;
    border-radius: 8px;
    font-size: 13px;
}
.triplet-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    margin: 3px 2px;
}
.t-supported   { background:#d1fae5; color:#065f46; }
.t-hallucinated{ background:#fee2e2; color:#991b1b; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────
for k, v in [("results_history", []), ("current", None)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────
#  RESOURCE LOADING  (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading models & FAISS indexes…")
def load_resources():
    status = {}

    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        model_kwargs={"device": "cpu"},
    )
    status["Embedding model"] = "✅ Loaded"

    st_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    status["SentenceTransformer"] = "✅ Loaded"

    stores = {}
    index_map = {
        "SQuAD (General Knowledge)": "squad_faiss_index",
    }

    for name, path in index_map.items():
        if os.path.exists(path):
            try:
                stores[name] = FAISS.load_local(
                    path, embed_model, allow_dangerous_deserialization=True
                )
                status[name] = f"✅ Loaded from {path}"
            except Exception as e:
                status[name] = f"❌ Failed ({e}) — using demo"
                stores[name] = _make_demo_store(embed_model)
        else:
            status[name] = f"⚠️ Not found — demo mode"
            stores[name] = _make_demo_store(embed_model)

    return embed_model, st_model, stores, status


def _make_demo_store(embed_model):
    docs = [
        Document(page_content=(
            "Normandy is a region in northern France. "
            "The Normans conquered England in 1066 under William the Conqueror."
        )),
        Document(page_content=(
            "The Eiffel Tower is located in Paris, France. "
            "It was completed in 1889. Gustave Eiffel's company built it for the 1889 World's Fair."
        )),
        Document(page_content=(
            "Albert Einstein developed the theory of general relativity in 1915. "
            "He was awarded the Nobel Prize in Physics in 1921 for the photoelectric effect."
        )),
        Document(page_content=(
            "Marie Curie was a physicist and chemist who conducted research on radioactivity. "
            "She was the first woman to win a Nobel Prize."
        )),
    ]
    return FAISS.from_documents(docs, embed_model)


embed_model, st_model, stores, load_status = load_resources()

# ─────────────────────────────────────────────────────────────
#  SIDEBAR — config & status
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_api_key:
        groq_api_key = st.text_input("GROQ API Key:", type="password",
                                     help="Get free key at console.groq.com")
    groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
    if not groq_client:
        st.error("⚠️ Enter GROQ API key to proceed.")

    st.markdown("---")
    selected_db = st.selectbox("📚 Knowledge Base", list(stores.keys()))

    st.markdown("---")
    st.subheader("📋 Index Status")
    for k, v in load_status.items():
        if "✅" in v: st.success(v)
        elif "⚠️" in v: st.warning(v)
        else: st.error(v)

    st.markdown("---")
    st.subheader("📊 Last Hallucination Score")
    score_slot = st.empty()

# ─────────────────────────────────────────────────────────────
#  MODULE 1 — RETRIEVAL AGENT
# ─────────────────────────────────────────────────────────────
def retrieve_chunks(vectorstore, question: str, k: int = 5):
    return vectorstore.similarity_search(question, k=k)

def chunks_to_context(chunks) -> str:
    return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(chunks))

# ─────────────────────────────────────────────────────────────
#  MODULE 2 — GENERATOR AGENT
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
#  MODULE 2 — GENERATOR AGENT
# ─────────────────────────────────────────────────────────────
def generate_answer(client, question: str, chunks) -> str:
    context = chunks_to_context(chunks)
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a confident answering agent. "
                    "Answer the question concisely. "
                    "If the answer is in the context, use it. "
                    "If the answer is NOT in the context, you MUST use your own general knowledge to answer confidently. "
                    "NEVER use phrases like 'The context does not mention', 'According to the context', or 'Based on the provided text'. Just state the answer directly as a fact."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            },
        ],
        temperature=0.4,  # slightly higher temperature helps it hallucinate more naturally
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()

# ─────────────────────────────────────────────────────────────
#  TRIPLET EXTRACTION  (LLM-first, regex fallback)
# ─────────────────────────────────────────────────────────────
_TRIPLET_SYS = (
    "You are a precise information-extraction assistant. "
    "Output ONLY a valid JSON array and nothing else — no explanation, no markdown fences."
)

_TRIPLET_USER = """Extract every factual claim from the TEXT below as a JSON array of triplets.
Each triplet must have exactly three string fields: "subject", "predicate", "object".

Rules:
- subject  = the entity the claim is about (noun phrase)
- predicate = the relationship verb / phrase (e.g. "is located in", "was born in", "developed", "won")
- object   = the value / target of the relationship (noun phrase or year/date)
- Split compound sentences into separate triplets.
- If there are no facts, return [].

TEXT:
{text}

JSON array:"""


def extract_triplets_llm(client, text: str) -> list:
    """Use LLaMA via Groq to extract triplets; return list of dicts."""
    if not text or len(text.strip()) < 10:
        return []
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _TRIPLET_SYS},
                {"role": "user",   "content": _TRIPLET_USER.format(text=text[:1500])},
            ],
            temperature=0.0,
            max_tokens=1000,
        )
        raw = resp.choices[0].message.content.strip()

        # Strip accidental markdown fences
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*",     "", raw)
        raw = raw.strip()

        # Find the JSON array
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []

        parsed = json.loads(match.group())
        if not isinstance(parsed, list):
            return []

        result = []
        for t in parsed:
            if not isinstance(t, dict):
                continue
            # Accept both lower and Title case keys
            s = str(t.get("subject",   t.get("Subject",   ""))).strip()
            p = str(t.get("predicate", t.get("Predicate", ""))).strip()
            o = str(t.get("object",    t.get("Object",    ""))).strip()
            if s and p and o:
                result.append({"subject": s, "predicate": p, "object": o})
        return result

    except Exception as e:
        st.warning(f"⚠️ Triplet LLM call failed: {e}")
        return []


def extract_triplets_regex(text: str) -> list:
    """Lightweight regex fallback."""
    triplets = []
    patterns = [
        (r"([\w\s]+?)\s+is located in\s+([\w\s]+)", "is located in"),
        (r"([\w\s]+?)\s+is a\s+([\w\s]+)",           "is a"),
        (r"([\w\s]+?)\s+is in\s+([\w\s]+)",           "is in"),
        (r"([\w\s]+?)\s+was born in\s+([\w\s]+)",     "was born in"),
        (r"([\w\s]+?)\s+was completed in\s+([\w\s]+)","was completed in"),
        (r"([\w\s]+?)\s+developed\s+([\w\s]+)",       "developed"),
        (r"([\w\s]+?)\s+built\s+([\w\s]+)",           "built"),
        (r"([\w\s]+?)\s+won\s+([\w\s]+)",             "won"),
        (r"([\w\s]+?)\s+conquered\s+([\w\s]+)",       "conquered"),
    ]
    for pat, pred in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            s = m.group(1).strip()
            o = m.group(2).strip()
            # Clean trailing punctuation
            o = re.sub(r"[,\.\s]+$", "", o)
            s = re.sub(r"[,\.\s]+$", "", s)
            if len(s) > 1 and len(o) > 1:
                triplets.append({"subject": s, "predicate": pred, "object": o})
    return triplets


def extract_triplets(client, text: str) -> list:
    """Try LLM first; fall back to regex if LLM returns nothing."""
    triplets = extract_triplets_llm(client, text)
    if not triplets:
        triplets = extract_triplets_regex(text)
    return triplets

# ─────────────────────────────────────────────────────────────
#  MODULE 3 — HALLUCINATION DETECTION (HDM)
# ─────────────────────────────────────────────────────────────
def _cosine(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def nli_score(client, context: str, claim: str) -> float:
    """
    Strict NLI: asks the LLM whether the context EXPLICITLY supports the claim.
    Returns:
      0.0 = explicitly entailed by context  → supported
      0.8 = not mentioned / neutral          → treat as unsupported
      1.0 = contradicted                     → hallucinated
    """
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict fact-checker. "
                        "Output exactly ONE word: entails, neutral, or contradicts.\n"
                        "Rules:\n"
                        "- 'entails'    : the context EXPLICITLY states this claim as a clear fact.\n"
                        "- 'neutral'    : the context does NOT mention this claim (even if it sounds plausible).\n"
                        "- 'contradicts': the context states something that CONFLICTS with this claim.\n"
                        "Do NOT infer or assume. If the claim is not directly stated, output 'neutral'."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"CONTEXT:\n{context[:800]}\n\n"
                        f"CLAIM: {claim}\n\n"
                        "One word (entails / neutral / contradicts):"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=10,
        )
        label = resp.choices[0].message.content.strip().lower()
        if "contradict" in label:
            return 1.0
        if "entail" in label:
            return 0.0
        return 0.8   # neutral → lean unsupported
    except Exception:
        return 0.8


def _entity_in_context(entity: str, context_text: str) -> bool:
    """Hard check: is this entity string (or a close token) mentioned in the raw context?"""
    entity_lower = entity.lower().strip()
    ctx_lower    = context_text.lower()
    # Accept if any word of the entity (>3 chars) appears in context
    tokens = [tok for tok in entity_lower.split() if len(tok) > 3]
    if not tokens:
        return entity_lower in ctx_lower
    return any(tok in ctx_lower for tok in tokens)


def hallucination_detector(client, context_text: str, answer_text: str,
                           embed_model_st, threshold: float = 0.40):
    """
    Scoring logic per answer triplet:
      1. HARD GATE — if neither subject nor object appears anywhere in the raw
         context text, the claim is immediately flagged as unsupported (score = 1.0).
      2. Semantic similarity (cosine) between answer triplet and all context triplets.
         Entity overlap boosts similarity; no overlap penalises it.
      3. Strict NLI — neutral now scores 0.8 (lean unsupported).
      4. Hybrid = 0.5 * (1 - cosine) + 0.5 * nli
         threshold = 0.40  →  claim flagged if hybrid >= threshold.

    Returns dict with:
      hallucination_score, supported_claims, unsupported_claims,
      context_triplets, answer_triplets, total_claims
    """
    ctx_triplets = extract_triplets(client, context_text)
    ans_triplets = extract_triplets(client, answer_text)

    st.caption(
        f"🔬 Extracted **{len(ctx_triplets)}** context triplets  |  "
        f"**{len(ans_triplets)}** answer triplets"
    )

    if not ans_triplets:
        return {
            "hallucination_score": 0.0,
            "supported_claims":   [],
            "unsupported_claims": [],
            "context_triplets":   ctx_triplets,
            "answer_triplets":    ans_triplets,
            "total_claims":       0,
        }

    # Pre-compute context triplet embeddings
    ctx_texts = [
        f"{t['subject']} {t['predicate']} {t['object']}" for t in ctx_triplets
    ]
    ctx_embs = embed_model_st.encode(ctx_texts) if ctx_texts else []

    supported, unsupported = [], []

    for ans_t in ans_triplets:
        claim_text = f"{ans_t['subject']} {ans_t['predicate']} {ans_t['object']}"

        # ── HARD GATE ──────────────────────────────────────────────────────────
        # If neither the subject NOR the object appears anywhere in the raw
        # context, we skip the hybrid score and flag immediately.
        subj_in_ctx = _entity_in_context(ans_t["subject"], context_text)
        obj_in_ctx  = _entity_in_context(ans_t["object"],  context_text)

        if not subj_in_ctx and not obj_in_ctx:
            record = {
                "subject":          ans_t["subject"],
                "predicate":        ans_t["predicate"],
                "object":           ans_t["object"],
                "claim_text":       claim_text,
                "similarity_score": 0.0,
                "nli_score":        1.0,
                "hybrid_score":     1.0,
                "reason":           "entities absent from context",
            }
            unsupported.append(record)
            continue

        # ── SEMANTIC SIMILARITY ────────────────────────────────────────────────
        best_cos = 0.0
        if len(ctx_embs) > 0:
            ans_emb = embed_model_st.encode(claim_text)
            for i, ce in enumerate(ctx_embs):
                sim = _cosine(ans_emb, ce)

                s_overlap = (
                    ans_t["subject"].lower() in ctx_triplets[i]["subject"].lower()
                    or ctx_triplets[i]["subject"].lower() in ans_t["subject"].lower()
                )
                o_overlap = (
                    ans_t["object"].lower() in ctx_triplets[i]["object"].lower()
                    or ctx_triplets[i]["object"].lower() in ans_t["object"].lower()
                )
                # Require BOTH subject AND object to match for full credit
                if s_overlap and o_overlap:
                    pass                  # full similarity
                elif s_overlap or o_overlap:
                    sim *= 0.6            # partial match
                else:
                    sim *= 0.15           # unrelated triplet

                best_cos = max(best_cos, sim)

        # ── STRICT NLI ────────────────────────────────────────────────────────
        nli = nli_score(client, context_text[:800], claim_text)

        # ── HYBRID SCORE ──────────────────────────────────────────────────────
        # 0.0 = fully supported, 1.0 = fully hallucinated
        hybrid = 0.50 * (1.0 - best_cos) + 0.50 * nli

        record = {
            "subject":          ans_t["subject"],
            "predicate":        ans_t["predicate"],
            "object":           ans_t["object"],
            "claim_text":       claim_text,
            "similarity_score": round(best_cos, 4),
            "nli_score":        round(nli, 4),
            "hybrid_score":     round(hybrid, 4),
            "reason":           "hybrid score",
        }

        if hybrid < threshold:
            supported.append(record)
        else:
            unsupported.append(record)

    total = len(supported) + len(unsupported)
    score = round(len(unsupported) / total, 4) if total else 0.0

    return {
        "hallucination_score": score,
        "supported_claims":    supported,
        "unsupported_claims":  unsupported,
        "context_triplets":    ctx_triplets,
        "answer_triplets":     ans_triplets,
        "total_claims":        total,
    }

# ─────────────────────────────────────────────────────────────
#  CORRECTION AGENT
# ─────────────────────────────────────────────────────────────
def correction_agent(client, question, original_answer, context, unsupported):
    if not unsupported:
        return original_answer

    claims_str = "\n".join(f"  • {c['claim_text']}" for c in unsupported[:6])
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict answer corrector. "
                    "You ONLY use information explicitly present in the provided context. "
                    "You NEVER use your own training knowledge or general world knowledge. "
                    "If a claim is not supported by the context, say exactly: "
                    "'This information is not mentioned in the provided context.' "
                    "Do NOT add any facts from outside the context under any circumstances."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"QUESTION: {question}\n\n"
                    f"CONTEXT (the ONLY allowed source of facts):\n{context[:1800]}\n\n"
                    f"ORIGINAL ANSWER (contains unsupported claims):\n{original_answer}\n\n"
                    f"UNSUPPORTED CLAIMS TO FIX:\n{claims_str}\n\n"
                    "Rewrite the answer using ONLY facts from the context above. "
                    "If the answer cannot be found in the context, respond with: "
                    "'This information is not mentioned in the provided context.' "
                    "Do NOT use any outside knowledge. Be concise (1-3 sentences)."
                ),
            },
        ],
        temperature=0.0,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()

# ─────────────────────────────────────────────────────────────
#  GRAPH HELPERS
# ─────────────────────────────────────────────────────────────
def build_graph(triplets: list) -> nx.DiGraph:
    G = nx.DiGraph()
    for t in triplets:
        G.add_edge(t["subject"], t["object"], label=t["predicate"])
    return G


def _draw_empty(ax, title):
    ax.text(0.5, 0.5, "No triplets extracted — graph unavailable",
            ha="center", va="center", fontsize=12, color="gray",
            transform=ax.transAxes)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")


def plot_context_graph(G: nx.DiGraph):
    fig, ax = plt.subplots(figsize=(13, 7))
    if len(G.nodes) == 0:
        _draw_empty(ax, "Context Knowledge Graph")
        return fig

    pos = nx.spring_layout(G, k=2.2, iterations=60, seed=42)

    nx.draw_networkx_nodes(G, pos, node_color="#bbf7d0", node_size=2200,
                           edgecolors="#16a34a", linewidths=2, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#16a34a", width=2,
                           arrows=True, arrowstyle="->", arrowsize=18,
                           alpha=0.75, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels={(u, v): d["label"] for u, v, d in G.edges(data=True)},
        font_size=8, ax=ax,
    )

    ax.set_title("Context Knowledge Graph  (all green = fully supported)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_answer_graph(G: nx.DiGraph, unsupported_claims: list):
    fig, ax = plt.subplots(figsize=(13, 7))
    if len(G.nodes) == 0:
        _draw_empty(ax, "Answer Knowledge Graph")
        return fig

    # Build set of hallucinated edges
    hall_edges = set()
    hall_nodes = set()
    for c in unsupported_claims:
        s, o = c["subject"], c["object"]
        if G.has_edge(s, o):
            hall_edges.add((s, o))
            hall_nodes.update([s, o])

    pos = nx.spring_layout(G, k=2.2, iterations=60, seed=42)

    # Nodes
    ok_nodes   = [n for n in G.nodes if n not in hall_nodes]
    bad_nodes  = [n for n in G.nodes if n in hall_nodes]

    nx.draw_networkx_nodes(G, pos, nodelist=ok_nodes,
                           node_color="#bfdbfe", node_size=2200,
                           edgecolors="#2563eb", linewidths=2, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=bad_nodes,
                           node_color="#fecaca", node_size=2200,
                           edgecolors="#dc2626", linewidths=3, ax=ax)

    # Edges
    ok_edges  = [(u, v) for u, v in G.edges if (u, v) not in hall_edges]
    bad_edges = list(hall_edges)

    if ok_edges:
        nx.draw_networkx_edges(G, pos, edgelist=ok_edges,
                               edge_color="#2563eb", width=2,
                               arrows=True, arrowstyle="->", arrowsize=18,
                               alpha=0.7, ax=ax)
    if bad_edges:
        nx.draw_networkx_edges(G, pos, edgelist=bad_edges,
                               edge_color="#dc2626", width=3,
                               arrows=True, arrowstyle="->", arrowsize=22,
                               style="dashed", alpha=0.9, ax=ax)

    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels={(u, v): G[u][v]["label"] for u, v in G.edges},
        font_size=8, ax=ax,
    )

    legend = [
        mpatches.Patch(color="#bfdbfe", label="Supported node"),
        mpatches.Patch(color="#fecaca", label="Hallucinated node"),
        plt.Line2D([0], [0], color="#2563eb", lw=2, label="Supported relation"),
        plt.Line2D([0], [0], color="#dc2626", lw=2,
                   linestyle="dashed", label="Hallucinated relation"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title("Answer Knowledge Graph  (red = hallucinated, blue = supported)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────
#  MAIN UI
# ─────────────────────────────────────────────────────────────
st.title("🔍 RAG Hallucination Detector")
st.markdown(
    "**Pipeline:** Retrieval Agent → Generator Agent → "
    "Hallucination Detection Module → Correction Agent"
)

col_q, col_btn = st.columns([4, 1])
with col_q:
    user_question = st.text_area(
        "Ask a question:",
        placeholder="e.g.  In what country is Normandy located?",
        height=80,
    )
with col_btn:
    st.write("")
    st.write("")
    run_btn = st.button(
        "🚀 Analyse",
        type="primary",
        use_container_width=True,
        disabled=(not groq_client),
    )

# ─── PIPELINE EXECUTION ───────────────────────────────────────
if run_btn and user_question.strip() and groq_client:
    with st.spinner("Running pipeline…"):

        # 1. Retrieval
        chunks       = retrieve_chunks(stores[selected_db], user_question, k=5)
        context_text = chunks_to_context(chunks)

        # 2. Generation
        answer = generate_answer(groq_client, user_question, chunks)

        # 3. HDM
        hdm = hallucination_detector(
            groq_client, context_text, answer, st_model, threshold=0.35
        )

        # 4. Correction (if needed)
        needs_correction = (
            hdm["hallucination_score"] >= 0.3 and hdm["unsupported_claims"]
        )
        if needs_correction:
            corrected = correction_agent(
                groq_client, user_question, answer,
                context_text, hdm["unsupported_claims"]
            )
            verdict = "⚠️ CORRECTED"
        else:
            corrected = answer
            verdict   = "✅ ACCEPTED"

        # Graphs
        ctx_graph = build_graph(hdm["context_triplets"])
        ans_graph = build_graph(hdm["answer_triplets"])

        result = {
            "question":           user_question,
            "chunks":             chunks,
            "context_text":       context_text,
            "original_answer":    answer,
            "corrected_answer":   corrected,
            "verdict":            verdict,
            "hdm":                hdm,
            "ctx_graph":          ctx_graph,
            "ans_graph":          ans_graph,
            "timestamp":          time.strftime("%Y-%m-%d %H:%M:%S"),
            "database":           selected_db,
        }
        st.session_state.current = result
        st.session_state.results_history.insert(0, result)

    # Sidebar score
    with score_slot:
        hs = hdm["hallucination_score"]
        st.metric("Hallucination Score", f"{hs:.1%}")
        if hs < 0.3:
            st.success("✅ ACCEPTED")
        elif hs < 0.6:
            st.warning("⚠️ CORRECTED (moderate)")
        else:
            st.error("🚨 CORRECTED (high risk)")

# ─── RESULTS DISPLAY ─────────────────────────────────────────
if st.session_state.current:
    r   = st.session_state.current
    hdm = r["hdm"]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Retrieved Chunks",
        "🌐 Context Graph",
        "💬 Answer & Answer Graph",
        "🔍 Detection Details",
        "📋 History",
    ])

    # ── Tab 1: Retrieved Chunks ──────────────────────────────
    with tab1:
        st.subheader("📄 Retrieved Context Chunks")
        st.caption(f"Knowledge base: **{r['database']}**")
        for i, chunk in enumerate(r["chunks"]):
            with st.expander(f"Chunk {i+1}", expanded=(i == 0)):
                st.markdown(
                    f'<div class="chunk-box">{chunk.page_content}</div>',
                    unsafe_allow_html=True,
                )

    # ── Tab 2: Context Graph ─────────────────────────────────
    with tab2:
        st.subheader("🌐 Context Knowledge Graph")
        if hdm["context_triplets"]:
            st.caption(f"{len(hdm['context_triplets'])} triplets extracted from retrieved context")
            with st.expander("View raw context triplets"):
                for t in hdm["context_triplets"]:
                    st.write(f"**{t['subject']}** → *{t['predicate']}* → **{t['object']}**")
            fig = plot_context_graph(r["ctx_graph"])
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No triplets could be extracted from the context.")

    # ── Tab 3: Answer + Answer Graph ────────────────────────
    with tab3:
        st.subheader("🤖 Original Generated Answer")
        st.info(r["original_answer"])

        if r["verdict"] == "⚠️ CORRECTED":
            st.subheader("✅ Corrected Answer")
            st.success(r["corrected_answer"])
        else:
            st.success(f"**Verdict:** {r['verdict']}  — answer accepted as-is.")

        st.markdown("---")
        st.subheader("💬 Answer Knowledge Graph")

        if hdm["answer_triplets"]:
            st.caption(f"{len(hdm['answer_triplets'])} answer triplets  |  "
                       f"🔴 {len(hdm['unsupported_claims'])} hallucinated  |  "
                       f"🔵 {len(hdm['supported_claims'])} supported")

            with st.expander("View answer triplets with labels"):
                for t in hdm["answer_triplets"]:
                    c_text = f"{t['subject']} {t['predicate']} {t['object']}"
                    is_hall = any(
                        u["claim_text"] == c_text for u in hdm["unsupported_claims"]
                    )
                    css = "t-hallucinated" if is_hall else "t-supported"
                    label = "HALLUCINATED" if is_hall else "supported"
                    st.markdown(
                        f"<span class='triplet-tag {css}'>{label}</span> "
                        f"**{t['subject']}** → *{t['predicate']}* → **{t['object']}**",
                        unsafe_allow_html=True,
                    )

            fig = plot_answer_graph(r["ans_graph"], hdm["unsupported_claims"])
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No triplets could be extracted from the answer.")

    # ── Tab 4: Detection Details ─────────────────────────────
    with tab4:
        st.subheader("🔍 Hallucination Detection Results")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Hallucination Score", f"{hdm['hallucination_score']:.1%}")
        col2.metric("Total Claims",        hdm["total_claims"])
        col3.metric("Supported",           len(hdm["supported_claims"]))
        col4.metric("Hallucinated",        len(hdm["unsupported_claims"]))

        st.markdown(f"**Verdict:** {r['verdict']}")
        st.markdown("---")

        if hdm["unsupported_claims"]:
            st.markdown("### ❌ Hallucinated Claims")
            for c in hdm["unsupported_claims"]:
                with st.expander(c["claim_text"], expanded=True):
                    cols = st.columns(3)
                    cols[0].metric("Semantic Sim", f"{c['similarity_score']:.3f}")
                    cols[1].metric("NLI Score",    f"{c['nli_score']:.3f}")
                    cols[2].metric("Hybrid Score", f"{c['hybrid_score']:.3f}")

        if hdm["supported_claims"]:
            st.markdown("### ✅ Supported Claims")
            for c in hdm["supported_claims"]:
                st.markdown(
                    f"<span class='triplet-tag t-supported'>supported</span> {c['claim_text']}",
                    unsafe_allow_html=True,
                )

    # ── Tab 5: History ───────────────────────────────────────
    with tab5:
        st.subheader("📋 Query History")
        history = st.session_state.results_history
        if not history:
            st.info("No history yet.")
        for i, h in enumerate(history[:10]):
            score_label = f"{h['hdm']['hallucination_score']:.1%}"
            with st.expander(
                f"[{h['timestamp']}]  Q{i+1}: {h['question'][:60]}…  "
                f"| Score: {score_label}  | {h['verdict']}"
            ):
                st.write(f"**Original answer:** {h['original_answer']}")
                if h["verdict"] == "⚠️ CORRECTED":
                    st.write(f"**Corrected answer:** {h['corrected_answer']}")
                st.write(
                    f"Hallucinated claims: "
                    + ", ".join(c["claim_text"] for c in h["hdm"]["unsupported_claims"])
                    or "None"
                )

# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;font-size:12px;'>"
    "🔍 RAG Hallucination Detector | Groq LLaMA-3.1-8b | "
    "<strong>LLM outputs may be inaccurate — always verify</strong>"
    "</div>",
    unsafe_allow_html=True,
)