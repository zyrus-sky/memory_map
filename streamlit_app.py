# streamlit_app.py
# Memory Graph Builder — dark theme, plain-text hovers, persistent state, 2D+3D exports (standalone HTML)

import os
import re
import json
import time
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

import streamlit as st
import requests
import networkx as nx
from pyvis.network import Network
import plotly.graph_objects as go
import fitz  # PyMuPDF for PDF

# -------------------- Session state (persistence) --------------------
if "graph_built" not in st.session_state:
    st.session_state.graph_built = False
if "G" not in st.session_state:
    st.session_state.G = None
if "pyvis_html" not in st.session_state:
    st.session_state.pyvis_html = ""
if "plotly_fig3d" not in st.session_state:
    st.session_state.plotly_fig3d = None

# Optional spaCy for richer keyword extraction (safe fallback if missing)
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None

# -------------------- App config & theme --------------------
st.set_page_config(page_title="Memory Graph Builder", page_icon="🧠", layout="wide")

THEMES = {
    "Light": {
        "bg": "#ffffff",
        "font": "#222222",
        "edge": "rgba(120,120,120,0.65)",
        "node_label": "#333333",
        "edge_label": "#999999",
        "plot_bg": "#ffffff",
        "plot_font": "#111111",
        "edge_cone": "#707070"
    },
    "Dark": {
        "bg": "#111418",
        "font": "#E6E6E6",
        "edge": "rgba(200,200,200,0.35)",
        "node_label": "#E6E6E6",
        "edge_label": "#C8D0E0",
        "plot_bg": "#0E1117",
        "plot_font": "#F0F0F0",
        "edge_cone": "#C0C0C0"
    }
}

def get_streamlit_theme_pref():
    # Default to Dark when "System" is selected
    return "Dark"

with st.sidebar:
    st.markdown("---")
    st.subheader("Theme")
    theme_choice = st.radio("Appearance", ["System", "Light", "Dark"], horizontal=True, index=0)
    if theme_choice == "System":
        active_theme = THEMES[get_streamlit_theme_pref()]
    else:
        active_theme = THEMES[theme_choice]

# Global CSS (dark-friendly look)
st.markdown(f"""
<style>
            
header.stAppHeader[data-testid="stHeader"] .stAppToolbar[data-testid="stToolbar"] {{ display: none !important; }}
/* Basics */
.small-note {{ font-size: 0.9rem; opacity: 0.85; }}
.code-small {{ font-size: 0.85rem; }}
.stButton>button {{ height: 2.6rem; }}

/* 1) Main container: tighter padding */
main.block-container,
div[data-testid="stMainBlockContainer"].block-container {{
  padding-top: 0.35rem !important;
  padding-bottom: 0.7rem !important;
  padding-left: 0.9rem !important;
  padding-right: 0.9rem !important;
}}

/* 2) Slim top header/toolbar (keep menu intact) */
header.stAppHeader[data-testid="stHeader"] {{
  min-height: 30px !important;
  height: 30px !important;
  padding: 2px 6px !important;
  background: transparent !important;
  border: 0 !important;
}}
header.stAppHeader[data-testid="stHeader"] .stAppToolbar[data-testid="stToolbar"] {{
  gap: 4px !important;
  padding: 0 !important;
  min-height: 24px !important;
}}

/* 3) Sidebar header row: compact, keep collapse button */
div[data-testid="stSidebarHeader"] {{
  min-height: 28px !important;
  height: 28px !important;
  padding: 2px 4px !important;
  margin: 0 !important;
  border: 0 !important;
}}
div[data-testid="stSidebarHeader"] > div[data-testid="stLogoSpacer"] {{ display: none !important; }}
div[data-testid="stSidebarHeader"] > div[data-testid="stSidebarCollapseButton"] {{ padding: 0 !important; }}

/* 4) Sidebar inner padding */
section[data-testid="stSidebar"] .block-container {{
  padding-top: 0.5rem !important;
  padding-bottom: 0.5rem !important;
}}

/* 5) Reduce vertical gaps between elements */
div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] {{
  margin-top: 0.3rem !important;
  margin-bottom: 0.3rem !important;
}}
div[data-testid="stHeading"] h1,
div[data-testid="stHeading"] h2,
div[data-testid="stHeading"] h3 {{
  margin-top: 0.15rem !important;
  margin-bottom: 0.3rem !important;
}}
div[data-testid="stHorizontalBlock"] {{
  margin-top: 0.3rem !important;
  margin-bottom: 0.3rem !important;
}}
div[data-testid="stTextArea"] {{ margin-top: 0.25rem !important; }}
div[data-testid="stMarkdownContainer"] p {{ margin: 0.25rem 0 !important; }}
div[data-testid="stAlertContainer"] {{
  padding-top: 0.35rem !important;
  padding-bottom: 0.35rem !important;
}}
div[data-testid="stMarkdownContainer"] hr {{
  margin: 0.5rem 0 !important;
  border-top: 1px solid rgba(255,255,255,0.08) !important;
}}

/* 6) PyVis iframe container look */
iframe {{
  border-radius: 10px !important;
  border: 1px solid #222831 !important;
  box-shadow: 0 8px 24px rgba(0,0,0,0.35) !important;
  background: transparent !important;
}}

/* 7) vis-network internals */
div.vis-network, div.vis-network div {{ background: transparent !important; }}
div.vis-network div.vis-navigation div {{
  background: #222831 !important;
  color: #E6E6E6 !important;
  border: 1px solid #2F3640 !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
  border-radius: 10px !important;
}}
div.vis-network div.vis-navigation div:hover {{ background: #2B3240 !important; }}

/* Edge labels */
div.vis-network div.vis-label {{
  color: #C8D0E0 !important;
  background: transparent !important;
  font-size: 12px !important;
}}

/* 8) Scrollbars */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-thumb {{ background: #2F3640; border-radius: 8px; }}
::-webkit-scrollbar-track {{ background: #14181D; }}

/* 9) Sliders */
.stSlider > div[data-baseweb="slider"] > div {{ background: #2F3640 !important; }}
.stSlider [role="slider"] {{ background: #6C63FF !important; }}
</style>
""", unsafe_allow_html=True)
def normalize_model(m: str) -> str:
    if ":" in m:
        return m
    mapping = {
        "llama3.1": "llama3.1:8b",
        "mistral": "mistral:7b",
        "qwen2.5": "qwen2.5:7b",
        "phi4": "phi4:14b",
    }
    return mapping.get(m, f"{m}:latest")

# -------------------- Sidebar controls --------------------
with st.sidebar:
    st.title("🧠 Memory Graph Builder")
    st.caption("Paste notes → Structured JSON → Interactive Graph")

    st.subheader("Local LLM (Ollama required)")
    ollama_host = st.text_input("Ollama Host", value="http://localhost:11434")
    model = st.selectbox(
    "Model",
    options=[
        "llama3.1:8b",     # recommended default
        "mistral:7b",
        "qwen2.5:7b",
        "phi4:14b",
        "llama3.1:70b"     # advanced, large
    ],
    index=0
)

    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    top_p = st.slider("top_p", 0.1, 1.0, 0.9, 0.05)
    max_tokens = st.number_input("Max tokens (approx)", min_value=64, max_value=128000, value=30000, step=64)
    st.caption("Ollama must be installed and running. This app will help install/start it and pull the model if missing.")

    st.markdown("---")
    st.subheader("Visualization")
    layout_mode = st.selectbox("Layout", ["Hierarchical", "Force (Physics)"], index=1)  # default to Force
    show_3d = st.checkbox("Show 3D Plotly view (enhanced)", value=True)

    st.markdown("---")
    st.subheader("Colors (by kind)")
    color_defs = st.color_picker("Color: Definitions", "#1976D2")
    color_concepts = st.color_picker("Color: Concepts", "#43A047")
    color_equations = st.color_picker("Color: Equations", "#FBC02D")
    color_steps = st.color_picker("Color: Steps", "#E53935")
    color_examples = st.color_picker("Color: Examples", "#8E24AA")

# -------------------- Ollama helpers (install/start/check) --------------------
def run_cmd(cmd, cwd=None, timeout=None, shell=False) -> tuple[int, str, str]:
    try:
        p = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, shell=shell
        )
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out or "", err or ""
    except Exception as e:
        return 1, "", str(e)

def which(cmd_name: str) -> Optional[str]:
    return shutil.which(cmd_name)

def is_ollama_binary_available() -> bool:
    return which("ollama") is not None

def is_ollama_running(base_url: str, timeout=2.0) -> bool:
    try:
        r = requests.get(base_url.rstrip("/") + "/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False

def start_ollama_service() -> bool:
    system = platform.system().lower()
    if system == "windows":
        run_cmd(["powershell", "-Command", "Start-Process ollama -WindowStyle Hidden"])
        return True
    elif system == "darwin":
        run_cmd(["/usr/bin/open", "-a", "Ollama"])
        return True
    else:
        run_cmd(["bash", "-lc", "nohup ollama serve >/dev/null 2>&1 &"])
        return True

def ensure_model_present(model: str) -> tuple[bool, str]:
    # model may already include a size tag (e.g., "llama3.1:8b")
    target = normalize_model(model)
    name_tag = target.split(":", 1)
    name = name_tag[0]
    tag = name_tag[1] if len(name_tag) > 1 else ""

    code, out, err = run_cmd(["ollama", "list"])
    if code != 0:
        return False, f"Could not enumerate models: {err or out}"

    have = False
    for line in (out or "").splitlines():
        s = (line or "").strip()
        if not s:
            continue
        # Accept exact "name:tag" prefix OR columnar formats that contain both name and tag.
        if s.startswith(target) or ((name in s) and (tag in s if tag else True)):
            have = True
            break

    if not have:
        rc, pout, perr = run_cmd(["ollama", "pull", target])
        if rc != 0:
            return False, f"Model pull failed: {perr or pout}"

    return True, "Model available."

def install_ollama_windows_with_progress() -> tuple[bool, str]:
    import tempfile
    url = "https://ollama.com/download/OllamaSetup.exe"
    st.info("Downloading Ollama installer...")
    prog = st.progress(0, text="Starting download...")
    status = st.empty()
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            chunk = 1024 * 128
            got = 0
            with tempfile.NamedTemporaryFile(delete=False, suffix=".exe") as tmp:
                for i, data in enumerate(r.iter_content(chunk_size=chunk), start=1):
                    if not data:
                        continue
                    tmp.write(data)
                    got += len(data)
                    if total > 0:
                        frac = min(got / total, 1.0)
                        prog.progress(int(frac * 100), text=f"Downloading... {int(frac*100)}%")
                    else:
                        if i % 5 == 0:
                            prog.progress(min((i % 100), 99), text="Downloading...")
                path = tmp.name
        prog.progress(100, text="Download complete.")
        status.info("Launching Ollama installer. Please complete the setup window, then return here.")
        os.startfile(path)  # Windows only
        return True, "Installer launched. After setup, click 'Retry check'."
    except Exception as e:
        prog.empty()
        return False, f"Failed to download or launch installer: {e}"

def install_ollama_interactive() -> tuple[bool, str]:
    system = platform.system().lower()
    if system == "windows":
        return install_ollama_windows_with_progress()
    elif system == "darwin":
        if which("brew"):
            code, out, err = run_cmd(["brew", "install", "ollama"])
            if code == 0:
                return True, "Ollama installed via Homebrew."
            return False, f"Brew install failed: {err or out}"
        else:
            run_cmd(["/usr/bin/open", "https://ollama.com/download"])
            return False, "Opened Ollama download page. Install it, then return."
    else:
        code, out, err = run_cmd(["bash", "-lc", "curl -fsSL https://ollama.com/install.sh | sh"])
        if code == 0:
            return True, "Ollama installed."
        return False, f"Install failed: {err or out}"

def ollama_check_and_setup(base_url: str, model: str) -> bool:
    # 1) Installed?
    if not is_ollama_binary_available():
        st.error("Ollama is not installed.")
        c0, c1 = st.columns(2)
        with c0:
            if st.button("Install Ollama", key="install_ollama"):
                ok, msg = install_ollama_interactive()
                st.info(msg)
        with c1:
            if st.button("Retry check", key="retry_after_install"):
                st.rerun()
        return False

    # 2) Running?
    if not is_ollama_running(base_url):
        st.warning("Ollama daemon is not running. Attempting to start it...")
        if start_ollama_service():
            for _ in range(20):
                if is_ollama_running(base_url):
                    break
                time.sleep(0.5)
        if not is_ollama_running(base_url):
            c0, c1 = st.columns(2)
            with c0:
                st.error("Could not reach Ollama service after starting. Start it manually, then retry.")
            with c1:
                if st.button("Retry check", key="retry_after_start"):
                    st.rerun()
            return False

    # 3) Model present?
    ok, msg = ensure_model_present(normalize_model(model))
    if not ok:
        st.error(msg)
        base = model.split(":")[0]
        c0, c1 = st.columns(2)
        with c0:
            if st.button(f"Pull model '{base}' now", key="pull_model_now"):
                rc, pout, perr = run_cmd(["ollama", "pull", base])
                if rc == 0:
                    st.success("Model pulled successfully. Click 'Retry check'.")
                else:
                    st.error(f"Pull failed: {perr or pout}")
        with c1:
            if st.button("Retry check", key="retry_after_pull"):
                st.rerun()
        return False

    return True

# -------------------- Schema + prompts --------------------
SCHEMA_DOC = {
    "title": "KnowledgeGraph",
    "type": "object",
    "properties": {
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": ["definition", "concept", "equation", "example"]},
                    "summary": {"type": "string"},
                    "equation": {"type": "string"},
                    "children": {"type": "array", "items": {"$ref": "#/properties/concepts/items"}}
                },
                "required": ["name", "kind"]
            }
        }
    },
    "required": ["concepts"]
}

BASE_SYSTEM_INSTRUCTIONS = """
You are a hierarchical data extractor. Identify main topics and components and structure them into a nested JSON using the `children` array.

RULES:
1) Output ONLY JSON. No other text.
2) Be direct: avoid wrapper nodes like "List of". Connect items directly. e.g., DFA/NFA/PDA are direct children of "Automata".
3) Group logically: the top-level concept(s) must be the main subject(s) of the provided text.
"""

FEW_SHOT_EXAMPLES = """
INPUT: Automata is a model for computation. Its types include DFA, NFA, and PDA. A DFA has a set of states, an alphabet, and a start state.
JSON:
{
  "concepts": [
    {
      "name": "Automata",
      "kind": "definition",
      "summary": "A model for computation.",
      "children": [
        { "name": "DFA", "kind": "concept" },
        { "name": "NFA", "kind": "concept" },
        { "name": "PDA", "kind": "concept" }
      ]
    },
    {
      "name": "DFA",
      "kind": "definition",
      "summary": "A specific type of automaton.",
      "children": [
        { "name": "Finite Set of States", "kind": "concept" },
        { "name": "Alphabet", "kind": "concept" },
        { "name": "Start State", "kind": "concept" }
      ]
    }
  ]
}
"""

def build_prompt(user_text: str) -> str:
    schema_str = json.dumps(SCHEMA_DOC, ensure_ascii=False)
    return f"""{BASE_SYSTEM_INSTRUCTIONS}

{FEW_SHOT_EXAMPLES}

---
Create both hierarchical `children` and, where items are inherently ordered, you may imply sequential understanding with the content (no extra fields).

JSON_SCHEMA:
{schema_str}

INPUT_NOTES:
{user_text}
"""

# -------------------- Ollama call --------------------
def call_ollama(ollama_host: str, model: str, prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
    url = f"{ollama_host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {"temperature": temperature, "top_p": top_p, "num_predict": max_tokens},
        "format": "json"
    }
    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            chunks = []
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                if "response" in obj:
                    chunks.append(obj["response"])
                if obj.get("done"):
                    break
            return "".join(chunks).strip()
    except Exception as e:
        return json.dumps({"error": f"Ollama request failed: {e}"})

# -------------------- JSON extraction --------------------
def extract_json_maybe(text_response: str) -> (Optional[Dict[str, Any]], Optional[str]):
    if not text_response:
        return None, "Empty response."
    try:
        parsed = json.loads(text_response)
        return parsed, None
    except Exception:
        pass

    text = text_response
    start = text.find("{")
    if start == -1:
        return None, "No JSON object found."
    stack, end_index = [], None
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            stack.append("{")
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack:
                    end_index = i + 1
                    break
    if end_index is None:
        return None, "Unbalanced JSON braces."
    block = text[start:end_index]
    try:
        parsed = json.loads(block)
        return parsed, None
    except Exception as e:
        try:
            fixed = block.replace("'", '"')
            parsed = json.loads(fixed)
            return parsed, None
        except Exception:
            return None, f"Failed to parse JSON: {e}"

# -------------------- Simple fallback keyword extractor (unused now, kept for future) --------------------
STOPWORDS = set("""
a an the is are was were be been being of in on at to for with and or but which that this these those as by from about into through during including until
""".split())

def simple_extract_keywords(sentence: str, max_keywords=4) -> List[str]:
    s = sentence.strip()
    if not s:
        return []
    if _NLP:
        doc = _NLP(s)
        cands = []
        for nc in doc.noun_chunks:
            t = nc.text.strip()
            if 1 <= len(t.split()) <= 6:
                cands.append(t)
        for ent in doc.ents:
            if ent.text not in cands:
                cands.append(ent.text)
        if not cands:
            nouns = [tok.text for tok in doc if tok.pos_ in ("NOUN", "PROPN")]
            cands = nouns[:max_keywords]
        return cands[:max_keywords]
    else:
        s2 = re.sub(r'[^\w\s]', ' ', s)
        toks = [t for t in s2.split() if t.lower() not in STOPWORDS]
        if not toks:
            toks = s2.split()
        bigrams = [" ".join(toks[i:i+2]) for i in range(len(toks)-1)]
        cands = []
        for bg in bigrams:
            if bg not in cands:
                cands.append(bg)
        for t in toks:
            if t not in cands:
                cands.append(t)
        return cands[:max_keywords]

def human_readable_edge(label: str, src: str, dst: str) -> str:
    lbl = (label or "").lower().strip()
    if lbl == "contains":
        return f"{dst} is a type of {src}."
    if lbl == "next":
        return f"{src} is typically followed by {dst}."
    if lbl == "defines":
        return f"{src} defines or introduces {dst}."
    if lbl == "rel":
        return f"{src} is related to {dst}."
    return f"{src} → {dst} ({label})."

# -------------------- Node tooltip (plain text) --------------------
def build_node_title(node_name: str, attrs: dict, G: nx.DiGraph) -> str:
    kind = (attrs.get("kind", "concept") or "concept")
    summary = attrs.get("summary", "") or ""
    equation = attrs.get("equation", "") or ""
    parents = [u for u, v in G.in_edges(node_name) if G[u][v].get("label") == "contains"]
    children = [v for u, v in G.out_edges(node_name) if G[u][v].get("label") == "contains"]

    lines = [f"{node_name} ({kind})"]
    if summary:
        lines.append(summary)
    if equation:
        lines.append(f"Equation: {equation}")
    if parents:
        lines.append(f"Parent: {', '.join(parents)}")
    if children:
        lines.append(f"Children: {', '.join(children)}")
    return "\n".join(lines)

# -------------------- Build graphs --------------------
def build_networkx_graph_from_json(data: Dict[str, Any]) -> nx.DiGraph:
    G = nx.DiGraph()
    concepts = data.get("concepts", [])

    def process_node(node, parent=None):
        name = node.get("name")
        if not name:
            return
        G.add_node(name, **node)

        if parent:
            label = "contains"
            G.add_edge(parent, name, label=label, explanation=human_readable_edge(label, parent, name))

        children = node.get("children", []) or []
        for i, child_node in enumerate(children):
            process_node(child_node, parent=name)
            if i < len(children) - 1:
                a = child_node.get("name")
                b = children[i+1].get("name")
                if a and b:
                    label = "next"
                    G.add_edge(a, b, label=label, explanation=human_readable_edge(label, a, b))

    for root in concepts:
        process_node(root)
    return G

# -------------------- Renderers --------------------
def color_for_kind(kind: str, palette: Dict[str, str]) -> str:
    return {
        "definition": palette["definition"],
        "concept": palette["concept"],
        "equation": palette["equation"],
        "step": palette["step"],
        "example": palette["example"],
    }.get(kind, palette.get("concept", "#43A047"))

def render_pyvis(G: nx.DiGraph, palette: Dict[str, str], hierarchical: bool, theme: Dict[str, str]) -> str:
    net = Network(height="780px", width="100%",
                  bgcolor=theme["bg"], font_color=theme["font"],
                  directed=True, notebook=False)

    for node, attrs in G.nodes(data=True):
        kind = attrs.get("kind", "concept")
        node_shape = "box" if kind in ("definition", "equation", "example") else "ellipse"
        node_color = color_for_kind(kind, palette)
        title_text = build_node_title(node, attrs, G)
        net.add_node(
            node,
            label=node,
            title=title_text,  # plain text tooltip
            shape=node_shape,
            color=node_color,
            borderWidth=2,
            margin=8,
            font={'face': 'arial', 'color': theme["node_label"], 'size': 16}
        )

    for s, t, attrs in G.edges(data=True):
        lbl = attrs.get("label", "")
        expl = attrs.get("explanation") or human_readable_edge(lbl, s, t)
        net.add_edge(
            s, t,
            label=lbl,
            title=expl,
            arrows="to",
            smooth={'type': 'dynamic'},  # keeps bezier curve
            color={'color': theme["edge"], 'inherit': False, 'opacity': 0.6},
            dashes=False,
            chosen=False,                 # no bold-on-hover
            font={
                'size': 11,
                'face': 'arial',
                'color': 'rgba(200,200,200,0.6)',  # blend with line
                'strokeWidth': 0,
                'align': 'horizontal'              # draw along the edge path
            },
            labelHighlightBold=False              # keep same weight on hover
        )

    options_json = """
    {
      "layout": {
        "hierarchical": {
          "enabled": %s,
          "direction": "UD",
          "sortMethod": "directed",
          "levelSeparation": 170,
          "nodeSpacing": 140,
          "treeSpacing": 220
        }
      },
      "physics": {
        "enabled": %s,
        "solver": "forceAtlas2Based",
        "stabilization": {"enabled": true, "iterations": 400}
      },
      "interaction": {
        "dragNodes": true,
        "dragView": true,
        "zoomView": true,
        "hover": true,
        "navigationButtons": true,
        "tooltipDelay": 60
      },
      "edges": {
        "smooth": {"enabled": true, "type": "cubicBezier", "roundness": 0.6},
        "color": {"inherit": false}
      }
    }
    """ % ("true" if hierarchical else "false", "false" if hierarchical else "true")

    net.set_options(options_json)

    # Save and return HTML as string
    tmp_path = "_pyvis_graph.html"
    net.save_graph(tmp_path)
    with open(tmp_path, "r", encoding="utf-8") as f:
        html = f.read()
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return html

def render_plotly_3d(G: nx.DiGraph, palette: Dict[str, str], theme: Dict[str, str]) -> go.Figure:
    pos = nx.spring_layout(G, dim=3, seed=42)

    # Nodes
    xs, ys, zs, hovertexts, labels, colors = [], [], [], [], [], []
    for n, attrs in G.nodes(data=True):
        x, y, z = pos[n]
        xs.append(x); ys.append(y); zs.append(z)
        kind = attrs.get("kind", "concept")
        summary = attrs.get("summary") or ""
        eq = attrs.get("equation") or ""

        htxt = f"{n} ({kind})"
        if summary:
            htxt += f"\n{summary}"
        if eq:
            htxt += f"\nEq: {eq}"
        hovertexts.append(htxt)

        labels.append(n)
        colors.append(color_for_kind(kind, {
            "definition": palette["definition"],
            "concept": palette["concept"],
            "equation": palette["equation"],
            "step": palette["step"],
            "example": palette["example"],
        }))

    # Edges with Bezier + mid-point hover
    edge_x, edge_y, edge_z = [], [], []
    mid_x, mid_y, mid_z, mid_text = [], [], [], []
    cone_x, cone_y, cone_z, cone_u, cone_v, cone_w = [], [], [], [], [], []

    for (u, v, attrs) in G.edges(data=True):
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]

        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        cz = (z0 + z1) / 2 + 0.03

        steps = 10
        last_x, last_y, last_z = x0, y0, z0
        for t in range(1, steps + 1):
            tau = t / steps
            bx = (1 - tau) ** 2 * x0 + 2 * (1 - tau) * tau * cx + tau ** 2 * x1
            by = (1 - tau) ** 2 * y0 + 2 * (1 - tau) * tau * cy + tau ** 2 * y1
            bz = (1 - tau) ** 2 * z0 + 2 * (1 - tau) * tau * cz + tau ** 2 * z1
            edge_x += [last_x, bx, None]
            edge_y += [last_y, by, None]
            edge_z += [last_z, bz, None]
            last_x, last_y, last_z = bx, by, bz

        # Hover for edge
        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2
        mz = (z0 + z1) / 2
        text = attrs.get("explanation") or human_readable_edge(attrs.get("label", ""), u, v)
        mid_x.append(mx); mid_y.append(my); mid_z.append(mz); mid_text.append(text)

        # Arrowhead vector
        dirx, diry, dirz = (x1 - x0), (y1 - y0), (z1 - z0)
        cone_x.append(x1); cone_y.append(y1); cone_z.append(z1)
        cone_u.append(dirx); cone_v.append(diry); cone_w.append(dirz)

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color=active_theme["edge"], width=2),
        hoverinfo='none',
        showlegend=False
    )

    edge_hover_trace = go.Scatter3d(
        x=mid_x, y=mid_y, z=mid_z,
        mode='markers',
        marker=dict(size=6, color='rgba(0,0,0,0)'),
        hovertext=mid_text,
        hoverinfo='text',
        showlegend=False
    )

    node_trace = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers+text',
        text=labels,
        textposition='top center',
        hovertext=hovertexts,
        hoverinfo='text',
        marker=dict(
            size=8, color=colors, opacity=0.98,
            line=dict(width=1, color=active_theme["font"])
        ),
        textfont=dict(color=active_theme["font"])
    )

    cone_trace = go.Cone(
        x=cone_x, y=cone_y, z=cone_z,
        u=cone_u, v=cone_v, w=cone_w,
        sizemode='absolute', sizeref=0.10, anchor='tip',
        showscale=False,
        colorscale=[[0, active_theme["edge_cone"]], [1, active_theme["edge_cone"]]],
        opacity=0.55
    )

    fig = go.Figure(data=[edge_trace, edge_hover_trace, cone_trace, node_trace])
    fig.update_layout(
        paper_bgcolor=active_theme["plot_bg"],
        scene=dict(
            xaxis=dict(visible=False, backgroundcolor=active_theme["plot_bg"]),
            yaxis=dict(visible=False, backgroundcolor=active_theme["plot_bg"]),
            zaxis=dict(visible=False, backgroundcolor=active_theme["plot_bg"]),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        hoverlabel=dict(bgcolor=active_theme["plot_bg"],
                        font_size=12, font_color=active_theme["plot_font"], font_family="Arial"),
        font=dict(color=active_theme["plot_font"])
    )
    return fig

# -------------------- UI --------------------
st.header("Memory Graph Builder")
st.write("Paste notes, or upload a .txt/.pdf. Uses a local LLM (Ollama) to structure content, then renders 2D/3D graphs.")

default_text = """Automata is a mathematical model which performs computation using states and transitions.
Types include DFA, NFA, PDA, and Turing Machines.
A DFA has a finite set of states, an alphabet, a transition function, a start state and a set of accept states.
Quadratic solution example: Solve ax^2 + bx + c = 0 using the quadratic formula.
"""

inp_mode = st.radio("Input mode", ["Paste Text", "Upload .txt", "Upload .pdf"], index=0)
user_text = ""

if inp_mode == "Paste Text":
    user_text = st.text_area("Your notes:", value=default_text, height=260)
elif inp_mode == "Upload .txt":
    up = st.file_uploader("Upload a .txt file", type=["txt"])
    if up is not None:
        user_text = up.read().decode("utf-8")
        st.text_area("Text from file:", value=user_text, height=260)
elif inp_mode == "Upload .pdf":
    up = st.file_uploader("Upload a .pdf file", type=["pdf"])
    if up is not None:
        try:
            pdf_document = fitz.open(stream=up.read(), filetype="pdf")
            extracted_text = ""
            for page in pdf_document:
                extracted_text += page.get_text()
            user_text = extracted_text
            st.text_area("Extracted Text from PDF:", value=user_text, height=260)
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")
            user_text = ""

col1, col2 = st.columns(2)
with col1:
    build_btn = st.button("🚀 Build Graph")
with col2:
    st.download_button("⬇️ Download Notes", data=user_text or "", file_name="notes.txt",
                       mime="text/plain", disabled=(not user_text))

palette_map = {
    "definition": color_defs,
    "concept": color_concepts,
    "equation": color_equations,
    "step": color_steps,
    "example": color_examples
}

# -------------------- Build and render (Ollama required) --------------------
if build_btn and user_text.strip():
    # 1) Ensure Ollama + model are ready (installs/starts/pulls if needed)
    with st.spinner("Preparing local LLM (Ollama)..."):
        if not ollama_check_and_setup(ollama_host, model):
            st.stop()

    # 2) Call Ollama to get JSON and build graph (no fallback)
    with st.spinner("Building graph via Ollama..."):
        selected_model = normalize_model(model) # ensure size tag like :8b
        prompt = build_prompt(user_text)
        raw = call_ollama(ollama_host, selected_model, prompt, temperature, top_p, int(max_tokens))
        parsed_json, parse_error = extract_json_maybe(raw)
        if parse_error or not isinstance(parsed_json, dict):
            st.error("Failed to parse model output as JSON. Confirm the model and try again.")
            st.stop()
        G = build_networkx_graph_from_json(parsed_json)

    # 3) Guard empty and render
    if G.number_of_nodes() == 0:
        st.warning("No nodes were extracted from the input. Please add more content.")
    else:
        # Render 2D
        html = render_pyvis(G, palette_map, hierarchical=(layout_mode == "Hierarchical"), theme=active_theme)
        # Render 3D if enabled
        fig3d = render_plotly_3d(G, palette_map, active_theme) if show_3d else None

        # Persist in session
        st.session_state.G = G
        st.session_state.pyvis_html = html
        st.session_state.plotly_fig3d = fig3d
        st.session_state.graph_built = True

        # Display
        import streamlit.components.v1 as components
        components.html(html, height=780, scrolling=True)
        if fig3d is not None:
            st.subheader("3D View (Plotly)")
            st.plotly_chart(fig3d, use_container_width=True, key="fig3d_build")

        # Export
        with st.expander("Export / Save"):
            nodes_export = [{"name": n, **attrs} for n, attrs in G.nodes(data=True)]
            edges_export = [{"source": s, "target": t, **(attrs or {})} for s, t, attrs in G.edges(data=True)]
            export_data = {"nodes": nodes_export, "edges": edges_export}
            st.download_button("Download graph JSON", data=json.dumps(export_data, indent=2),
                               file_name="graph.json", mime="application/json", key="dl_json_build")
            st.download_button("Download Graph HTML (2D)", data=st.session_state.pyvis_html,
                               file_name="graph.html", mime="text/html", key="dl_html2d_build")
            if st.session_state.plotly_fig3d is not None:
                try:
                    html3d = st.session_state.plotly_fig3d.to_html(full_html=True, include_plotlyjs="cdn")
                    st.download_button("Download Graph 3D (HTML)", data=html3d,
                                       file_name="graph_3d.html", mime="text/html", key="dl_html3d_build")
                except Exception as e:
                    st.warning(f"Could not create 3D HTML export: {e}")

    st.success("Done — explore the graph. Drag, zoom, and hover nodes/edges for details.")
else:
    st.info("Paste notes above and click 'Build Graph' to generate a memory graph.")

# -------------------- Persistent render on reruns --------------------
if st.session_state.graph_built and st.session_state.G is not None:
    import streamlit.components.v1 as components
    components.html(st.session_state.pyvis_html, height=780, scrolling=True)
    if show_3d and st.session_state.plotly_fig3d is not None:
        st.subheader("3D View (Plotly)")
        st.plotly_chart(st.session_state.plotly_fig3d, use_container_width=True, key="fig3d_persist")
    with st.expander("Export / Save"):
        nodes_export = [{"name": n, **attrs} for n, attrs in st.session_state.G.nodes(data=True)]
        edges_export = [{"source": s, "target": t, **(attrs or {})} for s, t, attrs in st.session_state.G.edges(data=True)]
        export_data = {"nodes": nodes_export, "edges": edges_export}
        st.download_button("Download graph JSON", data=json.dumps(export_data, indent=2),
                           file_name="graph.json", mime="application/json", key="dl_json_persist")
        st.download_button("Download Graph HTML (2D)", data=st.session_state.pyvis_html,
                           file_name="graph.html", mime="text/html", key="dl_html2d_persist")
        if st.session_state.plotly_fig3d is not None:
            try:
                html3d = st.session_state.plotly_fig3d.to_html(full_html=True, include_plotlyjs="cdn")
                st.download_button("Download Graph 3D (HTML)", data=html3d,
                                   file_name="graph_3d.html", mime="text/html", key="dl_html3d_persist")
            except Exception as e:
                st.warning(f"Could not create 3D HTML export: {e}")

st.markdown("---")
st.markdown("""
How it works
1) Ollama (local LLM) converts notes into a strict JSON schema.
2) The app builds a directed graph and renders:
   - 2D: PyVis/vis-network with dark UI and plain-text hovers.
   - 3D: Plotly with curved edges, arrowheads, and mid-edge hover markers.
3) Exports: interactive 2D HTML, interactive 3D HTML, and JSON nodes/edges.
""")
