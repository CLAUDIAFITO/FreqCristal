import os, json, io, wave
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# =========================================================
# Credenciais (.env ou st.secrets)
# =========================================================
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY")

try:
    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
except Exception:
    sb = None

# =========================================================
# Página / Estilos
# =========================================================
st.set_page_config(
    page_title="Frequências • Cama de Cristal",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.main .block-container { padding-top: 1rem; padding-bottom: 1rem; }
.modern-card { background: white; padding: 1.25rem; border-radius: 14px; 
               box-shadow: 0 6px 18px rgba(0,0,0,.07); border: 1px solid #eef; }
.controls button { margin-right: .5rem; }
.badge { display:inline-block; padding:.15rem .5rem; border-radius:999px; background:#f2f6ff; border:1px solid #e5eaff; font-size:.8rem; margin-right:.25rem; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# Dados estáticos / Mapas
# =========================================================
INTENCOES = {
    "Relaxamento": {"base": ["SOL174","SOL432","CHAKRA_CARDIACO"]},
    "Aterramento": {"base": ["SOL396","CHAKRA_RAIZ"]},
    "Clareza mental": {"base": ["SOL741","CHAKRA_TERCEIRO_OLHO"]},
    "Harmonização do coração": {"base": ["SOL528","SOL639","CHAKRA_CARDIACO"]},
    "Limpeza/Desbloqueio": {"base": ["SOL417","SOL741"]},
}

CHAKRA_MAP = {
    "Raiz":"raiz",
    "Sacral":"sacral",
    "Plexo":"plexo",
    "Cardíaco":"cardiaco",
    "Laríngeo":"laringeo",
    "Terceiro Olho":"terceiro_olho",
    "Coronal":"coronal",
    "Nenhum":None
}

# =========================================================
# Funções utilitárias — catálogo e protocolo
# =========================================================
def carregar_catalogo_freq() -> pd.DataFrame:
    """Carrega catálogo do Supabase; fallback mínimo local se não houver conexão."""
    if sb:
        try:
            data = sb.table("frequencies").select("*").execute().data
            return pd.DataFrame(data or [])
        except Exception:
            return pd.DataFrame([])
    # fallback local mínimo
    return pd.DataFrame([
        {"code":"SOL174","nome":"Solfeggio 174 Hz","hz":174,"tipo":"solfeggio","chakra":None,"cor":None},
        {"code":"SOL396","nome":"Solfeggio 396 Hz","hz":396,"tipo":"solfeggio","chakra":"raiz","cor":"vermelho"},
        {"code":"SOL417","nome":"Solfeggio 417 Hz","hz":417,"tipo":"solfeggio","chakra":"sacral","cor":"laranja"},
        {"code":"SOL528","nome":"Solfeggio 528 Hz","hz":528,"tipo":"solfeggio","chakra":"cardiaco","cor":"verde"},
        {"code":"SOL639","nome":"Solfeggio 639 Hz","hz":639,"tipo":"solfeggio","chakra":"cardiaco","cor":"verde"},
        {"code":"SOL741","nome":"Solfeggio 741 Hz","hz":741,"tipo":"solfeggio","chakra":"laringeo","cor":"azul"},
        {"code":"SOL852","nome":"Solfeggio 852 Hz","hz":852,"tipo":"solfeggio","chakra":"terceiro_olho","cor":"anil"},
        {"code":"SOL963","nome":"Solfeggio 963 Hz","hz":963,"tipo":"solfeggio","chakra":"coronal","cor":"violeta"},
        {"code":"CHAKRA_CARDIACO","nome":"Chakra Cardíaco","hz":639,"tipo":"chakra","chakra":"cardiaco","cor":"verde"},
        {"code":"CH
