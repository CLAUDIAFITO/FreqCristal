import os, json
from datetime import datetime
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# =========================================================
# Config inicial (credenciais via .env OU Streamlit Secrets)
# =========================================================
load_dotenv()

# lida com .env e com st.secrets (deploy Streamlit Cloud)
SUPABASE_URL = os.getenv("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY")

# Cliente Supabase (opcional – app funciona parcialmente sem conexão)
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
# Funções utilitárias
# =========================================================
def carregar_catalogo_freq() -> pd.DataFrame:
    """
    Tenta carregar o catálogo a partir do Supabase; se não houver conexão,
    usa um fallback mínimo em memória.
    """
    if sb:
        try:
            data = sb.table("frequencies").select("*").execute().data
            return pd.DataFrame(data or [])
        except Exception:
            # Em caso de erro na consulta, evita quebrar a UI
            return pd.DataFrame([])
    # Fallback local (mínimo)
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
        {"code":"CHAKRA_RAIZ","nome":"Chakra Raiz","hz":396,"tipo":"chakra","chakra":"raiz","cor":"vermelho"},
        {"code":"CHAKRA_TERCEIRO_OLHO","nome":"Chakra Terceiro Olho","hz":852,"tipo":"chakra","chakra":"terceiro_olho","cor":"anil"},
        {"code":"SOL432","nome":"Acorde 432 Hz","hz":432,"tipo":"custom","chakra":None,"cor":None},
    ])

def gerar_protocolo(intencao: str, chakra_alvo: str|None, duracao_min: int, catalogo: pd.DataFrame) -> pd.DataFrame:
    """
    Gera uma playlist com duração distribuída entre as frequências base da intenção.
    Reforça o chakra alvo se estiver no catálogo.
    """
    base = INTENCOES.get(intencao, {"base": []})
    sel = list(dict.fromkeys(base["base"]))  # sem repetição mantendo ordem

    if chakra_alvo:
        ccode = f"CHAKRA_{chakra_alvo.upper()}"
        if (catalogo["code"] == ccode).any() and ccode not in sel:
            sel.append(ccode)

    if not sel:
        sel = ["SOL432","SOL528"]
        sel = [c for c in sel if (catalogo["code"] == c).any()]

    total = max(1, int(duracao_min)) * 60
    bloco = max(90, total // max(1, len(sel)))

    linhas = []
    for i, code in enumerate(sel, start=1):
        row = catalogo.loc[catalogo["code"] == code]
        if row.empty:
            continue
        row = row.iloc[0]
        linhas.append({
            "ordem": i,
            "code": code,
            "nome": row.get("nome"),
            "hz": float(row.get("hz")),
            "duracao_seg": int(bloco),
            "chakra": row.get("chakra"),
            "cor": row.get("cor")
        })

    usado = sum(l["duracao_seg"] for l in linhas)
    if linhas and usado < total:
        linhas[-1]["duracao_seg"] += total - usado

    return pd.DataFrame(linhas)

# =========================================================
# UI
# =========================================================
st.title("💫 Frequências — Cama de Cristal")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Gerador", "Pacientes", "Sessões", "Catálogo", "Admin"])

# ---------------- Gerador ----------------
with tab1:
    st.subheader("Gerar protocolo de frequências")
    catalogo = carregar_catalogo_freq()

    col1, col2, col3 = st.columns(3)
    intencao = col1.selectbox("Intenção", list(INTENCOES.keys()), key="ger_intencao")
    chakra_label = col2.selectbox("Chakra alvo", list(CHAKRA_MAP.keys()), index=list(CHAKRA_MAP.keys()).index("Nenhum"), key="ger_chakra")
    chakra_alvo = CHAKRA_MAP[chakra_label]
    duracao = int(col3.number_input("Duração (min)", 10, 120, 30, step=5, key="ger_dur"))

    if st.button("Gerar protocolo", type="primary", key="btn_gerar_protocolo"):
        if catalogo.empty or "code" not in catalogo.columns:
            st.error("Catálogo vazio ou inválido. Vá na aba **Admin** e importe o `seed_frequencies.csv`.")
        else:
            plano = gerar_protocolo(intencao, chakra_alvo, duracao, catalogo)
            if plano.empty:
                st.warning("Não foi possível montar a playlist com as frequências atuais do catálogo.")
            else:
                st.dataframe(plano, use_container_width=True, hide_index=True)
                st.download_button(
                    "Baixar CSV",
                    data=plano.to_csv(index=False).encode("utf-8"),
                    file_name="protocolo.csv",
                    mime="text/csv",
                    key="btn_dl_protocolo"
                )

# ---------------- Pacientes ----------------
with tab2:
    st.subheader("Pacientes")
    if not sb:
        st.info("Conecte seu Supabase (defina SUPABASE_URL e SUPABASE_KEY) para ativar cadastros.")
    else:
        with st.form("pac_form"):
            nome = st.text_input("Nome", key="pac_nome")
            nasc = st.date_input("Nascimento", value=None, key="pac_nasc")
            notas = st.text_area("Notas", key="pac_notas")
            if st.form_submit_button("Salvar", use_container_width=False):
                if nome.strip():
                    try:
                        sb.table("patients").insert([{
                            "nome": nome.strip(),
                            "nascimento": str(nasc) if nasc else None,
                            "notas": notas
                        }]).execute()
                        st.success("Paciente salvo!")
                    except Exception as e:
                        st.error(f"Erro ao salvar paciente: {e}")
                else:
                    st.warning("Informe o nome do paciente.")

        try:
            pats = sb.table("patients").select("*").order("created_at", desc=True).execute().data
        except Exception:
            pats = []
        if pats:
            st.dataframe(pd.DataFrame(pats)[["nome","nascimento","notas","created_at"]], use_container_width=True, hide_index=True)
        else:
            st.caption("Nenhum paciente cadastrado ainda.")

# ---------------- Sessões ----------------
with tab3:
    st.subheader("Sessões")
    if not sb:
        st.info("Conecte o Supabase para salvar sessões.")
    else:
        try:
            pats = sb.table("patients").select("id,nome").execute().data
        except Exception:
            pats = []

        mapa = {p["nome"]: p["id"] for p in (pats or [])}
        nome = st.selectbox("Paciente", list(mapa.keys()) if mapa else ["—"], key="sess_paciente")

        catalogo = carregar_catalogo_freq()
        intencao2 = st.selectbox("Intenção", list(INTENCOES.keys()), key="sess_intencao")
        chakra_label2 = st.selectbox("Chakra alvo", list(CHAKRA_MAP.keys()), index=list(CHAKRA_MAP.keys()).index("Nenhum"), key="sess_chakra")
        chakra_alvo2 = CHAKRA_MAP[chakra_label2]
        dur2 = int(st.number_input("Duração (min)", 10, 120, 30, step=5, key="sess_dur"))

        if st.button("Gerar + salvar", type="primary", key="sess_btn_salvar"):
            if not mapa:
                st.warning("Cadastre pelo menos um paciente na aba **Pacientes**.")
            elif catalogo.empty or "code" not in catalogo.columns:
                st.error("Catálogo vazio ou inválido. Importe o seed na aba **Admin**.")
            else:
                plano = gerar_protocolo(intencao2, chakra_alvo2, dur2, catalogo)
                payload = {
                    "patient_id": mapa.get(nome),
                    "data": datetime.utcnow().isoformat(),
                    "duracao_min": dur2,
                    "intencao": intencao2,
                    "chakra_alvo": chakra_alvo2,
                    "status": "rascunho",
                    "protocolo": json.loads(plano.to_json(orient="records"))
                }
                try:
                    s = sb.table("sessions").insert([payload]).execute().data
                    if s:
                        st.success("Sessão criada!")
                    else:
                        st.error("Falha ao criar sessão.")
                except Exception as e:
                    st.error(f"Erro ao criar sessão: {e}")

        # listar sessões
        try:
            sess = sb.table("sessions").select("id,data,intencao,duracao_min,status").order("created_at", desc=True).execute().data
        except Exception:
            sess = []
        if sess:
            st.dataframe(pd.DataFrame(sess), use_container_width=True, hide_index=True)
        else:
            st.caption("Nenhuma sessão registrada ainda.")

# ---------------- Catálogo ----------------
with tab4:
    st.subheader("Catálogo")
    df = carregar_catalogo_freq()
    expected = ["code","nome","hz","tipo","chakra","cor"]
    has_cols = set(expected).issubset(df.columns)

    if df.empty or not has_cols:
        st.info(
            "Catálogo ainda não está carregado ou está sem colunas esperadas. "
            "Vá até a aba **Admin** e importe o arquivo `seed_frequencies.csv`."
        )
        # Mostra o que tiver (útil para depurar)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df[expected], use_container_width=True, hide_index=True)

# ---------------- Admin ----------------
with tab5:
    st.subheader("Admin")
    st.caption("Importe `seed_frequencies.csv` para a tabela `frequencies`.")
    st.code(f"URL set? {bool(SUPABASE_URL)} | KEY set? {bool(SUPABASE_KEY)} | Client? {bool(sb)}")

    up = st.file_uploader("seed_frequencies.csv", type=["csv"], key="admin_seed_upload")
    if up and sb and st.button("Importar agora", key="admin_btn_importar"):
        try:
            df = pd.read_csv(up)

            # Converte strings vazias para None (NULL no Postgres)
            for col in ["chakra", "cor", "descricao", "code", "nome"]:
                if col in df.columns:
                    df[col] = df[col].replace({"": None})

            # Converte hz para número
            if "hz" in df.columns:
                df["hz"] = pd.to_numeric(df["hz"], errors="coerce")

            # Converte tipo para um dos enums válidos
            if "tipo" in df.columns and df["tipo"].dtype == object:
                df["tipo"] = df["tipo"].str.lower().replace({"color":"cor"})

            rows = df.to_dict(orient="records")
            ok, fail = 0, 0
            for r in rows:
                try:
                    sb.table("frequencies").upsert(r).execute()
                    ok += 1
                except Exception:
                    fail += 1
            st.success(f"Importadas/atualizadas: {ok}. Falhas: {fail}.")
        except Exception as e:
            st.error(f"Erro ao importar seed: {e}")
    elif not sb:
        st.warning("Defina SUPABASE_URL e SUPABASE_KEY para habilitar a importação.")
