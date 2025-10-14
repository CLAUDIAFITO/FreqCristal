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
        {"code":"CHAKRA_RAIZ","nome":"Chakra Raiz","hz":396,"tipo":"chakra","chakra":"raiz","cor":"vermelho"},
        {"code":"CHAKRA_TERCEIRO_OLHO","nome":"Chakra Terceiro Olho","hz":852,"tipo":"chakra","chakra":"terceiro_olho","cor":"anil"},
        {"code":"SOL432","nome":"Acorde 432 Hz","hz":432,"tipo":"custom","chakra":None,"cor":None},
    ])

def gerar_protocolo(intencao: str, chakra_alvo: str|None, duracao_min: int, catalogo: pd.DataFrame) -> pd.DataFrame:
    """Gera playlist distribuindo duração entre frequências da intenção; reforça chakra alvo."""
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
# Áudio — síntese WAV e players WebAudio
# =========================================================
def synth_tone_wav(freq: float, seconds: float = 20.0, sr: int = 22050, amp: float = 0.2) -> bytes:
    """Gera amostra WAV mono 16-bit com fade-in/out curto para evitar clicks."""
    t = np.linspace(0, seconds, int(sr*seconds), endpoint=False)
    wavef = np.sin(2*np.pi*freq*t)
    ramp = max(1, int(sr * 0.01))  # 10ms
    env = np.ones_like(wavef)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    y = (wavef * env * amp).astype(np.float32)
    y_int16 = np.int16(np.clip(y, -1, 1) * 32767)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(y_int16.tobytes())
    return buf.getvalue()

def webaudio_player_html(plano: pd.DataFrame) -> str:
    """Player WebAudio para tocar a sequência inteira no navegador."""
    items = [
        {"hz": float(r["hz"]), "dur": int(r["duracao_seg"]), "label": f'{int(r["hz"])} Hz — {r["nome"] or r["code"]}'}
        for _, r in plano.iterrows()
    ]
    playlist_json = json.dumps(items)
    html = f"""
<div class="modern-card">
  <div><strong>Player (WebAudio):</strong> toca no navegador, sem baixar arquivos.</div>
  <div class="controls" style="margin-top:.5rem;">
    <button id="btnPlay">▶️ Play</button>
    <button id="btnPause">⏸️ Pause</button>
    <button id="btnStop">⏹️ Stop</button>
  </div>
  <div id="status" style="margin-top:.5rem; font-size:.9rem;"></div>
  <div id="now" style="margin-top:.25rem;"></div>
</div>
<script>
const playlist = {playlist_json};
let ctx = null, osc = null, gain = null;
let idx = 0;
let playing = false;
let stepTimer = null;

function fmtSec(s) {{
  const m = Math.floor(s/60), r = s%60;
  return m + "m " + r + "s";
}}

function updateStatus() {{
  const item = playlist[idx] || null;
  let text = "Pronto.";
  if (playing && item) {{
    text = `Tocando: ${{item.label}} | Etapa ${{idx+1}}/${{playlist.length}} (~${{fmtSec(item.dur)}})`;
  }}
  document.getElementById("status").textContent = text;
}}

function stopAll() {{
  playing = false;
  if (stepTimer) {{ clearTimeout(stepTimer); stepTimer = null; }}
  if (osc) {{ try {{ osc.stop(); }} catch(e){{}} osc.disconnect(); osc = null; }}
  if (gain) {{ gain.disconnect(); gain = null; }}
  document.getElementById("now").textContent = "";
  updateStatus();
}}

function playStep() {{
  if (!playing) return;
  if (idx >= playlist.length) {{
    stopAll();
    return;
  }}
  const item = playlist[idx];

  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  osc = ctx.createOscillator();
  gain = ctx.createGain();

  osc.type = "sine";
  osc.frequency.value = item.hz;
  gain.gain.setValueAtTime(0.0001, ctx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.2, ctx.currentTime + 0.05);
  osc.connect(gain).connect(ctx.destination);
  osc.start();

  document.getElementById("now").textContent = item.label;
  updateStatus();

  const dur = Math.max(1, item.dur);
  stepTimer = setTimeout(() => {{
    if (!playing) return;
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.05);
    try {{ osc.stop(ctx.currentTime + 0.06); }} catch(e) {{}}
    idx += 1;
    setTimeout(() => playStep(), 100);
  }}, dur * 1000);
}}

document.getElementById("btnPlay").onclick = () => {{
  if (!playlist.length) return;
  if (!playing) {{ playing = true; if (idx >= playlist.length) idx = 0; }}
  playStep();
}};
document.getElementById("btnPause").onclick = () => {{
  playing = false;
  if (stepTimer) {{ clearTimeout(stepTimer); stepTimer = null; }}
  if (gain) gain.gain.setTargetAtTime(0.0001, ctx.currentTime, 0.02);
  if (osc) try {{ osc.stop(ctx.currentTime + 0.05); }} catch(e) {{}}
  updateStatus();
}};
document.getElementById("btnStop").onclick = () => {{
  idx = 0;
  stopAll();
}};
updateStatus();
</script>
"""
    return html

def webaudio_single_html(freq_hz: float, seconds: int = 20) -> str:
    """Player WebAudio para tocar UMA frequência por X segundos."""
    return f"""
<div class="modern-card">
  <div><strong>{int(freq_hz)} Hz</strong> — tocar no navegador (WebAudio)</div>
  <div class="controls" style="margin-top:.5rem;">
    <button id="s_play">▶️ Play</button>
    <button id="s_stop">⏹️ Stop</button>
  </div>
  <div id="s_status" style="margin-top:.5rem; font-size:.9rem;"></div>
</div>
<script>
let ctx = null, osc = null, gain = null, timer = null;
const freq = {float(freq_hz)};
const dur = {int(seconds)};
function stopAll(){{
  if (timer) {{ clearTimeout(timer); timer = null; }}
  if (osc) {{ try{{osc.stop();}}catch(e){{}} osc.disconnect(); osc = null; }}
  if (gain) {{ gain.disconnect(); gain = null; }}
  document.getElementById("s_status").textContent = "Parado.";
}}
document.getElementById("s_play").onclick = () => {{
  stopAll();
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  osc = ctx.createOscillator();
  gain = ctx.createGain();
  osc.type = "sine";
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0.0001, ctx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.2, ctx.currentTime + 0.05);
  osc.connect(gain).connect(ctx.destination);
  osc.start();
  document.getElementById("s_status").textContent = "Tocando " + freq + " Hz (" + dur + "s)";
  timer = setTimeout(() => {{ stopAll(); }}, dur * 1000);
}};
document.getElementById("s_stop").onclick = () => stopAll();
</script>
"""

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
    chakra_label = col2.selectbox("Chakra alvo", list(CHAKRA_MAP.keys()),
                                  index=list(CHAKRA_MAP.keys()).index("Nenhum"), key="ger_chakra")
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
                st.subheader("Protocolo sugerido")
                st.dataframe(plano, use_container_width=True, hide_index=True)

                # Player do protocolo inteiro
                from streamlit.components.v1 import html as st_html
                st.markdown("#### ▶️ Tocar protocolo no navegador (WebAudio)")
                st_html(webaudio_player_html(plano), height=260)

                # Selecionar e tocar UMA frequência do protocolo (label -> hz)
                st.markdown("#### 🔎 Tocar frequência específica do protocolo")
                opcoes_labels = []
                label_to_hz = {}
                for i, r in enumerate(plano.itertuples(index=False), start=1):
                    hz = float(getattr(r, "hz"))
                    nome = getattr(r, "nome") or getattr(r, "code")
                    label = f"{int(hz)} Hz — {nome}"
                    if label in label_to_hz:
                        label = f"{label} (#{i})"  # garante unicidade
                    opcoes_labels.append(label)
                    label_to_hz[label] = hz

                sel_label = st.selectbox(
                    "Escolha a etapa (uma frequência)",
                    opcoes_labels,
                    key="play_sel_protocolo"
                )
                hz_escolhido = label_to_hz[sel_label]

                colA, colB = st.columns([0.6, 0.4])
                with colA:
                    st_html(webaudio_single_html(hz_escolhido, seconds=20), height=180)
                with colB:
                    wav_bytes = synth_tone_wav(hz_escolhido, seconds=20.0, sr=22050, amp=0.2)
                    st.download_button(
                        "Baixar WAV (20s) desta frequência",
                        data=wav_bytes,
                        file_name=f"{int(hz_escolhido)}Hz_preview.wav",
                        mime="audio/wav",
                        key="dl_wav_freq_unica"
                    )

                # Prévias em WAV para cada etapa
                st.markdown("#### 🎧 Prévias em WAV (20s por etapa)")
                for _, r in plano.iterrows():
                    hz = float(r["hz"])
                    wav_bytes = synth_tone_wav(hz, seconds=20.0, sr=22050, amp=0.2)
                    colX, colY = st.columns([0.7, 0.3])
                    with colX:
                        st.audio(wav_bytes, format="audio/wav", start_time=0)
                        st.caption(f'{int(hz)} Hz — {r["nome"] or r["code"]} (prévia 20s)')
                    with colY:
                        st.download_button(
                            "Baixar WAV (20s)",
                            data=wav_bytes,
                            file_name=f'{int(hz)}Hz_preview.wav',
                            mime="audio/wav",
                            key=f'dl_wav_{int(hz)}'
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
            st.dataframe(pd.DataFrame(pats)[["nome","nascimento","notas","created_at"]],
                         use_container_width=True, hide_index=True)
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
        chakra_label2 = st.selectbox("Chakra alvo", list(CHAKRA_MAP.keys()),
                                     index=list(CHAKRA_MAP.keys()).index("Nenhum"), key="sess_chakra")
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
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df[expected], use_container_width=True, hide_index=True)

        # Audição rápida do catálogo (label -> hz)
        st.markdown("#### 🎛️ Audição rápida (Catálogo)")
        df_ok = df.dropna(subset=["hz"])
        if not df_ok.empty:
            labels = []
            label_to_hz = {}
            for i, row in enumerate(df_ok.itertuples(index=False), start=1):
                hz = float(getattr(row, "hz"))
                # getattr(row, "nome", None) não funciona em namedtuple: usamos lógica abaixo
                try:
                    nome = getattr(row, "nome")
                except Exception:
                    # fallback usando índice de colunas
                    nome = df_ok.columns[df_ok.columns.get_loc("nome")] if "nome" in df_ok.columns else None
                if not nome:
                    try:
                        code_val = getattr(row, "code")
                    except Exception:
                        code_val = ""
                    nome = code_val
                label = f"{int(hz)} Hz — {nome}"
                if label in label_to_hz:
                    label = f"{label} (#{i})"
                labels.append(label)
                label_to_hz[label] = hz

            sel_cat = st.selectbox(
                "Escolha uma frequência do catálogo",
                labels,
                key="play_sel_catalogo"
            )
            hz_cat = label_to_hz[sel_cat]

            from streamlit.components.v1 import html as st_html
            col1, col2 = st.columns([0.6, 0.4])
            with col1:
                st_html(webaudio_single_html(hz_cat, seconds=20), height=180)
            with col2:
                wav_bytes = synth_tone_wav(hz_cat, seconds=20.0, sr=22050, amp=0.2)
                st.download_button(
                    "Baixar WAV (20s)",
                    data=wav_bytes,
                    file_name=f"{int(hz_cat)}Hz_preview.wav",
                    mime="audio/wav",
                    key="dl_wav_cat"
                )

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

            # Normaliza 'tipo' para ENUM válido
            if "tipo" in df.columns and df["tipo"].dtype == object:
                df["tipo"] = (
                    df["tipo"].str.strip().str.lower()
                    .replace({"color": "cor"})
                )

            rows = df.to_dict(orient="records")
            ok, fail = 0, 0
            falhas = []
            for r in rows:
                try:
                    sb.table("frequencies").upsert(r, on_conflict="code").execute()
                    ok += 1
                except Exception as e:
                    fail += 1
                    falhas.append({"code": r.get("code"), "erro": str(e)})

            st.success(f"Importadas/atualizadas: {ok}. Falhas: {fail}.")
            if falhas:
                st.write("Falhas detalhadas:")
                st.dataframe(pd.DataFrame(falhas), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Erro ao importar seed: {e}")
    elif not sb:
        st.warning("Defina SUPABASE_URL e SUPABASE_KEY para habilitar a importação.")
