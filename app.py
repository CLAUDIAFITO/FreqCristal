# app.py — Frequências • Cama de Cristal (com Binaural completo + Fitoterapia + Prescrições)
# Requisitos locais: pip install streamlit pandas python-dotenv supabase==2.4.6 numpy
# No deploy (Streamlit Cloud): defina SUPABASE_URL e SUPABASE_KEY (ANON) em Secrets/Vars.

import os, io, json, wave
from datetime import datetime, date
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ------------------------ ENV / SUPABASE ------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

try:
    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
except Exception:
    sb = None

# ------------------------ PAGE CONFIG / STYLE ------------------------
st.set_page_config(page_title="Frequências • Cama de Cristal", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.main .block-container { padding-top: 0.8rem; padding-bottom: 1rem; }
.modern-card { background: white; padding: 1.1rem; border-radius: 14px; 
               box-shadow: 0 6px 18px rgba(0,0,0,.07); border: 1px solid #eef; }
.small { font-size: 0.9rem; color:#666; }
.badge { display:inline-block; background:#f5f7ff; border:1px solid #e6ebff; padding:.2rem .5rem; border-radius:8px; margin-right:.4rem; }
.section-title { font-weight:600; margin-bottom:.25rem; }
.help { color:#666; font-size:.92rem; }
hr { border: 0; height: 1px; background: #eee; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

st.title("💫 Frequências — Cama de Cristal")

# ------------------------ DADOS BÁSICOS ------------------------
INTENCOES = {
    "Relaxamento": {"base": ["SOL174","SOL432","CHAKRA_CARDIACO"]},
    "Aterramento": {"base": ["SOL396","CHAKRA_RAIZ"]},
    "Clareza mental": {"base": ["SOL741","CHAKRA_TERCEIRO_OLHO"]},
    "Harmonização do coração": {"base": ["SOL528","SOL639","CHAKRA_CARDIACO"]},
    "Limpeza/Desbloqueio": {"base": ["SOL417","SOL741"]},
}
CHAKRA_MAP = {
    "Raiz":"raiz","Sacral":"sacral","Plexo":"plexo","Cardíaco":"cardiaco",
    "Laríngeo":"laringeo","Terceiro Olho":"terceiro_olho","Coronal":"coronal","Nenhum":None
}

# ------------------------ HELPERS ------------------------
@st.cache_data(ttl=60, show_spinner=False)
def carregar_catalogo_freq() -> pd.DataFrame:
    """Carrega catálogo da tabela 'frequencies' (Supabase), ou fallback local."""
    if sb:
        try:
            data = sb.table("frequencies").select("*").execute().data
            return pd.DataFrame(data or [])
        except Exception:
            pass
    # fallback local
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

def gerar_protocolo(intencao: str, chakra_alvo: Optional[str], duracao_min: int, catalogo: pd.DataFrame) -> pd.DataFrame:
    """Gera sequência com blocos equidistantes baseado na intenção + chakra alvo."""
    base = INTENCOES.get(intencao, {"base": []})
    sel = list(dict.fromkeys(base["base"]))  # sem repetição mantendo ordem
    if chakra_alvo:
        ccode = f"CHAKRA_{chakra_alvo.upper()}"
        if (catalogo["code"] == ccode).any() and ccode not in sel:
            sel.append(ccode)
    if not sel:  # fallback se nada bater
        sel = [c for c in ["SOL432","SOL528"] if (catalogo["code"] == c).any()]

    total = max(5, int(duracao_min)) * 60
    bloco = max(90, total // max(1, len(sel)))
    linhas = []
    for i, code in enumerate(sel, start=1):
        row = catalogo.loc[catalogo["code"] == code]
        if row.empty:
            continue
        r = row.iloc[0]
        linhas.append({
            "ordem": i,
            "code": code,
            "nome": r.get("nome"),
            "hz": float(r.get("hz")) if pd.notna(r.get("hz")) else None,
            "duracao_seg": int(bloco),
            "chakra": r.get("chakra"),
            "cor": r.get("cor")
        })
    if not linhas:
        return pd.DataFrame()
    usado = sum(l["duracao_seg"] for l in linhas)
    if usado < total:
        linhas[-1]["duracao_seg"] += total - usado
    return pd.DataFrame(linhas)

from streamlit.components.v1 import html as stc_html

def st_html(html: str, height: int = 220):
    stc_html(html, height=height, scrolling=False)

# ------------------------ ÁUDIO: SÍNTESE / PLAYERS ------------------------
def synth_tone_wav(freq: float, seconds: float = 20.0, sr: int = 22050, amp: float = 0.2) -> bytes:
    """Mono WAV (seno), com fade in/out."""
    t = np.linspace(0, seconds, int(sr*seconds), endpoint=False)
    wavef = np.sin(2*np.pi*float(freq)*t)
    ramp = max(1, int(sr * 0.01))
    env = np.ones_like(wavef); env[:ramp] = np.linspace(0,1,ramp); env[-ramp:] = np.linspace(1,0,ramp)
    y = (wavef * env * float(amp)).astype(np.float32)
    y_int16 = np.int16(np.clip(y, -1, 1) * 32767)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(y_int16.tobytes())
    return buf.getvalue()

def synth_binaural_wav(carrier_hz: float, beat_hz: float, seconds: float = 20.0, sr: int = 44100, amp: float = 0.2) -> bytes:
    """Estéreo WAV; L=(fc - beat/2), R=(fc + beat/2)."""
    bt = abs(float(beat_hz)); fc = float(carrier_hz)
    fl = max(1.0, fc - bt/2.0); fr = fc + bt/2.0
    t = np.linspace(0, seconds, int(sr*seconds), endpoint=False)
    left = np.sin(2*np.pi*fl*t); right = np.sin(2*np.pi*fr*t)
    ramp = max(1, int(sr * 0.01))
    env = np.ones_like(left); env[:ramp] = np.linspace(0,1,ramp); env[-ramp:] = np.linspace(1,0,ramp)
    left = left * env * float(amp); right = right * env * float(amp)
    stereo = np.vstack([left, right]).T
    y_int16 = np.int16(np.clip(stereo, -1, 1) * 32767)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(y_int16.tobytes())
    return buf.getvalue()

def webaudio_binaural_html(carrier_hz: float, beat_hz: float, seconds: int = 60) -> str:
    """HTML/JS com WebAudio para binaural simples (uma fase)."""
    try:
        fc = max(20.0, float(carrier_hz))
        bt = abs(float(beat_hz))
    except Exception:
        fc, bt = 220.0, 8.0
    fl = max(20.0, fc - bt / 2.0); fr = fc + bt / 2.0
    sec = max(5, int(seconds))
    return f"""
<div class='modern-card'>
  <div><b>WebAudio Binaural</b> • Portadora ~{fc:.1f} Hz • Batida {bt:.1f} Hz<br/>
  L={fl:.1f} Hz • R={fr:.1f} Hz • Duração {sec}s</div>
  <button id="start" style="margin-top:8px;">▶️ Tocar</button>
  <button id="stop" style="margin-left:6px;">⏹️ Parar</button>
  <div class='small'>Use fones. Volume moderado.</div>
</div>
<script>
let ctx, oscL, oscR, gainL, gainR, merger, stopTO;
function startAudio(){{
  if(ctx) return;
  ctx = new (window.AudioContext || window.webkitAudioContext)();
  oscL = ctx.createOscillator(); oscR = ctx.createOscillator();
  gainL = ctx.createGain(); gainR = ctx.createGain();
  merger = ctx.createChannelMerger(2);
  oscL.type = 'sine'; oscR.type='sine';
  oscL.frequency.value = {fl:.6f};
  oscR.frequency.value = {fr:.6f};
  gainL.gain.value = 0.05; gainR.gain.value = 0.05;
  oscL.connect(gainL); oscR.connect(gainR);
  gainL.connect(merger, 0, 0); gainR.connect(merger, 0, 1);
  merger.connect(ctx.destination);
  oscL.start(); oscR.start();
  stopTO = setTimeout(stopAudio, {sec*1000});
}}
function stopAudio(){{
  if(!ctx) return;
  try{{ oscL.stop(); oscR.stop(); }}catch(e){{}}
  try{{ ctx.close(); }}catch(e){{}}
  ctx = null; oscL = null; oscR = null;
  if(stopTO) clearTimeout(stopTO);
}}
document.getElementById('start').onclick = startAudio;
document.getElementById('stop').onclick = stopAudio;
</script>
"""

def webaudio_playlist_binaural_html(fases: list) -> str:
    """HTML/JS para tocar várias fases L/R em sequência (roteiro)."""
    data = json.dumps(fases)
    return f"""
<div class="modern-card">
  <div class="section-title">Roteiro binaural — Execução automática</div>
  <div class="help">Toca cada fase na ordem. Use <b>fones</b>. Edite a tabela acima para ajustar.</div>
  <div class="controls" style="margin-top:.5rem;">
    <button id="rb_play">▶️ Play Roteiro</button><button id="rb_stop">⏹️ Stop</button>
  </div>
  <div id="rb_status" style="margin-top:.35rem; font-size:.95rem;"></div>
  <div id="rb_now" class="help"></div>
</div>
<script>
const roteiro = {data}; let ctx=null,oscL=null,oscR=null,gainL=null,gainR=null,merger=null,timer=null,i=0,playing=false;
function stopAll(){{ playing=false; if(timer){{clearTimeout(timer);timer=null;}} [oscL,oscR].forEach(o=>{{if(o)try{{o.stop();}}catch(e){{}}}}); 
[oscL,oscR,gainL,gainR].forEach(n=>{{if(n)n.disconnect();}}); oscL=oscR=gainL=gainR=merger=null; 
document.getElementById("rb_status").textContent="Parado."; document.getElementById("rb_now").textContent=""; }}
function playStep(){{ 
  if(!playing) return; if(i>=roteiro.length){{stopAll();return;}} const f=roteiro[i]; 
  if(!ctx) ctx=new (window.AudioContext||window.webkitAudioContext)(); 
  oscL=ctx.createOscillator(); oscL.type="sine"; oscL.frequency.value=f.left_hz; 
  oscR=ctx.createOscillator(); oscR.type="sine"; oscR.frequency.value=f.right_hz; 
  gainL=ctx.createGain(); gainR=ctx.createGain(); 
  gainL.gain.setValueAtTime(0.0001,ctx.currentTime); gainR.gain.setValueAtTime(0.0001,ctx.currentTime); 
  gainL.gain.exponentialRampToValueAtTime(0.2,ctx.currentTime+0.05); gainR.gain.exponentialRampToValueAtTime(0.2,ctx.currentTime+0.05); 
  merger=ctx.createChannelMerger(2); 
  oscL.connect(gainL).connect(merger,0,0); oscR.connect(gainR).connect(merger,0,1); merger.connect(ctx.destination); 
  oscL.start(); oscR.start(); 
  document.getElementById("rb_status").textContent=`Fase ${'{'}i+1{'}'}/${'{'}roteiro.length{'}'}`; 
  document.getElementById("rb_now").innerHTML=`<span class="badge">L ${'{'}f.left_hz.toFixed(2){'}'} Hz</span> 
  <span class="badge">R ${'{'}f.right_hz.toFixed(2){'}'} Hz</span> — ${'{'}f.label{'}'}`; 
  timer=setTimeout(()=>{{ try{{oscL.stop();oscR.stop();}}catch(e){{}} 
  [oscL,oscR,gainL,gainR].forEach(n=>{{if(n)n.disconnect();}}); oscL=oscR=gainL=gainR=merger=null; i+=1; setTimeout(playStep,120); }}, Math.max(1,f.dur)*1000); }}
document.getElementById("rb_play").onclick=()=>{{ if(!roteiro.length) return; stopAll(); i=0; playing=true; playStep(); }};
document.getElementById("rb_stop").onclick=()=>stopAll();
</script>
"""

# ------------------------ TABS ------------------------
tabs = st.tabs([
    "Gerador","Binaural","Pacientes","Sessões","Catálogo","Templates","Fitoterapia","Prescrições","Admin"
])
tab_ger, tab_bin, tab_pac, tab_ses, tab_cat, tab_tpl, tab_phyto, tab_presc, tab_admin = tabs

# ======================== ABA: GERADOR ========================
with tab_ger:
    st.subheader("Gerador de protocolo terapêutico")
    catalogo = carregar_catalogo_freq()
    col1, col2, col3 = st.columns(3)
    intencao = col1.selectbox("Intenção", list(INTENCOES.keys()), key="ger_intencao")
    chakra_label = col2.selectbox("Chakra alvo", list(CHAKRA_MAP.keys()), index=list(CHAKRA_MAP.keys()).index("Nenhum"), key="ger_chakra")
    chakra_alvo = CHAKRA_MAP[chakra_label]
    duracao = int(col3.number_input("Duração (min)", min_value=10, max_value=120, value=30, step=5, key="ger_dur"))

    if st.button("Gerar protocolo", type="primary", key="ger_btn"):
        plano = gerar_protocolo(intencao, chakra_alvo, duracao, catalogo)
        if plano.empty:
            st.warning("Não foi possível gerar o protocolo com o catálogo atual.")
        else:
            st.dataframe(plano, use_container_width=True, hide_index=True)
            st.download_button("Baixar CSV", data=plano.to_csv(index=False).encode("utf-8"),
                               file_name="protocolo.csv", mime="text/csv", key="ger_csv")

# ======================== ABA: BINAURAL (COMPLETA) ========================
with tab_bin:
    st.subheader("Binaurais — tela guiada")

    # bandas/limites
    BANDS = {
        "Delta (1–4 Hz)": (1.0, 4.0),
        "Theta (4–8 Hz)": (4.0, 8.0),
        "Alpha (8–12 Hz)": (8.0, 12.0),
        "Beta (12–30 Hz)": (12.0, 30.0),
        "Gamma (30–40 Hz)": (30.0, 40.0),
    }

    c1, c2, c3 = st.columns(3)
    carrier = float(c1.number_input("Carrier (Hz)", 50.0, 1000.0, 220.0, step=1.0, key="bin_carrier"))
    banda = c2.selectbox("Faixa de batida", list(BANDS.keys()) + ["Personalizada"], key="bin_banda")
    if banda == "Personalizada":
        beat = float(c3.number_input("Batida (Hz)", 0.5, 40.0, 7.0, step=0.5, key="bin_beat_custom"))
    else:
        lo, hi = BANDS[banda]
        beat = float(c3.slider("Batida dentro da faixa", float(lo), float(hi), float((lo+hi)/2), 0.5, key="bin_beat_range"))

    d1, d2 = st.columns([0.5, 0.5])
    dur_binaural = int(d1.number_input("Duração (seg)", 10, 1200, 120, step=5, key="bin_dur"))
    amp = float(d2.slider("Volume relativo (WAV)", 0.05, 0.6, 0.2, 0.05, key="bin_amp"))

    left_hz = max(1.0, carrier - beat/2); right_hz = carrier + beat/2
    st.caption(f"L {left_hz:.2f} Hz | R {right_hz:.2f} Hz — Batida {beat:.2f} Hz")

    st.markdown("**Tocar (WebAudio, 1 fase)**")
    st_html(webaudio_binaural_html(carrier, beat, seconds=dur_binaural), height=240)

    st.markdown("**Prévia WAV (20s)**")
    wav_bin = synth_binaural_wav(carrier, beat, seconds=20.0, sr=44100, amp=amp)
    colL, colR = st.columns([0.7, 0.3])
    with colL:
        st.audio(wav_bin, format="audio/wav")
        st.caption("Use fones para efeito binaural.")
    with colR:
        st.download_button(
            "Baixar WAV (20s)",
            data=wav_bin,
            file_name=f"binaural_{int(carrier)}Hz_{beat:.2f}Hz_20s.wav",
            mime="audio/wav",
            key="dl_bin_wav"
        )

    st.divider()
    st.markdown("**Roteiro binaural (várias fases)**")
    default_rows = pd.DataFrame([
        {"fase":"Chegada/Relaxamento","carrier_hz":carrier,"beat_hz":10.0,"duracao_min":5},
        {"fase":"Aprofundamento","carrier_hz":carrier,"beat_hz":6.0,"duracao_min":15},
        {"fase":"Integração","carrier_hz":carrier,"beat_hz":10.0,"duracao_min":10},
    ])

    # editor com validação suave (compatível com versões mais novas do Streamlit)
    try:
        colcfg = None
        if hasattr(st, "column_config"):
            colcfg = {
                "fase": st.column_config.TextColumn("Fase"),
                "carrier_hz": st.column_config.NumberColumn("Carrier (Hz)", min_value=50.0, max_value=1000.0, step=1.0),
                "beat_hz": st.column_config.NumberColumn("Batida (Hz)", min_value=0.5, max_value=40.0, step=0.5),
                "duracao_min": st.column_config.NumberColumn("Duração (min)", min_value=1, max_value=120, step=1),
            }
        roteiro = st.data_editor(
            default_rows,
            key="roteiro_binaural",
            use_container_width=True,
            num_rows="dynamic",
            column_config=colcfg,
            help="Edite diretamente as células. Use o botão + para adicionar linhas."
        )
    except TypeError:
        roteiro = st.data_editor(default_rows, key="roteiro_binaural", use_container_width=True)

    if not roteiro.empty:
        roteiro = roteiro.copy()
        for col in ["carrier_hz","beat_hz","duracao_min"]:
            roteiro[col] = pd.to_numeric(roteiro[col], errors="coerce")
        roteiro["carrier_hz"] = roteiro["carrier_hz"].fillna(carrier)
        roteiro["beat_hz"] = roteiro["beat_hz"].fillna(7.0)
        roteiro["duracao_min"] = roteiro["duracao_min"].fillna(5).astype(int)
        roteiro["left_hz"] = (roteiro["carrier_hz"] - roteiro["beat_hz"]/2).clip(lower=1.0)
        roteiro["right_hz"] = roteiro["carrier_hz"] + roteiro["beat_hz"]/2
        roteiro["duracao_seg"] = (roteiro["duracao_min"]*60).astype(int)

        st.dataframe(
            roteiro[["fase","carrier_hz","beat_hz","left_hz","right_hz","duracao_min","duracao_seg"]],
            use_container_width=True, hide_index=True
        )

        fases = [
            {
                "label": f'{r["fase"]} — {r["beat_hz"]:.2f} Hz',
                "left_hz": float(r["left_hz"]),
                "right_hz": float(r["right_hz"]),
                "dur": int(r["duracao_seg"])
            }
            for _, r in roteiro.iterrows()
        ]

        st.markdown("**▶️ Tocar roteiro (WebAudio, estéreo)**")
        st_html(webaudio_playlist_binaural_html(fases), height=265)

        colx, coly, _ = st.columns(3)
        colx.download_button(
            "Baixar CSV", roteiro.to_csv(index=False).encode("utf-8"),
            "roteiro_binaural.csv", "text/csv", key="dl_rot_csv"
        )
        coly.download_button(
            "Baixar JSON", pd.DataFrame(fases).to_json(orient="records").encode("utf-8"),
            "roteiro_binaural.json", "application/json", key="dl_rot_json"
        )

        # opcional: salvar como sessão no banco
        if sb and st.button("Salvar como sessão (status=binaural)", type="primary", key="btn_save_rot"):
            payload = {
                "patient_id": None,
                "data": datetime.utcnow().isoformat(),
                "duracao_min": int(roteiro["duracao_min"].sum()),
                "intencao": "Roteiro Binaural",
                "chakra_alvo": None,
                "status": "binaural",
                "protocolo": {"fases": fases}
            }
            try:
                sb.table("sessions").insert([payload]).execute()
                st.success("Roteiro salvo!")
            except Exception as e:
                st.error(f"Erro ao salvar roteiro: {e}")

# ======================== ABA: PACIENTES ========================
with tab_pac:
    st.subheader("Pacientes")
    if not sb:
        st.info("Conecte seu Supabase (.env com SUPABASE_URL e SUPABASE_KEY) para ativar cadastros.")
    else:
        with st.form("pac_form"):
            c1, c2 = st.columns([2,1])
            nome = c1.text_input("Nome", key="pac_nome")
            nasc = c2.date_input("Nascimento", value=None, key="pac_nasc")
            notas = st.text_area("Notas", key="pac_notas")
            if st.form_submit_button("Salvar paciente", use_container_width=True):
                try:
                    payload = {"nome":nome.strip(), "nascimento": str(nasc) if nasc else None, "notas": notas.strip() or None}
                    sb.table("patients").insert([payload]).execute()
                    st.success("Paciente salvo!")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Erro ao salvar paciente: {e}")
        try:
            pats = sb.table("patients").select("*").order("created_at", desc=True).execute().data
        except Exception as e:
            pats = []
            st.error(f"Erro ao carregar pacientes: {e}")
        if pats:
            cols_show = [c for c in ["nome","nascimento","notas","created_at"] if c in (pats[0].keys() if pats else [])]
            st.dataframe(pd.DataFrame(pats)[cols_show], use_container_width=True, hide_index=True)

# ======================== ABA: SESSÕES ========================
with tab_ses:
    st.subheader("Sessões (protocolo salvo por paciente)")
    if not sb:
        st.info("Conecte o Supabase para salvar sessões.")
    else:
        try:
            pats = sb.table("patients").select("id,nome").order("created_at", desc=True).execute().data
        except Exception:
            pats = []
        mapa = {p["nome"]: p["id"] for p in (pats or [])}
        nome = st.selectbox("Paciente", list(mapa.keys()) if mapa else ["—"], key="ses_paciente")
        catalogo2 = carregar_catalogo_freq()
        colA, colB, colC = st.columns(3)
        intencao2 = colA.selectbox("Intenção", list(INTENCOES.keys()), key="ses_intencao")
        chakra_label2 = colB.selectbox("Chakra alvo", list(CHAKRA_MAP.keys()), index=list(CHAKRA_MAP.keys()).index("Nenhum"), key="ses_chakra")
        chakra_alvo2 = CHAKRA_MAP[chakra_label2]
        dur2 = int(colC.number_input("Duração (min)", 10, 120, 30, step=5, key="ses_dur"))
        if st.button("Gerar + salvar", type="primary", key="ses_btn"):
            try:
                plano = gerar_protocolo(intencao2, chakra_alvo2, dur2, catalogo2)
                payload = {
                    "patient_id": mapa.get(nome),
                    "data": datetime.utcnow().isoformat(),
                    "duracao_min": dur2,
                    "intencao": intencao2,
                    "chakra_alvo": chakra_alvo2,
                    "status": "rascunho",
                    "protocolo": json.loads(plano.to_json(orient="records")) if not plano.empty else []
                }
                s = sb.table("sessions").insert([payload]).execute().data
                st.success("Sessão criada!" if s is not None else "Falha ao criar sessão.")
            except Exception as e:
                st.error(f"Erro ao criar sessão: {e}")

        try:
            sess = sb.table("sessions").select("id,data,intencao,duracao_min,status,created_at").order("created_at", desc=True).execute().data
        except Exception:
            sess = []
        if sess:
            st.dataframe(pd.DataFrame(sess), use_container_width=True, hide_index=True)

# ======================== ABA: CATÁLOGO ========================
with tab_cat:
    st.subheader("Catálogo de Frequências")
    df = carregar_catalogo_freq()
    expected = ["code","nome","hz","tipo","chakra","cor"]
    has_cols = set(expected).issubset(df.columns)

    if df.empty or not has_cols:
        st.info("Catálogo vazio/inesperado. Vá à aba **Admin** e importe `seed_frequencies.csv`.")
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df[expected], use_container_width=True, hide_index=True)

# ======================== ABA: TEMPLATES ========================
with tab_tpl:
    st.subheader("Templates Terapêuticos")
    st.caption("Modelos salvos na tabela **therapy_templates** (educativo; não é aconselhamento médico).")
    if not sb:
        st.info("Conecte o Supabase para listar templates do banco.")
    else:
        try:
            cols = "id,name,objetivo,faixas_recomendadas,n_sessoes,cadencia,notas,roteiro_binaural,frequencias_suporte"
            tpls = sb.table("therapy_templates").select(cols).order("name", desc=False).execute().data
        except Exception as e:
            tpls = []
            st.error(f"Erro ao carregar templates: {e}")

        if tpls:
            nomes = [t["name"] for t in tpls]
            mapa_tpl = {t["name"]: t for t in tpls}
            colA, colB = st.columns([2,1])
            sel = colA.selectbox("Escolha um template", nomes, key="tpl_sel")
            dur_override = colB.number_input("Duração total (min) p/ aplicar no Gerador (opcional)", min_value=10, max_value=120, value=30, step=5, key="tpl_dur")

            if sel in mapa_tpl:
                t = mapa_tpl[sel]
                with st.expander("📋 Detalhes do template", expanded=True):
                    st.markdown(f"**Objetivo:** {t.get('objetivo','')}")
                    st.markdown(f"**Faixas recomendadas:** {', '.join(t.get('faixas_recomendadas') or [])}")
                    st.markdown(f"**Sessões:** {t.get('n_sessoes','?')} • **Cadência:** {t.get('cadencia','?')}")
                    st.markdown(f"**Notas:** {t.get('notas','—')}")
                    st.markdown("**Roteiro binaural:**")
                    try:
                        rb = t.get("roteiro_binaural") or []
                        st.dataframe(pd.DataFrame(rb), use_container_width=True, hide_index=True)
                    except Exception:
                        st.write(t.get("roteiro_binaural"))
                    st.markdown("**Frequências de suporte:**")
                    try:
                        fs = t.get("frequencias_suporte") or []
                        st.dataframe(pd.DataFrame(fs), use_container_width=True, hide_index=True)
                    except Exception:
                        st.write(t.get("frequencias_suporte"))
                st.caption("Para tocar binaural rápido, vá à aba **Binaural** e use a portadora/batida desejadas.")

# ======================== ABA: FITOTERAPIA (planos) ========================
with tab_phyto:
    st.subheader("Planos Fitoterápicos")
    st.caption("Tabela **phytotherapy_plans** (educativo).")
    if not sb:
        st.info("Conecte o Supabase para listar planos fitoterápicos.")
    else:
        try:
            cols = "id,name,objetivo,indicacoes,contraindicacoes,interacoes,posologia,duracao_sem,cadencia,notas"
            planos = sb.table("phytotherapy_plans").select(cols).order("name", desc=False).execute().data
        except Exception as e:
            planos = []
            st.error(f"Erro ao carregar planos: {e}")

        if planos:
            nomes = [p["name"] for p in planos]
            mapa = {p["name"]: p for p in planos}
            sel = st.selectbox("Plano", nomes, key="phyto_sel")
            if sel in mapa:
                p = mapa[sel]
                st.markdown(f"**Objetivo:** {p.get('objetivo','')}")
                col1, col2 = st.columns(2)
                col1.markdown(f"**Indicações:** {p.get('indicacoes','—')}")
                col2.markdown(f"**Contraindicações:** {p.get('contraindicacoes','—')}")
                st.markdown(f"**Interações:** {p.get('interacoes','—')}")
                st.markdown(f"**Duração sugerida:** {p.get('duracao_sem','—')} semanas • **Cadência:** {p.get('cadencia','—')}")
                st.markdown("**Posologia:**")
                try:
                    pos = p.get("posologia") or []
                    pos_df = pd.DataFrame(pos)
                    st.dataframe(pos_df, use_container_width=True, hide_index=True)
                except Exception:
                    st.write(p.get("posologia"))
                if p.get("notas"):
                    st.caption(p.get("notas"))

# ======================== ABA: PRESCRIÇÕES (fitoterapia) ========================
with tab_presc:
    st.subheader("🧾 Prescrições (Fitoterápicos)")
    if not sb:
        st.info("Conecte o Supabase (.env com SUPABASE_URL e SUPABASE_KEY) para usar prescrições.")
    else:
        @st.cache_data(ttl=60, show_spinner=False)
        def _carregar_pacientes():
            try:
                return sb.table("patients").select("id,nome").order("created_at", desc=True).execute().data or []
            except Exception:
                return []

        @st.cache_data(ttl=60, show_spinner=False)
        def _carregar_planos_fitoterapicos():
            try:
                cols = "id,name,objetivo,posologia,indicacoes,contraindicacoes,interacoes,duracao_sem,cadencia,notas"
                return sb.table("phytotherapy_plans").select(cols).order("name", desc=False).execute().data or []
            except Exception:
                return []

        def _posologia_md(posologia_json):
            try:
                items = posologia_json if isinstance(posologia_json, list) else json.loads(posologia_json or "[]")
            except Exception:
                items = []
            if not items:
                return "_(sem posologia cadastrada)_"
            linhas = []
            for it in items:
                erva = it.get("erva") or it.get("planta") or "Item"
                forma = it.get("forma", "")
                dose = it.get("dose", "")
                freq = it.get("frequencia", "")
                dur  = it.get("duracao", "")
                obs  = it.get("observacoes", "")
                linhas.append(f"- **{erva}** — {forma}; **dose:** {dose}; **freq.:** {freq}; **duração:** {dur}" + (f"; _obs.:_ {obs}" if obs else ""))
            return "\n".join(linhas)

        st.markdown("### Nova prescrição")
        pacientes = _carregar_pacientes()
        planos = _carregar_planos_fitoterapicos()

        mapa_pac = {p["nome"]: p["id"] for p in pacientes} if pacientes else {}
        nomes_pac = list(mapa_pac.keys()) if mapa_pac else ["—"]
        nomes_planos = [p["name"] for p in planos] if planos else ["—"]
        mapa_planos = {p["name"]: p for p in planos}

        with st.form("presc_form"):
            c1, c2 = st.columns([1,1])
            pac_nome = c1.selectbox("Paciente", nomes_pac, key="presc_paciente")
            plano_nome = c2.selectbox("Plano fitoterápico", nomes_planos, key="presc_plano")

            c3, c4, c5 = st.columns([1,1,1])
            data_ini = c3.date_input("Início", key="presc_ini")
            data_fim = c4.date_input("Término (opcional)", value=None, key="presc_fim")
            status = c5.selectbox("Status", ["ativa","concluída","suspensa","rascunho"], index=0, key="presc_status")

            notas = st.text_area("Notas adicionais", key="presc_notas", placeholder="Instruções personalizadas...")

            st.markdown("**Resumo do plano selecionado**")
            if planos and plano_nome in mapa_planos:
                plano_sel = mapa_planos[plano_nome]
                with st.expander(f"📋 {plano_sel['name']} — detalhes", expanded=False):
                    st.markdown(f"**Objetivo:** {plano_sel.get('objetivo','')}")
                    st.markdown(f"**Indicações:** {plano_sel.get('indicacoes','—')}")
                    st.markdown(f"**Contraindicações:** {plano_sel.get('contraindicacoes','—')}")
                    st.markdown(f"**Interações:** {plano_sel.get('interacoes','—')}")
                    st.markdown(f"**Duração sugerida:** {plano_sel.get('duracao_sem','—')} semanas • **Cadência:** {plano_sel.get('cadencia','—')}")
                    st.markdown("**Posologia sugerida:**")
                    st.markdown(_posologia_md(plano_sel.get("posologia")))

            bt = st.form_submit_button("💾 Salvar prescrição", use_container_width=True)

        if bt:
            try:
                payload = {
                    "patient_id": mapa_pac.get(pac_nome),
                    "plan_id": mapa_planos[plano_nome]["id"] if planos and plano_nome in mapa_planos else None,
                    "start_date": data_ini.isoformat() if data_ini else None,
                    "end_date": data_fim.isoformat() if data_fim else None,
                    "notes": (notas or "").strip() or None,
                    "status": status
                }
                sb.table("phytotherapy_prescriptions").insert(payload).execute()
                st.success("Prescrição salva com sucesso!")
                st.cache_data.clear()
            except Exception as e:
                st.error(f"Erro ao salvar prescrição: {e}")

        st.divider()
        st.markdown("### Prescrições recentes")

        # filtros
        fcol1, fcol2 = st.columns([1,1])
        filtro_pac = fcol1.selectbox("Filtrar por paciente", ["(todos)"] + nomes_pac, key="presc_filtro_pac")
        filtro_status = fcol2.selectbox("Status", ["(todos)","ativa","concluída","suspensa","rascunho"], key="presc_filtro_status")

        # consulta
        try:
            sel_cols = "id,created_at,start_date,end_date,status,notes,patient_id,plan_id,patients(nome),phytotherapy_plans(name)"
            q = sb.table("phytotherapy_prescriptions").select(sel_cols).order("created_at", desc=True)
            if filtro_pac != "(todos)" and filtro_pac in mapa_pac:
                q = q.eq("patient_id", mapa_pac[filtro_pac])
            if filtro_status != "(todos)":
                q = q.eq("status", filtro_status)
            data = q.execute().data
        except Exception as e:
            data = []
            st.error(f"Erro ao carregar prescrições: {e}")

        if data:
            df = pd.DataFrame(data)
            def _nome_p(x): 
                try: return (x.get("patients") or {}).get("nome","—")
                except Exception: return "—"
            def _nome_pl(x): 
                try: return (x.get("phytotherapy_plans") or {}).get("name","—")
                except Exception: return "—"
            df["Paciente"] = df.apply(_nome_p, axis=1)
            df["Plano"] = df.apply(_nome_pl, axis=1)
            df["Início"] = pd.to_datetime(df["start_date"]).dt.date
            df["Término"] = pd.to_datetime(df["end_date"]).dt.date
            df["Criado em"] = pd.to_datetime(df["created_at"]).dt.tz_convert(None)
            cols = ["Paciente","Plano","status","Início","Término","Criado em","notes"]
            cols_exist = [c for c in cols if c in df.columns]
            view = df[cols_exist].rename(columns={"notes":"Notas","status":"Status"})
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Exportar CSV", data=view.to_csv(index=False).encode("utf-8"),
                               file_name="prescricoes_fitoterapicas.csv", mime="text/csv", key="presc_dl")
        else:
            st.info("Nenhuma prescrição encontrada para os filtros atuais.")

# ======================== ABA: ADMIN ========================
with tab_admin:
    st.subheader("Admin")
    if not sb:
        st.info("Conecte o Supabase para as ações abaixo (.env com SUPABASE_URL e SUPABASE_KEY).")
    else:
        st.markdown("**Importar catálogo `seed_frequencies.csv` → tabela `frequencies`**")
        up = st.file_uploader("Selecione `seed_frequencies.csv`", type=["csv"], key="admin_csv")
        if up and st.button("Importar agora", key="admin_import"):
            try:
                df = pd.read_csv(up)

                # Strings vazias -> None
                for col in ["chakra","cor","descricao","code","nome","tipo"]:
                    if col in df.columns:
                        df[col] = df[col].replace({"": None})

                # tipos
                if "hz" in df.columns:
                    df["hz"] = pd.to_numeric(df["hz"], errors="coerce")
                if "tipo" in df.columns:
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
                st.cache_data.clear()
            except Exception as e:
                st.error(f"Falha ao importar: {e}")

        st.markdown("---")
        st.markdown("**Status do Supabase**")
        status_cols = st.columns(3)
        status_cols[0].metric("Cliente", "OK" if sb else "Sem conexão")
        status_cols[1].metric("URL", SUPABASE_URL or "—")
        status_cols[2].metric("Auth (ANON)", "OK" if SUPABASE_KEY else "—")

        st.caption("Se aparecer erro de RLS (row level security), crie políticas adequadas às tabelas (`patients`, `sessions`, `frequencies`, `therapy_templates`, `phytotherapy_plans`, `phytotherapy_prescriptions`).")

# ------------------------ FIM DO APP ------------------------
st.markdown("<div class='small'>© Seu App de Frequências — uso educativo. Não substitui cuidado médico.</div>", unsafe_allow_html=True)
