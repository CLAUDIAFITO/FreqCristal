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
.main .block-container { padding-top: 0.5rem; padding-bottom: 1rem; }
.modern-card { background: white; padding: 1rem 1.25rem; border-radius: 14px;
               box-shadow: 0 6px 18px rgba(0,0,0,.07); border: 1px solid #eef; margin-bottom: 0.75rem; }
.help { color:#556; font-size:.92rem; }
.kbd { display:inline-block; padding:0 .35rem; border-radius:4px; border:1px solid #ccd; background:#f8fafc; font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}
.badge { display:inline-block; padding:.3rem .55rem; border-radius:999px; background:#f2f6ff; border:1px solid #e5eaff; font-size:.86rem; margin-right:.35rem; }
.hz { font-weight: 600; } .section-title { margin: .25rem 0 .5rem 0; font-weight: 700; }
small.muted { color:#7a869a; }
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
    "Raiz":"raiz","Sacral":"sacral","Plexo":"plexo","Cardíaco":"cardiaco",
    "Laríngeo":"laringeo","Terceiro Olho":"terceiro_olho","Coronal":"coronal","Nenhum":None
}
BANDS = {
    "Delta (1–4 Hz)": (1.0, 4.0),"Theta (4–8 Hz)": (4.0, 8.0),"Alpha (8–12 Hz)": (8.0, 12.0),
    "Beta (12–30 Hz)": (12.0, 30.0),"Gamma (30–40 Hz)": (30.0, 40.0)
}

# =========================================================
# Sementes padrão (para primeiro uso) — educativas
# =========================================================
DEFAULT_TEMPLATES = [
    {
        "name": "Ansiedade leve / Estresse",
        "objetivo": "Reduzir ativação e promover relaxamento com integração.",
        "faixas_recomendadas": ["Alpha (8–12 Hz)", "Theta (4–8 Hz)"],
        "n_sessoes": 8, "cadencia": "1x/semana",
        "notas": "Respiração diafragmática; evitar cafeína 4–6h antes.",
        "roteiro_binaural": [
            {"fase":"Chegada/Aterramento","carrier_hz":220,"beat_hz":10.0,"duracao_min":5},
            {"fase":"Aprofundamento","carrier_hz":220,"beat_hz":6.0,"duracao_min":15},
            {"fase":"Integração","carrier_hz":220,"beat_hz":10.0,"duracao_min":10},
        ],
        "frequencias_suporte": [
            {"tipo":"frequencia","valor":"396 Hz (aterramento)","dur_min":5},
            {"tipo":"frequencia","valor":"528/639 Hz (integração coração)","dur_min":10},
        ],
    },
    {
        "name": "Insônia (higiene do sono)",
        "objetivo": "Facilitar transição vigília-sono e manter repouso.",
        "faixas_recomendadas": ["Theta (4–8 Hz)", "Delta (1–4 Hz)"],
        "n_sessoes": 6, "cadencia": "3x/semana",
        "notas": "Evitar telas/estímulos; ambiente escuro; volume baixo.",
        "roteiro_binaural": [
            {"fase":"Desaceleração","carrier_hz":200,"beat_hz":7.0,"duracao_min":10},
            {"fase":"Indução","carrier_hz":180,"beat_hz":3.0,"duracao_min":20},
        ],
        "frequencias_suporte": [{"tipo":"frequencia","valor":"174 Hz (acolhimento)","dur_min":5}],
    },
    {
        "name": "Foco / Produtividade calma",
        "objetivo": "Aumentar foco sustentado com baixo estresse.",
        "faixas_recomendadas": ["Alpha (8–12 Hz)"],
        "n_sessoes": 6, "cadencia": "2x/semana",
        "notas": "Ideal no início do expediente; pausas a cada 50 min.",
        "roteiro_binaural": [
            {"fase":"Preparação","carrier_hz":240,"beat_hz":10.0,"duracao_min":5},
            {"fase":"Foco calmo","carrier_hz":240,"beat_hz":10.0,"duracao_min":20},
            {"fase":"Descompressão","carrier_hz":220,"beat_hz":8.0,"duracao_min":5},
        ],
        "frequencias_suporte": [{"tipo":"frequencia","valor":"741 Hz (clareza)","dur_min":5}],
    },
    {
        "name": "Tensão muscular / Dor leve",
        "objetivo": "Reduzir tensão e sensibilização com relaxamento guiado.",
        "faixas_recomendadas": ["Theta (4–8 Hz)","Alpha (8–12 Hz)"],
        "n_sessoes": 8, "cadencia": "2x/semana",
        "notas": "Alongamentos suaves após a sessão; respeitar limites.",
        "roteiro_binaural": [
            {"fase":"Abrandamento","carrier_hz":200,"beat_hz":8.0,"duracao_min":10},
            {"fase":"Profundidade","carrier_hz":200,"beat_hz":6.0,"duracao_min":15},
        ],
        "frequencias_suporte": [
            {"tipo":"frequencia","valor":"174 Hz","dur_min":10},
            {"tipo":"frequencia","valor":"528 Hz (reparadora)","dur_min":10},
        ],
    },
    {
        "name": "Enxaqueca leve (entre crises)",
        "objetivo": "Desanuviar e prevenir tensão; não usar em crise aguda intensa.",
        "faixas_recomendadas": ["Alpha (8–12 Hz)"],
        "n_sessoes": 6, "cadencia": "1x/semana",
        "notas": "Evitar estímulos altos; interromper se desconforto.",
        "roteiro_binaural": [
            {"fase":"Descompressão","carrier_hz":210,"beat_hz":10.0,"duracao_min":10},
            {"fase":"Calmante","carrier_hz":200,"beat_hz":8.0,"duracao_min":15},
        ],
        "frequencias_suporte": [{"tipo":"frequencia","valor":"639 Hz (integração coração)","dur_min":10}],
    },
]

# =========================================================
# Banco — Catálogo e Templates (Supabase)
# =========================================================
def carregar_catalogo_freq() -> pd.DataFrame:
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

def load_templates_from_db() -> pd.DataFrame:
    """Lê therapy_templates; retorna DataFrame com colunas padronizadas."""
    if not sb:
        return pd.DataFrame([])
    try:
        data = sb.table("therapy_templates").select("*").order("created_at", desc=True).execute().data
        return pd.DataFrame(data or [])
    except Exception:
        return pd.DataFrame([])

def upsert_template(payload: dict) -> tuple[bool, str]:
    if not sb:
        return False, "Sem conexão com Supabase."
    try:
        sb.table("therapy_templates").upsert(payload, on_conflict="name").execute()
        return True, "Template salvo/atualizado."
    except Exception as e:
        return False, str(e)

def delete_template_by_name(name: str) -> tuple[bool, str]:
    if not sb:
        return False, "Sem conexão com Supabase."
    try:
        sb.table("therapy_templates").delete().eq("name", name).execute()
        return True, "Template removido."
    except Exception as e:
        return False, str(e)

def seed_default_templates_if_empty():
    if not sb:
        return False, "Sem conexão com Supabase."
    try:
        rows = sb.table("therapy_templates").select("id").limit(1).execute().data
        if rows:
            return False, "Já existem templates."
        for tpl in DEFAULT_TEMPLATES:
            sb.table("therapy_templates").insert([tpl]).execute()
        return True, "Templates padrão importados."
    except Exception as e:
        return False, str(e)

# =========================================================
# Protocolo a partir de intenção/chakra
# =========================================================
def gerar_protocolo(intencao: str, chakra_alvo: str|None, duracao_min: int, catalogo: pd.DataFrame) -> pd.DataFrame:
    base = INTENCOES.get(intencao, {"base": []})
    sel = list(dict.fromkeys(base["base"]))
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
        if row.empty: continue
        row = row.iloc[0]
        linhas.append({
            "ordem": i, "code": code, "nome": row.get("nome"), "hz": float(row.get("hz")),
            "duracao_seg": int(bloco), "chakra": row.get("chakra"), "cor": row.get("cor")
        })
    usado = sum(l["duracao_seg"] for l in linhas)
    if linhas and usado < total:
        linhas[-1]["duracao_seg"] += total - usado
    return pd.DataFrame(linhas)

# =========================================================
# Áudio — síntese WAV e players WebAudio
# =========================================================
def synth_tone_wav(freq: float, seconds: float = 20.0, sr: int = 22050, amp: float = 0.2) -> bytes:
    t = np.linspace(0, seconds, int(sr*seconds), endpoint=False)
    wavef = np.sin(2*np.pi*freq*t)
    ramp = max(1, int(sr * 0.01))
    env = np.ones_like(wavef); env[:ramp] = np.linspace(0,1,ramp); env[-ramp:] = np.linspace(1,0,ramp)
    y = (wavef * env * amp).astype(np.float32)
    y_int16 = np.int16(np.clip(y, -1, 1) * 32767)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(y_int16.tobytes())
    return buf.getvalue()

def synth_binaural_wav(carrier_hz: float, beat_hz: float, seconds: float = 20.0, sr: int = 44100, amp: float = 0.2) -> bytes:
    bt = abs(float(beat_hz)); fc = float(carrier_hz)
    fl = max(1.0, fc - bt/2.0); fr = fc + bt/2.0
    t = np.linspace(0, seconds, int(sr*seconds), endpoint=False)
    left = np.sin(2*np.pi*fl*t); right = np.sin(2*np.pi*fr*t)
    ramp = max(1, int(sr * 0.01))
    env = np.ones_like(left); env[:ramp] = np.linspace(0,1,ramp); env[-ramp:] = np.linspace(1,0,ramp)
    left = left * env * amp; right = right * env * amp
    stereo = np.vstack([left, right]).T
    y_int16 = np.int16(np.clip(stereo, -1, 1) * 32767)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(y_int16.tobytes())
    return buf.getvalue()

def webaudio_player_html(plano: pd.DataFrame) -> str:
    items = [{"hz": float(r["hz"]), "dur": int(r["duracao_seg"]), "label": f'{int(r["hz"])} Hz — {r["nome"] or r["code"]}'} for _, r in plano.iterrows()]
    playlist_json = json.dumps(items)
    return f"""
<div class="modern-card"><div class="section-title">Player (WebAudio)</div>
<div class="help">Toca a sequência do protocolo diretamente no navegador.</div>
<div class="controls" style="margin-top:.5rem;">
  <button id="btnPlay">▶️ Play</button><button id="btnPause">⏸️ Pause</button><button id="btnStop">⏹️ Stop</button>
</div><div id="status" style="margin-top:.5rem; font-size:.95rem;"></div><div id="now" style="margin-top:.25rem;"></div></div>
<script>
const playlist = {playlist_json}; let ctx=null,osc=null,gain=null,idx=0,playing=false,timer=null;
function up(){{const it=playlist[idx]||null;document.getElementById("status").textContent=playing&&it?`Tocando: ${'{'}it.label{'}'}`:"Pronto.";}}
function stopAll(){{playing=false;if(timer){{clearTimeout(timer);timer=null;}} if(osc){{try{{osc.stop();}}catch(e){{}} osc.disconnect();osc=null;}} if(gain){{gain.disconnect();gain=null;}} document.getElementById("now").textContent=""; up();}}
function step(){{
  if(!playing) return; if(idx>=playlist.length){{stopAll();return;}}
  const it=playlist[idx]; if(!ctx) ctx=new (window.AudioContext||window.webkitAudioContext)();
  osc=ctx.createOscillator(); gain=ctx.createGain(); osc.type="sine"; osc.frequency.value=it.hz;
  gain.gain.setValueAtTime(0.0001,ctx.currentTime); gain.gain.exponentialRampToValueAtTime(0.2,ctx.currentTime+0.05);
  osc.connect(gain).connect(ctx.destination); osc.start(); document.getElementById("now").textContent=it.label; up();
  timer=setTimeout(()=>{{ if(!playing) return; gain.gain.exponentialRampToValueAtTime(0.0001,ctx.currentTime+0.05); try{{osc.stop(ctx.currentTime+0.06);}}catch(e){{}} idx+=1; setTimeout(step,100); }}, Math.max(1,it.dur)*1000);
}}
document.getElementById("btnPlay").onclick=()=>{{ if(!playlist.length) return; if(!playing){{playing=true;if(idx>=playlist.length)idx=0;}} step(); }};
document.getElementById("btnPause").onclick=()=>{{ playing=false; if(timer){{clearTimeout(timer);timer=null;}} if(gain) gain.gain.setTargetAtTime(0.0001, (ctx?ctx.currentTime:0), 0.02); if(osc) try{{osc.stop((ctx?ctx.currentTime:0)+0.05);}}catch(e){{}} up(); }};
document.getElementById("btnStop").onclick=()=>{{ idx=0; stopAll(); }};
up();
</script>
"""

def webaudio_single_html(freq_hz: float, seconds: int = 20) -> str:
    return f"""
<div class="modern-card"><div class="section-title">Frequência única</div>
<div class="help">Toca apenas esta frequência por {int(seconds)}s.</div>
<div><span class="badge">Frequência <span class="hz">{int(freq_hz)} Hz</span></span></div>
<div class="controls" style="margin-top:.5rem;"><button id="s_play">▶️ Play</button><button id="s_stop">⏹️ Stop</button></div>
<div id="s_status" style="margin-top:.5rem; font-size:.95rem;"></div></div>
<script>
let ctx=null,osc=null,gain=null,timer=null; const f={float(freq_hz)}, d={int(seconds)};
function stopAll(){{ if(timer){{clearTimeout(timer);timer=null;}} if(osc){{try{{osc.stop();}}catch(e){{}} osc.disconnect();osc=null;}} if(gain){{gain.disconnect();gain=null;}} document.getElementById("s_status").textContent="Parado."; }}
document.getElementById("s_play").onclick=()=>{{ stopAll(); if(!ctx) ctx=new (window.AudioContext||window.webkitAudioContext)(); osc=ctx.createOscillator(); gain=ctx.createGain(); osc.type="sine"; osc.frequency.value=f; gain.gain.setValueAtTime(0.0001,ctx.currentTime); gain.gain.exponentialRampToValueAtTime(0.2,ctx.currentTime+0.05); osc.connect(gain).connect(ctx.destination); osc.start(); document.getElementById("s_status").textContent="Tocando "+f+" Hz ("+d+"s)"; timer=setTimeout(stopAll, d*1000); }};
document.getElementById("s_stop").onclick=stopAll;
</script>
"""

def webaudio_binaural_html(carrier_hz: float, beat_hz: float, seconds: int = 20) -> str:
    fc = float(carrier_hz)
    bt = abs(float(beat_hz))
    fl = max(1.0, fc - bt/2.0)  # calcula em Python
    fr = fc + bt/2.0            # calcula em Python

    return f"""
<div class="modern-card">
  <div class="section-title">Binaural (L/R)</div>
  <div class="help">Use <b>fones</b>. O efeito é a diferença entre os ouvidos. Batida = {bt:.2f} Hz.</div>
  <div style="margin:.25rem 0;">
    <span class="badge">Left <span class="hz">{fl:.2f} Hz</span></span>
    <span class="badge">Right <span class="hz">{fr:.2f} Hz</span></span>
    <span class="badge">Carrier <span class="hz">{int(fc)} Hz</span></span>
  </div>
  <div class="controls" style="margin-top:.5rem;">
    <button id="b_play">▶️ Play</button>
    <button id="b_stop">⏹️ Stop</button>
  </div>
  <div id="b_status" class="help"></div>
</div>
<script>
let ctx=null,oscL=null,oscR=null,gainL=null,gainR=null,merger=null,timer=null;
const sec={int(seconds)}, fL={fl}, fR={fr};
function stopAll(){{
  if(timer){{clearTimeout(timer);timer=null;}}
  [oscL,oscR].forEach(o=>{{if(o) try{{o.stop();}}catch(e){{}}}});
  [oscL,oscR,gainL,gainR].forEach(n=>{{if(n)n.disconnect();}});
  oscL=oscR=gainL=gainR=merger=null;
  document.getElementById("b_status").textContent="Parado.";
}}
document.getElementById("b_play").onclick=()=>{{
  stopAll(); if(!ctx) ctx=new (window.AudioContext||window.webkitAudioContext)();
  oscL=ctx.createOscillator(); oscL.type="sine"; oscL.frequency.value=fL;
  oscR=ctx.createOscillator(); oscR.type="sine"; oscR.frequency.value=fR;
  gainL=ctx.createGain(); gainR=ctx.createGain();
  gainL.gain.setValueAtTime(0.0001, ctx.currentTime);
  gainR.gain.setValueAtTime(0.0001, ctx.currentTime);
  gainL.gain.exponentialRampToValueAtTime(0.2, ctx.currentTime+0.05);
  gainR.gain.exponentialRampToValueAtTime(0.2, ctx.currentTime+0.05);
  merger=ctx.createChannelMerger(2);
  oscL.connect(gainL).connect(merger,0,0);
  oscR.connect(gainR).connect(merger,0,1);
  merger.connect(ctx.destination);
  oscL.start(); oscR.start();
  document.getElementById("b_status").textContent=`L ${'{'}fL.toFixed(2){'}'} Hz | R ${'{'}fR.toFixed(2){'}'} Hz`;
  timer=setTimeout(()=>stopAll(), sec*1000);
}};
document.getElementById("b_stop").onclick=()=>stopAll();
</script>
"""


def webaudio_playlist_binaural_html(fases: list) -> str:
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
function stopAll(){{ playing=false; if(timer){{clearTimeout(timer);timer=null;}} [oscL,oscR].forEach(o=>{{if(o)try{{o.stop();}}catch(e){{}}}}); [oscL,oscR,gainL,gainR].forEach(n=>{{if(n)n.disconnect();}}); oscL=oscR=gainL=gainR=merger=null; document.getElementById("rb_status").textContent="Parado."; document.getElementById("rb_now").textContent=""; }}
function playStep(){{ if(!playing) return; if(i>=roteiro.length){{stopAll();return;}} const f=roteiro[i]; if(!ctx) ctx=new (window.AudioContext||window.webkitAudioContext)(); oscL=ctx.createOscillator(); oscL.type="sine"; oscL.frequency.value=f.left_hz; oscR=ctx.createOscillator(); oscR.type="sine"; oscR.frequency.value=f.right_hz; gainL=ctx.createGain(); gainR=ctx.createGain(); gainL.gain.setValueAtTime(0.0001,ctx.currentTime); gainR.gain.setValueAtTime(0.0001,ctx.currentTime); gainL.gain.exponentialRampToValueAtTime(0.2,ctx.currentTime+0.05); gainR.gain.exponentialRampToValueAtTime(0.2,ctx.currentTime+0.05); merger=ctx.createChannelMerger(2); oscL.connect(gainL).connect(merger,0,0); oscR.connect(gainR).connect(merger,0,1); merger.connect(ctx.destination); oscL.start(); oscR.start(); document.getElementById("rb_status").textContent=`Fase ${'{'}i+1{'}'}/${'{'}roteiro.length{'}'}`; document.getElementById("rb_now").innerHTML=`<span class="badge">L ${'{'}f.left_hz.toFixed(2){'}'} Hz</span> <span class="badge">R ${'{'}f.right_hz.toFixed(2){'}'} Hz</span> — ${'{'}f.label{'}'}`; timer=setTimeout(()=>{{ try{{oscL.stop();oscR.stop();}}catch(e){{}} [oscL,oscR,gainL,gainR].forEach(n=>{{if(n)n.disconnect();}}); oscL=oscR=gainL=gainR=merger=null; i+=1; setTimeout(playStep,120); }}, Math.max(1,f.dur)*1000); }}
document.getElementById("rb_play").onclick=()=>{{ if(!roteiro.length) return; stopAll(); i=0; playing=true; playStep(); }};
document.getElementById("rb_stop").onclick=()=>stopAll();
</script>
"""

# =========================================================
# UI
# =========================================================
st.title("💫 Frequências — Cama de Cristal")
st.caption("Interface guiada. Material educativo; não substitui orientação médica.")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["Gerador", "Pacientes", "Sessões", "Catálogo", "Admin", "Binaurais", "Plano Terapêutico", "Templates"]
)

# ---------------- Gerador (frequências puras) ----------------
with tab1:
    st.subheader("Passo 1 — Escolher intenção e parâmetros")
    catalogo = carregar_catalogo_freq()
    col1, col2, col3 = st.columns(3)
    intencao = col1.selectbox("Intenção", list(INTENCOES.keys()), key="ger_intencao")
    chakra_label = col2.selectbox("Chakra alvo", list(CHAKRA_MAP.keys()),
                                  index=list(CHAKRA_MAP.keys()).index("Nenhum"), key="ger_chakra")
    chakra_alvo = CHAKRA_MAP[chakra_label]
    duracao = int(col3.number_input("Duração total (min)", 10, 120, 30, step=5, key="ger_dur"))
    if st.button("Gerar protocolo", type="primary", key="btn_gerar_protocolo"):
        if catalogo.empty or "code" not in catalogo.columns:
            st.error("Catálogo vazio ou inválido. Aba **Admin** → importar seed.")
        else:
            plano = gerar_protocolo(intencao, chakra_alvo, duracao, catalogo)
            if plano.empty:
                st.warning("Não foi possível montar a playlist com as frequências atuais.")
            else:
                st.dataframe(plano, use_container_width=True, hide_index=True)
                from streamlit.components.v1 import html as st_html
                st.markdown("**Tocar protocolo (WebAudio)**")
                st_html(webaudio_player_html(plano), height=260)
                # tocar uma etapa
                labels, l2h = [], {}
                for i, r in enumerate(plano.itertuples(index=False), start=1):
                    hz = float(getattr(r,"hz")); nome = getattr(r,"nome") or getattr(r,"code")
                    label = f"{int(hz)} Hz — {nome}"
                    if label in l2h: label += f" (#{i})"
                    labels.append(label); l2h[label] = hz
                sel = st.selectbox("Etapa para tocar 20s", labels, key="play_sel_protocolo")
                hz_sel = l2h[sel]
                colA, colB = st.columns([0.6, 0.4])
                with colA: st_html(webaudio_single_html(hz_sel, seconds=20), height=180)
                with colB:
                    st.download_button("Baixar WAV (20s)", synth_tone_wav(hz_sel, 20), f"{int(hz_sel)}Hz_preview.wav", "audio/wav", key="dl_wav_step")

# ---------------- Pacientes ----------------
with tab2:
    st.subheader("Pacientes")
    if not sb:
        st.info("Conecte SUPABASE_URL/KEY para habilitar.")
    else:
        with st.form("pac_form"):
            nome = st.text_input("Nome", key="pac_nome")
            nasc = st.date_input("Nascimento", value=None, key="pac_nasc")
            notas = st.text_area("Notas", key="pac_notas")
            if st.form_submit_button("Salvar"):
                if not nome.strip():
                    st.warning("Informe o nome.")
                else:
                    try:
                        sb.table("patients").insert([{"nome":nome.strip(),"nascimento":str(nasc) if nasc else None,"notas":notas}]).execute()
                        st.success("Paciente salvo!")
                    except Exception as e:
                        st.error(f"Erro ao salvar paciente: {e}")
        try:
            pats = sb.table("patients").select("*").order("created_at", desc=True).execute().data
        except Exception:
            pats = []
        st.dataframe(pd.DataFrame(pats)[["nome","nascimento","notas","created_at"]] if pats else pd.DataFrame(), use_container_width=True, hide_index=True)

# ---------------- Sessões ----------------
with tab3:
    st.subheader("Sessões (registro)")
    if not sb:
        st.info("Conecte o Supabase.")
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
                st.warning("Cadastre um paciente.")
            elif catalogo.empty or "code" not in catalogo.columns:
                st.error("Catálogo vazio/inválido. Aba **Admin**.")
            else:
                plano = gerar_protocolo(intencao2, chakra_alvo2, dur2, catalogo)
                payload = {"patient_id": mapa.get(nome),"data": datetime.utcnow().isoformat(),"duracao_min": dur2,
                           "intencao": intencao2,"chakra_alvo": chakra_alvo2,"status":"rascunho",
                           "protocolo": json.loads(plano.to_json(orient="records"))}
                try:
                    s = sb.table("sessions").insert([payload]).execute().data
                    st.success("Sessão criada!" if s else "Falha ao criar sessão.")
                except Exception as e:
                    st.error(f"Erro ao criar sessão: {e}")
        try:
            sess = sb.table("sessions").select("id,data,intencao,duracao_min,status").order("created_at", desc=True).execute().data
        except Exception:
            sess = []
        st.dataframe(pd.DataFrame(sess) if sess else pd.DataFrame(), use_container_width=True, hide_index=True)

# ---------------- Catálogo ----------------
with tab4:
    st.subheader("Catálogo (frequências)")
    df = carregar_catalogo_freq()
    expected = ["code","nome","hz","tipo","chakra","cor"]
    if df.empty or not set(expected).issubset(df.columns):
        st.info("Vá na aba **Admin** e importe `seed_frequencies.csv`.")
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df[expected], use_container_width=True, hide_index=True)
        df_ok = df.dropna(subset=["hz"])
        if not df_ok.empty:
            labels, l2h = [], {}
            for i, row in enumerate(df_ok.itertuples(index=False), start=1):
                hz = float(getattr(row,"hz")); nome = getattr(row,"nome") or getattr(row,"code")
                label = f"{int(hz)} Hz — {nome}"
                if label in l2h: label += f" (#{i})"
                labels.append(label); l2h[label] = hz
            sel_cat = st.selectbox("Ouvir 20s", labels, key="play_sel_catalogo")
            hz_cat = l2h[sel_cat]
            from streamlit.components.v1 import html as st_html
            col1, col2 = st.columns([0.6, 0.4])
            with col1: st_html(webaudio_single_html(hz_cat, seconds=20), height=180)
            with col2: st.download_button("Baixar WAV (20s)", synth_tone_wav(hz_cat, 20), f"{int(hz_cat)}Hz_preview.wav", "audio/wav", key="dl_wav_cat")

# ---------------- Admin ----------------
with tab5:
    st.subheader("Admin — Importar catálogo")
    st.code(f"URL? {bool(SUPABASE_URL)} | KEY? {bool(SUPABASE_KEY)} | Client? {bool(sb)}")
    up = st.file_uploader("seed_frequencies.csv", type=["csv"], key="admin_seed_upload")
    if up and sb and st.button("Importar agora", key="admin_btn_importar"):
        try:
            df = pd.read_csv(up)
            for col in ["chakra","cor","descricao","code","nome"]:
                if col in df.columns: df[col] = df[col].replace({"": None})
            if "hz" in df.columns: df["hz"] = pd.to_numeric(df["hz"], errors="coerce")
            if "tipo" in df.columns and df["tipo"].dtype == object:
                df["tipo"] = df["tipo"].str.strip().str.lower().replace({"color":"cor"})
            rows = df.to_dict(orient="records"); ok=fail=0; falhas=[]
            for r in rows:
                try:
                    sb.table("frequencies").upsert(r, on_conflict="code").execute(); ok+=1
                except Exception as e:
                    fail+=1; falhas.append({"code": r.get("code"), "erro": str(e)})
            st.success(f"Importadas/atualizadas: {ok}. Falhas: {fail}.")
            if falhas: st.dataframe(pd.DataFrame(falhas), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Erro ao importar seed: {e}")
    st.divider()
    st.subheader("Admin — Templates")
    if sb:
        c1, c2 = st.columns(2)
        if c1.button("Semear templates padrão", key="btn_seed_tpl"):
            ok, msg = seed_default_templates_if_empty()
            (st.success if ok else st.info)(msg)
        if c2.button("Recarregar lista", key="btn_reload_tpl"):
            st.experimental_rerun()
        df_tpl = load_templates_from_db()
        if not df_tpl.empty:
            st.dataframe(df_tpl[["name","n_sessoes","cadencia","objetivo","created_at"]], use_container_width=True, hide_index=True)
        else:
            st.caption("Nenhum template no banco ainda.")
    else:
        st.info("Conecte o Supabase para gerenciar templates.")

# ---------------- Binaurais ----------------
with tab6:
    st.subheader("Binaurais — tela guiada")
    c1, c2, c3 = st.columns(3)
    carrier = float(c1.number_input("Carrier (Hz)", 50.0, 1000.0, 220.0, step=1.0, key="bin_carrier"))
    banda = c2.selectbox("Faixa de batida", list(BANDS.keys()) + ["Personalizada"], key="bin_banda")
    if banda == "Personalizada":
        beat = float(c3.number_input("Batida (Hz)", 0.5, 40.0, 7.0, step=0.5, key="bin_beat_custom"))
    else:
        lo, hi = BANDS[banda]
        beat = float(c3.slider("Batida dentro da faixa", float(lo), float(hi), float((lo+hi)/2), 0.5, key="bin_beat_range"))
    d1, d2 = st.columns([0.5, 0.5])
    dur_binaural = int(d1.number_input("Duração (seg)", 10, 600, 30, step=5, key="bin_dur"))
    amp = float(d2.slider("Volume relativo", 0.05, 0.6, 0.2, 0.05, key="bin_amp"))
    left_hz = max(1.0, carrier - beat/2); right_hz = carrier + beat/2
    st.caption(f"L {left_hz:.2f} Hz | R {right_hz:.2f} Hz — Batida {beat:.2f} Hz")
    from streamlit.components.v1 import html as st_html
    st.markdown("**Tocar (WebAudio)**"); st_html(webaudio_binaural_html(carrier, beat, seconds=dur_binaural), height=230)
    st.markdown("**Prévia WAV (20s)**")
    wav_bin = synth_binaural_wav(carrier, beat, seconds=20.0, sr=44100, amp=amp)
    colL, colR = st.columns([0.7, 0.3])
    with colL: st.audio(wav_bin, format="audio/wav"); st.caption("Use fones para efeito binaural.")
    with colR: st.download_button("Baixar WAV (20s)", data=wav_bin, file_name=f"binaural_{int(carrier)}Hz_{beat:.2f}Hz_20s.wav", mime="audio/wav", key="dl_bin_wav")
    st.divider()
    st.markdown("**Roteiro (várias fases)**")
    default_rows = pd.DataFrame([
        {"fase":"Chegada/Relaxamento","carrier_hz":carrier,"beat_hz":10.0,"duracao_min":5},
        {"fase":"Aprofundamento","carrier_hz":carrier,"beat_hz":6.0,"duracao_min":15},
        {"fase":"Integração","carrier_hz":carrier,"beat_hz":10.0,"duracao_min":10},
    ])
    try:
        colcfg = None
        if hasattr(st, "column_config"):
            colcfg = {
                "fase": st.column_config.TextColumn("Fase"),
                "carrier_hz": st.column_config.NumberColumn("Carrier (Hz)", min_value=50.0, max_value=1000.0, step=1.0),
                "beat_hz": st.column_config.NumberColumn("Batida (Hz)", min_value=0.5, max_value=40.0, step=0.5),
                "duracao_min": st.column_config.NumberColumn("Duração (min)", min_value=1, max_value=120, step=1),
            }
        roteiro = st.data_editor(default_rows, key="roteiro_binaural", use_container_width=True,
                                 num_rows="dynamic", column_config=colcfg)
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
        st.dataframe(roteiro[["fase","carrier_hz","beat_hz","left_hz","right_hz","duracao_min","duracao_seg"]], use_container_width=True, hide_index=True)
        fases = [{"label": f'{r["fase"]} — {r["beat_hz"]:.2f} Hz', "left_hz": float(r["left_hz"]), "right_hz": float(r["right_hz"]), "dur": int(r["duracao_seg"])} for _, r in roteiro.iterrows()]
        st.markdown("**▶️ Tocar roteiro (WebAudio, estéreo)**"); st_html(webaudio_playlist_binaural_html(fases), height=260)
        colx, coly, _ = st.columns(3)
        colx.download_button("Baixar CSV", roteiro.to_csv(index=False).encode("utf-8"), "roteiro_binaural.csv", "text/csv", key="dl_rot_csv")
        coly.download_button("Baixar JSON", pd.DataFrame(fases).to_json(orient="records").encode("utf-8"), "roteiro_binaural.json", "application/json", key="dl_rot_json")
        if sb and st.button("Salvar como sessão (status=binaural)", type="primary", key="btn_save_rot"):
            payload = {"patient_id": None,"data": datetime.utcnow().isoformat(),"duracao_min": int(roteiro["duracao_min"].sum()),
                       "intencao":"Roteiro Binaural","chakra_alvo":None,"status":"binaural","protocolo":{"fases":fases}}
            try:
                sb.table("sessions").insert([payload]).execute(); st.success("Roteiro salvo!")
            except Exception as e:
                st.error(f"Erro ao salvar roteiro: {e}")

# ---------------- Plano Terapêutico (carrega do DB) ----------------
with tab7:
    st.subheader("Plano Terapêutico — usar sugestão do banco")
    if not sb:
        st.info("Conecte o Supabase para usar templates do banco. (Ou use a aba **Binaurais**/Gerador.)")
    else:
        df_tpl = load_templates_from_db()
        nomes_tpl = ["— Selecione —"] + (list(df_tpl["name"]) if not df_tpl.empty else [])
        sel_tpl = st.selectbox("Sugestão", nomes_tpl, index=0, key="tpl_select")
        if sel_tpl != "— Selecione —":
            tpl = df_tpl.loc[df_tpl["name"] == sel_tpl].iloc[0].to_dict()
            st.markdown(f"**Objetivo:** {tpl.get('objetivo','')}")
            st.caption(f"**Sessões:** {tpl.get('n_sessoes','?')} • **Cadência:** {tpl.get('cadencia','?')}")
            bands = tpl.get("faixas_recomendadas") or []
            st.caption("**Faixas:** " + ", ".join(bands) if bands else "—")
            notas = tpl.get("notas") or ""
            if notas: st.info(notas)
            # Roteiro binaural
            r = tpl.get("roteiro_binaural") or []
            df_rot = pd.DataFrame(r)
            if not df_rot.empty:
                df_rot["left_hz"] = (df_rot["carrier_hz"] - df_rot["beat_hz"]/2).clip(lower=1.0)
                df_rot["right_hz"] = df_rot["carrier_hz"] + df_rot["beat_hz"]/2
                df_rot["duracao_seg"] = (df_rot["duracao_min"]*60).astype(int)
                st.dataframe(df_rot[["fase","carrier_hz","beat_hz","left_hz","right_hz","duracao_min"]], use_container_width=True, hide_index=True)
                from streamlit.components.v1 import html as st_html
                fases = [{"label": f'{row["fase"]} — {row["beat_hz"]:.2f} Hz',"left_hz": float(row["left_hz"]),"right_hz": float(row["right_hz"]),"dur": int(row["duracao_seg"])} for _, row in df_rot.iterrows()]
                st.markdown("**▶️ Tocar roteiro (WebAudio)**"); st_html(webaudio_playlist_binaural_html(fases), height=260)
            # Frequências de suporte
            sup = tpl.get("frequencias_suporte") or []
            if sup:
                st.markdown("**Frequências de suporte**"); st.table(pd.DataFrame(sup))
            # Exportar / salvar
            plano_tpl = {
                "objetivo": tpl.get("objetivo"), "faixas_recomendadas": bands, "n_sessoes": tpl.get("n_sessoes"),
                "cadencia": tpl.get("cadencia"), "bloco_sessao": [{"fase": x["fase"], "tipo":"binaural", "valor": f'{x["beat_hz"]} Hz @ carrier {x["carrier_hz"]} Hz', "dur_min": x["duracao_min"]} for x in r],
                "notas": notas
            }
            colx, coly, _ = st.columns(3)
            colx.download_button("Baixar plano (JSON)", data=json.dumps(plano_tpl, ensure_ascii=False, indent=2).encode("utf-8"),
                                 file_name=f"plano_{sel_tpl.replace(' ','_')}.json", mime="application/json", key="tpl_dl_json")
            coly.download_button("Baixar roteiro (CSV)", data=(pd.DataFrame(r) if r else pd.DataFrame()).to_csv(index=False).encode("utf-8"),
                                 file_name=f"roteiro_{sel_tpl.replace(' ','_')}.csv", mime="text/csv", key="tpl_dl_csv")
            if st.button("Salvar sugestão como sessão", type="primary", key="tpl_save_session"):
                try:
                    dur_total = int(sum(int(x["duracao_min"]) for x in r)) if r else 0
                    payload = {"patient_id": None, "data": datetime.utcnow().isoformat(), "duracao_min": dur_total,
                               "intencao": sel_tpl, "chakra_alvo": None, "status": "plano",
                               "protocolo": {"plano": plano_tpl, "roteiro_binaural": (fases if r else [])}}
                    sb.table("sessions").insert([payload]).execute()
                    st.success("Sugestão salva como sessão!")
                except Exception as e:
                    st.error(f"Erro ao salvar sessão: {e}")

# ---------------- Templates (CRUD) ----------------
with tab8:
    st.subheader("Templates — gerenciar (Supabase)")
    if not sb:
        st.info("Conecte o Supabase para gerenciar templates.")
    else:
        df_tpl = load_templates_from_db()
        list_names = list(df_tpl["name"]) if not df_tpl.empty else []
        colA, colB = st.columns([0.6, 0.4])
        sel_edit = colA.selectbox("Editar template existente", ["—"] + list_names, index=0, key="tpl_edit_sel")
        if sel_edit != "—":
            rec = df_tpl.loc[df_tpl["name"] == sel_edit].iloc[0].to_dict()
        else:
            rec = {"name":"", "objetivo":"", "faixas_recomendadas":[], "n_sessoes":6, "cadencia":"1x/semana",
                   "notas":"", "roteiro_binaural":[{"fase":"Exemplo","carrier_hz":220,"beat_hz":10.0,"duracao_min":10}], "frequencias_suporte":[]}
        with st.form("tpl_form"):
            name = st.text_input("Nome do template", value=rec.get("name",""), key="tpl_name")
            objetivo = st.text_input("Objetivo", value=rec.get("objetivo",""), key="tpl_obj")
            bands = st.multiselect("Faixas recomendadas", list(BANDS.keys()), default=rec.get("faixas_recomendadas") or [], key="tpl_bands")
            col1, col2 = st.columns(2)
            n_ses = int(col1.number_input("Nº sessões", 1, 48, int(rec.get("n_sessoes") or 6), step=1, key="tpl_ns"))
            cad = col2.selectbox("Cadência", ["1x/semana","2x/semana","3x/semana","Diária"], index=0 if (rec.get("cadencia") not in ["2x/semana","3x/semana","Diária"]) else ["1x/semana","2x/semana","3x/semana","Diária"].index(rec.get("cadencia")), key="tpl_cad")
            notas = st.text_area("Notas", value=rec.get("notas",""), key="tpl_notes")

            st.markdown("**Roteiro binaural** (edite linhas)")
            df_rot = pd.DataFrame(rec.get("roteiro_binaural") or [{"fase":"Exemplo","carrier_hz":220,"beat_hz":10.0,"duracao_min":10}])
            try:
                colcfg = None
                if hasattr(st,"column_config"):
                    colcfg = {
                        "fase": st.column_config.TextColumn("Fase"),
                        "carrier_hz": st.column_config.NumberColumn("Carrier (Hz)", min_value=50.0, max_value=1000.0),
                        "beat_hz": st.column_config.NumberColumn("Batida (Hz)", min_value=0.5, max_value=40.0, step=0.5),
                        "duracao_min": st.column_config.NumberColumn("Duração (min)", min_value=1, max_value=120),
                    }
                df_rot_edit = st.data_editor(df_rot, key="tpl_rot_editor", use_container_width=True, num_rows="dynamic", column_config=colcfg)
            except TypeError:
                df_rot_edit = st.data_editor(df_rot, key="tpl_rot_editor", use_container_width=True)

            st.markdown("**Frequências de suporte** (puras, opcional)")
            df_sup = pd.DataFrame(rec.get("frequencias_suporte") or [])
            try:
                df_sup_edit = st.data_editor(
                    df_sup if not df_sup.empty else pd.DataFrame([{"tipo":"frequencia","valor":"396 Hz","dur_min":5}]),
                    key="tpl_sup_editor", use_container_width=True, num_rows="dynamic"
                )
            except TypeError:
                df_sup_edit = st.data_editor(df_sup, key="tpl_sup_editor", use_container_width=True)

            submit = st.form_submit_button("Salvar/Atualizar template", type="primary")
            if submit:
                # coerção básica
                rlist = df_rot_edit.fillna({"fase":"Fase","carrier_hz":220,"beat_hz":7.0,"duracao_min":5}).to_dict(orient="records")
                for r in rlist:
                    r["carrier_hz"] = float(r.get("carrier_hz") or 220)
                    r["beat_hz"] = float(r.get("beat_hz") or 7.0)
                    r["duracao_min"] = int(r.get("duracao_min") or 5)
                sup_list = df_sup_edit.to_dict(orient="records") if not df_sup_edit.empty else []
                payload = {
                    "name": name.strip(),
                    "objetivo": objetivo.strip(),
                    "faixas_recomendadas": bands,
                    "n_sessoes": n_ses,
                    "cadencia": cad,
                    "notas": notas,
                    "roteiro_binaural": rlist,
                    "frequencias_suporte": sup_list,
                }
                if not payload["name"]:
                    st.warning("Informe um nome para o template.")
                else:
                    ok, msg = upsert_template(payload)
                    (st.success if ok else st.error)(msg)

        colD, colE = st.columns(2)
        if sel_edit != "—" and colD.button("Excluir template selecionado", key="btn_del_tpl"):
            ok, msg = delete_template_by_name(sel_edit)
            (st.success if ok else st.error)(msg)
            if ok: st.experimental_rerun()
        if colE.button("Recarregar lista", key="btn_reload_tpl2"):
            st.experimental_rerun()
