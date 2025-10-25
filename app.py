# app.py — MVP Clínico Holístico (Anamnese avançada + Binaural + Cama + Fitoterapia + Cristais + Biblioteca + Emoções)
# Secrets/ENV: SUPABASE_URL, SUPABASE_KEY (anon)

import os, io, json, wave, base64, time, pathlib
from datetime import datetime, date
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ----------------- ENV / SUPABASE -----------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
try:
    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
except Exception:
    sb = None

st.set_page_config(page_title="Programa Frequencias e Fito", layout="wide")

# Banner de build/arquivo para checagem de deploy
st.markdown(
    f"🛠️ **BUILD:** {time.strftime('%Y-%m-%d %H:%M:%S')} — "
    f"**arquivo:** `{pathlib.Path(__file__).resolve()}` — **cwd:** `{os.getcwd()}`"
)

# ----------------- KEYS ÚNICAS -----------------
def K(*parts: str) -> str:
    """Gera uma key única estável para widgets: K('aba','secao','campo')."""
    return "k_" + "_".join(str(p).strip().lower().replace(" ", "_") for p in parts if p)

# ----------------- HELPERS -----------------
@st.cache_data(ttl=60)
def sb_select(table, cols="*", order=None, desc=False, limit=None):
    """Consulta Supabase com fallback seguro para '*' e evita derrubar a UI."""
    if not sb:
        return []
    try:
        q = sb.table(table).select(cols)
        if order:
            try:
                q = q.order(order, desc=desc)
            except Exception:
                pass
        if limit:
            q = q.limit(limit)
        return q.execute().data or []
    except Exception as e:
        st.warning(f"[{table}] Falha na consulta com cols='{cols}'. Tentando '*'…")
        try:
            q = sb.table(table).select("*")
            if order:
                try:
                    q = q.order(order, desc=desc)
                except Exception:
                    pass
            if limit:
                q = q.limit(limit)
            return q.execute().data or []
        except Exception as e2:
            st.error(f"[{table}] Supabase erro: {e2}")
            return []

def synth_binaural_wav(fc: float, beat: float, seconds: float=20.0, sr: int=44100, amp: float=0.2) -> bytes:
    bt = abs(float(beat)); fl = max(1.0, float(fc)-bt/2); fr = float(fc)+bt/2
    t = np.linspace(0, seconds, int(sr*seconds), endpoint=False)
    left = np.sin(2*np.pi*fl*t); right = np.sin(2*np.pi*fr*t)
    ramp = int(sr*0.02)
    if ramp > 0:
        left[:ramp]*=np.linspace(0,1,ramp); right[:ramp]*=np.linspace(0,1,ramp)
        left[-ramp:]*=np.linspace(1,0,ramp); right[-ramp:]*=np.linspace(1,0,ramp)
    stereo = np.vstack([left,right]).T * float(amp)
    y = np.int16(np.clip(stereo,-1,1)*32767)
    buf = io.BytesIO()
    with wave.open(buf,"wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(y.tobytes())
    return buf.getvalue()

# Limite p/ data-URL (evita MessageSizeError)
MAX_BG_MB = 12  # ~12MB (vira ~16MB base64)

def bytes_to_data_url_safe(raw: bytes, filename: str|None, max_mb: int = MAX_BG_MB):
    """Converte bytes em data URL, mas recusa arquivos grandes para evitar MessageSizeError."""
    if not raw:
        return None, None, None
    size_mb = len(raw) / (1024*1024)
    name = (filename or "").lower()
    mime = "audio/mpeg"
    if name.endswith(".wav"): mime = "audio/wav"
    elif name.endswith(".ogg") or name.endswith(".oga"): mime = "audio/ogg"

    if size_mb > max_mb:
        return None, mime, f"Arquivo de {size_mb:.1f} MB excede o limite de {max_mb} MB para tocar embutido. Use arquivo menor ou Storage/URL."
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}", mime, None

def webaudio_binaural_html(fc: float, beat: float, seconds: int=60,
                           bg_data_url: str|None=None, bg_gain: float=0.12):
    """Player binaural + música de fundo usando <audio> (estável)."""
    bt = abs(float(beat))
    fl = max(20.0, float(fc) - bt/2)
    fr = float(fc) + bt/2
    sec = int(max(5, seconds))
    bg = json.dumps(bg_data_url) if bg_data_url else "null"
    g  = float(bg_gain)

    return f"""
<div style="padding:.6rem;border:1px solid #eee;border-radius:10px;">
  <b>Binaural</b> — L {fl:.2f} Hz • R {fr:.2f} Hz • {sec}s {'<span style="margin-left:6px;">🎵 fundo</span>' if bg_data_url else ''}<br/>
  <button id="bplay">▶️ Tocar</button> <button id="bstop">⏹️ Parar</button>
  <div style="font-size:.9rem;color:#666">Use fones · volume moderado</div>
</div>
<script>
let ctx=null, l=null, r=null, gL=null, gR=null, merger=null, timer=null;
let bgAudio=null, bgNode=null, bgGain=null;

function cleanup(){{
  try{{ if(l) l.stop(); if(r) r.stop(); }}catch(e){{}}
  [l,r,gL,gR,merger].forEach(n=>{{ if(n) try{{ n.disconnect(); }}catch(_e){{}} }});
  if(bgAudio){{ try{{ bgAudio.pause(); bgAudio.src=''; }}catch(_e){{}} bgAudio=null; }}
  if(bgNode)  {{ try{{ bgNode.disconnect(); }}catch(_e){{}} bgNode=null; }}
  if(bgGain)  {{ try{{ bgGain.disconnect(); }}catch(_e){{}} bgGain=null; }}
  if(ctx)     {{ try{{ ctx.close(); }}catch(_e){{}} ctx=null; }}
  if(timer) clearTimeout(timer);
}}

async function start(){{
  if(ctx) return;
  ctx = new (window.AudioContext || window.webkitAudioContext)();

  // --- Binaural (L/R) ---
  l = ctx.createOscillator(); r = ctx.createOscillator();
  l.type='sine'; r.type='sine';
  l.frequency.value={fl:.6f}; r.frequency.value={fr:.6f};
  gL = ctx.createGain(); gR = ctx.createGain();
  gL.gain.value = 0.05; gR.gain.value = 0.05;
  merger = ctx.createChannelMerger(2);
  l.connect(gL).connect(merger,0,0); r.connect(gR).connect(merger,0,1);
  merger.connect(ctx.destination);
  l.start(); r.start();

  // --- Música de fundo via <audio> ---
  const bg = {bg};
  if (bg) {{
    try {{
      bgAudio = new Audio(bg);
      bgAudio.loop = true;
      await bgAudio.play().catch(()=>{{ }});
      bgNode = ctx.createMediaElementSource(bgAudio);
      bgGain = ctx.createGain(); bgGain.gain.value = {g:.4f};

      // Forçar MONO
      const splitter = ctx.createChannelSplitter(2);
      const mergerMono = ctx.createChannelMerger(2);
      const gA = ctx.createGain(); gA.gain.value = 0.5;
      const gB = ctx.createGain(); gB.gain.value = 0.5;

      bgNode.connect(splitter);
      splitter.connect(gA, 0);
      splitter.connect(gB, 1);
      gA.connect(mergerMono, 0, 0);
      gB.connect(mergerMono, 0, 0);
      mergerMono.connect(bgGain).connect(ctx.destination);
      try {{ await bgAudio.play(); }} catch(e) {{ console.warn('Fundo não pôde iniciar:', e); }}
    }} catch(e) {{
      console.warn('Erro no fundo:', e);
    }}
  }}

  timer = setTimeout(()=>stop(), {sec*1000});
}}

function stop(){{
  cleanup();
}}

document.getElementById('bplay').onclick = start;
document.getElementById('bstop').onclick  = stop;
</script>
"""

def synth_tone_wav(hz: float, seconds: float = 10.0, sr: int = 44100, amp: float = 0.2) -> bytes:
    """Gera um WAV (estéreo) com tom puro em 'hz' por 'seconds'."""
    hz = max(1.0, float(hz))
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = np.sin(2 * np.pi * hz * t)
    ramp = int(sr * 0.02)  # fade 20ms
    if ramp > 0:
        y[:ramp] *= np.linspace(0, 1, ramp)
        y[-ramp:] *= np.linspace(1, 0, ramp)
    stereo = np.vstack([y, y]).T * float(amp)
    y16 = np.int16(np.clip(stereo, -1, 1) * 32767)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(y16.tobytes())
    return buf.getvalue()

def webaudio_tone_html(hz: float, seconds: int = 20, gain: float = 0.06, wave: str = "sine") -> str:
    """Player WebAudio de tom puro (um oscilador)."""
    hz = max(20.0, float(hz))
    sec = int(max(5, seconds))
    g = float(gain)
    wave = (wave or "sine")
    return f"""
<div style="padding:.6rem;border:1px solid #eee;border-radius:10px;">
  <b>Tom puro</b> — {hz:.2f} Hz • {sec}s
  <div style="margin:.3rem 0;">
    <button id="tone_play">▶️ Tocar</button>
    <button id="tone_stop">⏹️ Parar</button>
  </div>
  <div style="font-size:.9rem;color:#666">Use fones · volume moderado</div>
</div>
<script>
let ctx=null, osc=null, g=null, timer=null;
function cleanup(){{
  try{{ if(osc) osc.stop(); }}catch(e){{}}
  [osc,g].forEach(n=>{{ if(n) try{{ n.disconnect(); }}catch(_e){{}} }});
  if(ctx){{ try{{ ctx.close(); }}catch(_e){{}} ctx=null; }}
  if(timer) clearTimeout(timer);
}}
function start(){{
  if(ctx) return;
  ctx = new (window.AudioContext || window.webkitAudioContext)();
  osc = ctx.createOscillator(); osc.type = "{wave}"; osc.frequency.value = {hz:.6f};
  g = ctx.createGain(); g.gain.value = {g:.4f};
  osc.connect(g).connect(ctx.destination);
  osc.start();
  timer = setTimeout(()=>stop(), {sec*1000});
}}
function stop(){{ cleanup(); }}
document.getElementById('tone_play').onclick = start;
document.getElementById('tone_stop').onclick = stop;
</script>
"""

# ----------------- UI -----------------
st.title("🔗 DOCE CONEXÃO")

tabs = st.tabs([
    "Pacientes","Anamnese","Agenda","Sessão (Planner)","Frequências",
    "Binaural","Cama de Cristal","Fitoterapia","Cristais","Financeiro","Biblioteca","Emoções"
])

# ========== Pacientes ==========
with tabs[0]:
    st.subheader("Pacientes")
    if not sb:
        st.warning("Configure SUPABASE_URL/KEY.")

    with st.form(K("pacientes","form")):
        c1,c2,c3=st.columns([2,1,1])
        nome = c1.text_input("Nome", key=K("pacientes","form","nome"))
        nasc = c2.date_input("Nascimento", value=None, key=K("pacientes","form","nascimento"))
        tel  = c3.text_input("Telefone", key=K("pacientes","form","telefone"))
        email= st.text_input("E-mail", key=K("pacientes","form","email"))
        notas= st.text_area("Notas", key=K("pacientes","form","notas"))
        if st.form_submit_button("Salvar", use_container_width=True):
            if sb:
                sb.table("patients").insert({
                    "nome":nome or None,
                    "nascimento":str(nasc) if nasc else None,
                    "telefone":tel or None,
                    "email":email or None,
                    "notas":notas or None
                }).execute()
                st.success("Paciente salvo.")
                st.cache_data.clear()

    pts = sb_select("patients", "*", order="created_at", desc=True, limit=50)
    if pts:
        dfp = pd.DataFrame(pts)
        cols_show = [c for c in ["nome","nascimento","telefone","email","notas","created_at"] if c in dfp.columns]
        st.dataframe(dfp[cols_show] if cols_show else dfp, use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum paciente cadastrado ainda ou sem permissão de leitura (RLS).")

# ========== Anamnese (AVANÇADA) ==========
with tabs[1]:
    st.subheader("Anamnese — Avançada")

    pts = sb_select("patients","id,nome",order="created_at",desc=True)
    mapa = {p["nome"]:p["id"] for p in pts} if pts else {}
    psel = st.selectbox("Paciente", list(mapa.keys()) or ["—"], key=K("anv","paciente"))

    qs = sb_select("anamnesis_questions","section,qkey,label,qtype,min_val,max_val,options,weight,required,ord,active",order="section")
    if not qs:
        st.warning("Nenhuma pergunta cadastrada. Execute o SQL de perguntas (anamnesis_questions).")
    else:
        dfq = pd.DataFrame(qs)
        dfq = dfq[dfq["active"].fillna(True)]
        sections = list(dfq["section"].dropna().unique()) or ["Geral"]

        with st.form(K("anv","form")):
            tabs_sec = st.tabs(sections)
            respostas = {}
            for idx, sec in enumerate(sections):
                with tabs_sec[idx]:
                    bloco = dfq[dfq["section"]==sec].sort_values(["ord","qkey"])
                    for _, row in bloco.iterrows():
                        qk = row["qkey"]; label = row["label"]; qt = row["qtype"]
                        mn = row.get("min_val"); mx = row.get("max_val")
                        opts = row.get("options") or []
                        key = K("anv","q",sec,qk)

                        if qt == "likert":
                            base = int(((mn or 0) + (mx or 10))//2)
                            val = st.slider(label, int(mn or 0), int(mx or 10), base, key=key)
                        elif qt == "bool":
                            val = st.checkbox(label, key=key)
                        elif qt == "multi":
                            val = st.multiselect(label, opts, key=key)
                        elif qt == "number":
                            val = st.number_input(label, value=float(mn or 0), min_value=float(mn or 0), max_value=float(mx or 9999), step=1.0, key=key)
                        else:
                            val = st.text_input(label, key=key)
                        respostas[qk] = val

            def score_from_answers(ans: dict) -> dict:
                chakras = {k:0.0 for k in ["raiz","sacral","plexo","cardiaco","laringeo","terceiro_olho","coronal"]}
                flags = set()
                for _, row in dfq.iterrows():
                    qk = row["qkey"]; w = row.get("weight") or {}
                    if qk not in ans: continue
                    val = ans[qk]
                    if isinstance(w, dict) and "flags" in w:
                        if bool(val):
                            for fl in (w["flags"] or []):
                                flags.add(fl)
                    if isinstance(w, dict) and "chakra" in w:
                        for ch, wt in (w["chakra"] or {}).items():
                            try:
                                v = float(val) if isinstance(val,(int,float)) else (1.0 if val else 0.0)
                            except Exception:
                                v = 0.0
                            chakras[ch] = chakras.get(ch,0.0) + (float(wt) * v)

                for ch_key, eq_key in [
                    ("raiz","raiz_eq"),("sacral","sacral_eq"),("plexo","plexo_eq"),
                    ("cardiaco","cardiaco_eq"),("laringeo","laringeo_eq"),
                    ("terceiro_olho","terceiro_olho_eq"),("coronal","coronal_eq")
                ]:
                    if eq_key in respostas and isinstance(respostas[eq_key], (int,float)):
                        chakras[ch_key] += max(0.0, 10.0 - float(respostas[eq_key]))

                idx = {
                    "sono": float(respostas.get("sono_qualidade") or 0),
                    "estresse": float(respostas.get("estresse_nivel") or 0),
                    "ansiedade": float(respostas.get("ansiedade") or 0),
                }
                return {"chakras": chakras, "flags": sorted(list(flags)), "indices": idx}

            score = score_from_answers(respostas)

            st.markdown("### 🧭 Resumo clínico")
            CHAKRA_LABEL = {
                "raiz":"Raiz", "sacral":"Sacral", "plexo":"Plexo Solar", "cardiaco":"Cardíaco",
                "laringeo":"Laríngeo", "terceiro_olho":"Terceiro Olho", "coronal":"Coronal"
            }
            ch_items = sorted(score["chakras"].items(), key=lambda x: x[1], reverse=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Qualidade do sono", f"{int(score['indices'].get('sono',0))}/10")
            m2.metric("Nível de estresse", f"{int(score['indices'].get('estresse',0))}/10")
            m3.metric("Ansiedade (autoescala)", f"{int(score['indices'].get('ansiedade',0))}/10")

            if score["flags"]:
                st.markdown("**Contraindicações/sinais de cautela:**")
                st.write(" · ".join([f"`{f}`" for f in score["flags"]]))
            else:
                st.caption("Nenhuma contraindicação assinalada.")

            st.markdown("**Prioridade por chakra (maior = mais atenção):**")
            df_ch = pd.DataFrame(
                [{"Chakra": CHAKRA_LABEL.get(k,k).replace("_"," ").title(), "Atenção (score)": round(v,1)}
                 for k, v in ch_items]
            )
            st.dataframe(df_ch, use_container_width=True, hide_index=True)

            st.markdown("### 📝 Recomendações iniciais")
            if ch_items:
                top_ch, _val = ch_items[0]
                mapa_freq = {
                    "raiz":["SOL396","CHAKRA_RAIZ"],
                    "sacral":["SOL417","CHAKRA_SACRAL"],
                    "plexo":["SOL432","SOL417"],
                    "cardiaco":["SOL528","SOL639","CHAKRA_CARDIACO"],
                    "laringeo":["SOL741","CHAKRA_LARINGEO"],
                    "terceiro_olho":["SOL852","CHAKRA_TERCEIRO_OLHO"],
                    "coronal":["SOL963","CHAKRA_CORONAL"]
                }
                sugeridas = mapa_freq.get(top_ch, ["SOL528"])
                st.markdown(
                    f"- **Frequências de foco** (chakra principal: **{CHAKRA_LABEL.get(top_ch, top_ch).title()}**): "
                    + ", ".join([f"`{c}`" for c in sugeridas])
                )

            sono = float(score["indices"].get("sono",0))
            estresse = float(score["indices"].get("estresse",0))
            ans = float(score["indices"].get("ansiedade",0))
            if "epilepsia" in score["flags"]:
                st.markdown("- **Binaural:** evitar batidas altas; preferir **sem binaural** ou usar < 8 Hz com cautela.")
            else:
                if sono <= 5:
                    st.markdown("- **Binaural:** **Delta 2–3 Hz** (10–20 min) → **Theta 5–6 Hz** (10–15 min).")
                elif estresse >= 7 or ans >= 7:
                    st.markdown("- **Binaural:** **Theta 5–6 Hz** (15–20 min) e finalizar em **Alpha 10 Hz** (5–10 min).")
                else:
                    st.markdown("- **Binaural:** **Alpha 10 Hz** (10–15 min); opcional **Theta 6 Hz** curto (5–10 min).")

            st.markdown("- **Cama de Cristal:** sequência padrão **7×5 min**; dar **ênfase** no chakra principal (+2–3 min).")

            if "gravidez" in score["flags"]:
                st.markdown("- **Fitoterapia:** revisar contraindicações na gestação; usar apenas ervas **seguras**.")
            elif "sedativos" in score["flags"]:
                st.markdown("- **Fitoterapia:** atenção a **interações** com sedativos; preferir doses baixas.")
            else:
                st.markdown("- **Fitoterapia:** plano suave (ex.: **Camomila + Cidreira**, 2×/dia por 2–3 semanas).")

            agua = respostas.get("agua"); af = respostas.get("atividade_fisica")
            hab = []
            if isinstance(agua,(int,float)) and agua < 6: hab.append("Aumentar ingestão de água (≥ 6 copos/dia).")
            if isinstance(af,(int,float)) and af < 2:   hab.append("Mover o corpo ao menos 2–3×/semana.")
            try:
                if respostas.get("cafeina") and float(respostas.get("cafeina")) > 3:
                    hab.append("Reduzir cafeína após 16h.")
            except Exception:
                pass
            if hab:
                st.markdown("- **Hábitos:** " + " ".join(hab))

            st.caption("Observação: recomendações iniciais — ajustar conforme avaliação clínica e resposta do paciente.")

            if st.form_submit_button("Salvar anamnese", use_container_width=True) and sb:
                payload = {"patient_id": mapa.get(psel),
                           "respostas": respostas,
                           "score": score}
                sb.table("anamneses").insert(payload).execute()
                st.success("Anamnese salva com sucesso.")

# ========== Agenda ==========
with tabs[2]:
    st.subheader("Agenda")
    pts = sb_select("patients","id,nome",order="created_at",desc=True)
    mapa = {p["nome"]:p["id"] for p in pts} if pts else {}
    c1,c2,c3=st.columns(3)
    psel  = c1.selectbox("Paciente", list(mapa.keys()) or ["—"], key=K("agenda","paciente"))
    start = c2.date_input("Data", value=date.today(), key=K("agenda","data"))
    hora  = c3.time_input("Hora", key=K("agenda","hora"))
    tipo  = st.selectbox("Tipo", ["Cama","Binaural","Fitoterapia","Misto"], key=K("agenda","tipo"))
    notas = st.text_input("Notas", key=K("agenda","notas"))
    if st.button("Agendar", key=K("agenda","btn_agendar")) and sb:
        dt = datetime.combine(start, hora)
        sb.table("appointments").insert({
            "patient_id": mapa.get(psel),
            "inicio":dt.isoformat(),
            "tipo":tipo,
            "notas":notas or None
        }).execute()
        st.success("Agendado!")

    ag = sb_select("appointments","id,patient_id,inicio,tipo,notas,patients(nome)",order="inicio",desc=False,limit=100)
    if ag:
        df=pd.DataFrame(ag)
        try:
            df["inicio"] = pd.to_datetime(df["inicio"], utc=True).dt.tz_convert(None)
        except Exception:
            df["inicio"] = pd.to_datetime(df["inicio"], errors="coerce")
        df["Paciente"]=df.get("patients", [{}]).apply(lambda x: (x or {}).get("nome","—") if isinstance(x, dict) else "—")
        cols = [c for c in ["inicio","Paciente","tipo","notas"] if c in df.columns]
        st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum agendamento ou sem permissão de leitura (RLS).")

# ========== Sessão (Planner) ==========
with tabs[3]:
    st.subheader("Planner de Sessão")
    pts = sb_select("patients","id,nome",order="created_at",desc=True)
    mapa = {p["nome"]:p["id"] for p in pts} if pts else {}
    psel = st.selectbox("Paciente", list(mapa.keys()) or ["—"], key=K("planner","paciente"))

    st.markdown("**Escolha rapidamente componentes da sessão:**")
    colA,colB,colC,colD = st.columns(4)

    freqs = sb_select("frequencies","code,nome,hz,tipo,chakra,cor",order="code")
    opt_freq = [f'{f["code"]} • {f.get("nome","")}' for f in (freqs or [])]
    sel_freq = colA.multiselect("Frequências", opt_freq, key=K("planner","freqs"))

    pres = sb_select("binaural_presets","id,nome,carrier_hz,beat_hz,duracao_min",order="nome")
    mapa_pres = {p["nome"]:p for p in (pres or [])}
    sel_bina = colB.selectbox("Preset Binaural", list(mapa_pres.keys()) or ["(opcional)"], key=K("planner","binaural"))

    camas = sb_select("cama_presets","id,nome,etapas,duracao_min",order="nome")
    mapa_cama = {c["nome"]:c for c in (camas or [])}
    sel_cama = colC.selectbox("Preset Cama", list(mapa_cama.keys()) or ["(opcional)"], key=K("planner","cama"))

    plans = sb_select("phytotherapy_plans","id,name",order="name")
    mapa_plan = {p["name"]:p for p in (plans or [])}
    sel_plan = colD.selectbox("Plano Fitoterápico", list(mapa_plan.keys()) or ["(opcional)"], key=K("planner","fitoplan"))

    notas = st.text_area("Notas da sessão", key=K("planner","notas"))
    if st.button("Salvar sessão", key=K("planner","btn_salvar")) and sb:
        prot = {
            "frequencias":[{"code": s.split(" • ")[0]} for s in sel_freq],
            "binaural": mapa_pres.get(sel_bina),
            "cama": mapa_cama.get(sel_cama),
            "fitoterapia_plan": mapa_plan.get(sel_plan),
            "notas": notas
        }
        sb.table("sessions").insert({
            "patient_id": mapa.get(psel),
            "data": datetime.utcnow().isoformat(),
            "tipo":"Misto","protocolo":prot,"status":"rascunho"
        }).execute()
        st.success("Sessão salva!")

# ========== Frequências ==========
with tabs[4]:
    st.subheader("Catálogo de Frequências")
    df_f = pd.DataFrame(sb_select("frequencies","code,nome,hz,tipo,chakra,cor,descricao",order="code"))
    if not df_f.empty:
        st.dataframe(df_f, use_container_width=True, hide_index=True)

        st.markdown("### 🔊 Ouvir frequência selecionada")
        def _label_row(row):
            code = str(row.get("code") or "").strip()
            nome = str(row.get("nome") or "").strip()
            hz   = row.get("hz")
            hz_s = f"{float(hz):.2f} Hz" if pd.notnull(hz) else "—"
            base = code if code else "(sem code)"
            if nome:
                base += f" • {nome}"
            return f"{base} — {hz_s}"

        opts = df_f.apply(_label_row, axis=1).tolist()
        idx_map = {opts[i]: i for i in range(len(opts))}

        colL, colR = st.columns([2,1])
        sel_label = colL.selectbox("Selecione", opts, key=K("freq","player","sel"))
        row_sel = df_f.iloc[idx_map[sel_label]]
        hz_sel = float(row_sel.get("hz") or 0.0)

        colR.metric("Frequência (Hz)", f"{hz_sel:.2f}")
        chv = row_sel.get("chakra")
        colR.metric("Chakra", (chv.title() if isinstance(chv,str) and chv else "—"))

        modo = st.radio("Modo", ["Tom puro", "Binaural (diferença)"], horizontal=True, key=K("freq","player","modo"))
        dur  = st.slider("Duração (s)", 5, 120, 20, 5, key=K("freq","player","dur"))

        if modo == "Tom puro":
            st.components.v1.html(webaudio_tone_html(hz_sel, seconds=dur, gain=0.06, wave="sine"), height=160)
            wav_tone = synth_tone_wav(hz_sel, seconds=min(dur, 20), sr=44100, amp=0.2)
            st.audio(wav_tone, format="audio/wav")
            st.download_button("Baixar WAV (tom puro ~20s)", data=wav_tone,
                               file_name=f"tone_{hz_sel:.2f}Hz.wav", mime="audio/wav",
                               key=K("freq","player","dl_tone"))
        else:
            beat = st.slider("Batida (Hz)", 0.5, 45.0, 10.0, 0.5, key=K("freq","player","beat"))
            bt = abs(float(beat))
            fL = max(20.0, float(hz_sel) - bt/2.0)
            fR = float(hz_sel) + bt/2.0  # corrigido
            c1, c2 = st.columns(2)
            c1.metric("Esquerdo (L)", f"{fL:.2f} Hz")
            c2.metric("Direito (R)",  f"{fR:.2f} Hz")

            st.components.v1.html(webaudio_binaural_html(hz_sel, beat, seconds=dur, bg_data_url=None, bg_gain=0.12), height=300)
            wav_bin = synth_binaural_wav(hz_sel, beat, seconds=min(dur, 20), sr=44100, amp=0.2)
            st.audio(wav_bin, format="audio/wav")
            st.download_button("Baixar WAV (binaural ~20s)", data=wav_bin,
                               file_name=f"binaural_{hz_sel:.2f}Hz_{beat:.1f}Hz.wav", mime="audio/wav",
                               key=K("freq","player","dl_bin"))

        with st.expander("Adicionar/editar"):
            with st.form(K("freq","form")):
                code = st.text_input("code (único)", value="SOL528", key=K("freq","code"))
                nome = st.text_input("nome", value="Solfeggio 528 Hz", key=K("freq","nome"))
                hz   = st.number_input("hz", 1.0, 2000.0, 528.0, 1.0, key=K("freq","hz"))
                tipo = st.selectbox("tipo",["solfeggio","chakra","custom"],index=0, key=K("freq","tipo"))
                chakra = st.selectbox("chakra",["","raiz","sacral","plexo","cardiaco","laringeo","terceiro_olho","coronal"],index=0, key=K("freq","chakra"))
                cor  = st.text_input("cor", key=K("freq","cor"))
                desc = st.text_area("descrição", key=K("freq","desc"))
                if st.form_submit_button("Upsert", use_container_width=True) and sb:
                    sb.table("frequencies").upsert({
                        "code":code,"nome":nome,"hz":hz,"tipo":tipo,
                        "chakra":(chakra or None),"cor":(cor or None),"descricao":desc or None
                    }).execute()
                    st.success("Salvo."); st.cache_data.clear()
    else:
        st.info("Sem frequências na base ou sem permissão de leitura (RLS).")

# ========== Binaural ==========
with tabs[5]:
    st.subheader("Binaural — player rápido")

    band_map = {
        "Delta (1–4 Hz)": 3.0,
        "Theta (4–8 Hz)": 6.0,
        "Alpha (8–12 Hz)": 10.0,
        "Beta baixa (12–18 Hz)": 15.0,
        "Gamma (30–45 Hz)": 40.0,
    }
    bcol1, bcol2 = st.columns([2,1])
    faixa = bcol1.selectbox("Faixa de ondas (atalho)", list(band_map.keys()), index=2, key=K("binaural","faixa"))
    if bcol2.button("Aplicar faixa", key=K("binaural","faixa_apply")):
        st.session_state[K("binaural","beat")] = float(band_map[faixa])
        st.success(f"Batida ajustada para {band_map[faixa]} Hz")

    pres = sb_select("binaural_presets","id,nome,carrier_hz,beat_hz,duracao_min,notas",order="nome")
    mapa_pres = {p["nome"]:p for p in (pres or [])}
    cols_top = st.columns([2,1])
    preset_escolhido = cols_top[0].selectbox(
        "Tratamento pré-definido (binaural_presets)",
        list(mapa_pres.keys()) or ["(nenhum)"],
        key=K("binaural","preset_sel")
    )
    if cols_top[1].button("Aplicar preset", key=K("binaural","preset_apply")) and preset_escolhido in mapa_pres:
        p = mapa_pres[preset_escolhido]
        st.session_state[K("binaural","carrier")] = float(p.get("carrier_hz") or 220.0)
        st.session_state[K("binaural","beat")]    = float(p.get("beat_hz") or 10.0)
        st.session_state[K("binaural","dur")]     = int((p.get("duracao_min") or 10) * 60)
        st.success(f"Preset aplicado: {preset_escolhido}")

    c1,c2,c3=st.columns(3)
    carrier = c1.number_input("Carrier (Hz)",50.0,1000.0,
                              float(st.session_state.get(K("binaural","carrier"),220.0)),
                              1.0, key=K("binaural","carrier"))
    beat    = c2.number_input("Batida (Hz)",0.5,45.0,
                              float(st.session_state.get(K("binaural","beat"),10.0)),
                              0.5, key=K("binaural","beat"))
    dur     = int(c3.number_input("Duração (s)",10,3600,
                                  int(st.session_state.get(K("binaural","dur"),120)),
                                  5, key=K("binaural","dur")))

    bt = abs(float(beat))
    fL = max(20.0, float(carrier) - bt/2.0)
    fR = float(carrier) + bt/2.0
    mL, mR = st.columns(2)
    mL.metric("Esquerdo (L)", f"{fL:.2f} Hz")
    mR.metric("Direito (R)",  f"{fR:.2f} Hz")
    with st.expander("Como funciona?"):
        st.markdown(
            """
**Binaural** = duas frequências **puras** diferentes em cada ouvido → o cérebro percebe a **diferença** como um tom de batida (**beat**).  
**Cálculo:** `L = carrier − beat/2` e `R = carrier + beat/2`.  
Ex.: carrier 220 Hz e beat 10 Hz ⇒ L = **215 Hz**, R = **225 Hz** ⇒ o cérebro tende a sincronizar em **~10 Hz**.

**Faixas úteis (guia rápida):**
- **Delta** (1–4 Hz): sono profundo, reparo  
- **Theta** (4–8 Hz): imaginação, introspecção  
- **Alpha** (8–12 Hz): relaxamento atento, foco calmo  
- **Beta baixa** (12–18 Hz): atenção/alerta leve (use com cautela)
- **Gamma** (30–45 Hz): estimulação cognitiva breve (ex.: 40 Hz)
            """
        )

    st.markdown("🎵 Música de fundo (opcional)")
    bg_up   = st.file_uploader("MP3/WAV/OGG (até 12MB)",type=["mp3","wav","ogg"], key=K("binaural","bg_file"))
    bg_gain = st.slider("Volume do fundo",0.0,0.4,0.12,0.01, key=K("binaural","bg_gain"))

    raw = None; filename = None
    if bg_up:
        raw = bg_up.read(); filename = bg_up.name
        st.audio(raw)  # prévia

    bg_url, _mime, err = bytes_to_data_url_safe(raw, filename) if raw else (None, None, None)
    if err:
        st.warning(f"⚠️ {err}")

    st.components.v1.html(
        webaudio_binaural_html(carrier, beat, dur, bg_url, bg_gain),
        height=300
    )

    wav = synth_binaural_wav(carrier,beat,20,44100,0.2)
    st.audio(wav, format="audio/wav")
    st.download_button("Baixar WAV (20s)", data=wav,
                       file_name=f"binaural_{int(carrier)}_{beat:.1f}.wav",
                       mime="audio/wav", key=K("binaural","dl_wav"))

    with st.expander("Sugestões rápidas por objetivo"):
        st.markdown(
            """
- **Relaxar/ansiedade** → **Theta 5–6 Hz** (15–20 min) e fechar em **Alpha 10 Hz** (5–10 min).
- **Sono** → **Delta 2–3 Hz** (10–20 min) → **Theta 5–6 Hz** (10–15 min).
- **Foco calmo** → **Alpha 10 Hz** (10–15 min).
- **Gamma 40 Hz** → estimulação breve (5–12 min), volume baixo.  
> **Atenção:** epilepsia, marcapasso e outras condições pedem ajustes/evitar binaural.
            """
        )

# ========== Cama de Cristal ==========
with tabs[6]:
    st.subheader("Cama — presets de 7 luzes")

    camas = sb_select("cama_presets","id,nome,etapas,duracao_min,notas",order="nome")
    nomes = [c["nome"] for c in (camas or [])]
    sel = st.selectbox("Preset", nomes or ["—"], key=K("cama","sel"))

    CHAKRAS = ["raiz","sacral","plexo","cardiaco","laringeo","terceiro_olho","coronal"]
    CORES   = ["vermelho","laranja","amarelo","verde","azul","anil","violeta","branco"]

    def _df_from_preset(preset):
        try:
            etapas = preset.get("etapas") or []
            df = pd.DataFrame(etapas)
        except Exception:
            df = pd.DataFrame(columns=["ordem","chakra","cor","min"])
        for col in ["ordem","chakra","cor","min"]:
            if col not in df.columns: df[col] = None
        df["ordem"] = pd.to_numeric(df["ordem"], errors="coerce")
        df = df.sort_values("ordem", na_position="last").reset_index(drop=True)
        df["min"] = pd.to_numeric(df["min"], errors="coerce").fillna(5).astype(int)
        df["chakra"] = df["chakra"].fillna("").astype(str)
        df["cor"] = df["cor"].fillna("").astype(str)
        df["ordem"] = range(1, len(df)+1)
        return df[["ordem","chakra","cor","min"]]

    current_df = pd.DataFrame(columns=["ordem","chakra","cor","min"])
    preset_dict = None
    if nomes:
        preset_dict = [x for x in camas if x["nome"]==sel][0]
        current_df  = _df_from_preset(preset_dict)
        st.caption(f"Duração total: **{int(current_df['min'].sum())} min** — {preset_dict.get('notas','')}")

    st.markdown("### Editar preset (tabela)")
    col_a, col_b = st.columns([2,1])
    nome_edit = col_a.text_input("Nome do preset",
                                 value=(preset_dict["nome"] if preset_dict else "Chakras 7x5"),
                                 key=K("cama","nome"))
    notas_edit = col_b.text_input("Notas (opcional)",
                                  value=(preset_dict.get("notas","") if preset_dict else ""),
                                  key=K("cama","notas"))

    edited = st.data_editor(
        current_df if not current_df.empty else pd.DataFrame(
            [{"ordem":i+1,"chakra":CHAKRAS[i] if i<len(CHAKRAS) else "", "cor":CORES[i] if i<len(CORES) else "", "min":5}
             for i in range(7)]
        ),
        key=K("cama","editor"),
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "ordem": st.column_config.NumberColumn("Ordem", min_value=1, step=1, help="Sequência de aplicação"),
            "chakra": st.column_config.SelectboxColumn("Chakra", options=CHAKRAS, help="Selecione o chakra"),
            "cor": st.column_config.SelectboxColumn("Cor", options=CORES, help="Cor da luz"),
            "min": st.column_config.NumberColumn("Minutos", min_value=1, step=1, help="Duração desta etapa")
        }
    )

    if not edited.empty:
        edited = edited.copy()
        edited["ordem"] = pd.to_numeric(edited["ordem"], errors="coerce").fillna(0).astype(int)
        edited = edited.sort_values("ordem").reset_index(drop=True)
        edited["ordem"] = range(1, len(edited)+1)
        dur_total = int(edited["min"].sum())
        st.caption(f"🕒 Duração total prevista: **{dur_total} min**")

    c1, c2, _ = st.columns([1,1,2])
    salvar   = c1.button("💾 Salvar preset", key=K("cama","save"))
    duplicar = c2.button("🧬 Duplicar como…", key=K("cama","dup"))

    if (salvar or duplicar) and sb:
        target_name = nome_edit.strip()
        if duplicar and preset_dict and target_name == preset_dict["nome"]:
            st.error("Informe um **novo nome** para duplicar.")
        else:
            etapas_json = []
            for _, r in edited.iterrows():
                ch = (r.get("chakra") or "").strip()
                cr = (r.get("cor") or "").strip()
                mn = int(max(1, int(r.get("min") or 1)))
                if not ch:
                    continue
                etapas_json.append({
                    "ordem": int(r.get("ordem") or 1),
                    "chakra": ch,
                    "cor": cr if cr else None,
                    "min": mn
                })

            payload = {
                "nome": target_name,
                "etapas": etapas_json,
                "duracao_min": int(sum(e["min"] for e in etapas_json)),
                "notas": (notas_edit or None)
            }
            sb.table("cama_presets").upsert(payload).execute()
            st.success(("Duplicado como" if duplicar else "Salvo") + f" **{target_name}**.")
            st.cache_data.clear()

# ========== Fitoterapia ==========
with tabs[7]:
    st.subheader("Planos fitoterápicos")
    df = pd.DataFrame(sb_select("phytotherapy_plans","id,name,objetivo,posologia,duracao_sem,cadencia,notas",order="name"))
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Sem planos cadastrados ou sem permissão de leitura (RLS).")
    with st.expander("Novo plano"):
        with st.form(K("phyto","form")):
            name = st.text_input("Nome do plano", value="Calma Suave", key=K("phyto","nome"))
            obj  = st.text_area("Objetivo", key=K("phyto","objetivo"))
            pos  = st.text_area("Posologia (JSON)",
                                value='[{"erva":"Camomila","forma":"infusão","dose":"200 ml","frequencia":"2x/dia","duracao":"15 dias"}]',
                                key=K("phyto","posologia"))
            durw = st.number_input("Duração (semanas)",1,52,3, key=K("phyto","duracao"))
            cad  = st.text_input("Cadência", value="uso diário", key=K("phyto","cadencia"))
            notas = st.text_area("Notas", key=K("phyto","notas"))
            if st.form_submit_button("Salvar plano", use_container_width=True) and sb:
                sb.table("phytotherapy_plans").upsert({
                    "name":name,"objetivo":obj or None,"posologia":json.loads(pos),
                    "duracao_sem":int(durw),"cadencia":cad or None,"notas":notas or None
                }).execute()
                st.success("Plano salvo."); st.cache_data.clear()

# ========== Cristais ==========
with tabs[8]:
    st.subheader("Cristais")
    df = pd.DataFrame(sb_select("crystals","id,name,chakra,color,keywords,benefits,pairing_freq,notes",order="name"))
    if not df.empty:
        st.dataframe(df[["name","chakra","color","keywords"]] if set(["name","chakra","color","keywords"]).issubset(df.columns)
                     else df, use_container_width=True, hide_index=True)
        sel_opts = df["name"].dropna().tolist() if "name" in df.columns else []
        if sel_opts:
            sel = st.selectbox("Detalhes", sel_opts, key=K("cristais","sel"))
            row = df[df["name"]==sel].iloc[0]
            st.write("**Benefícios:**", row.get("benefits"))
            st.write("**Combinações de frequência:**", row.get("pairing_freq"))
            st.caption(row.get("notes") or "")
    else:
        st.info("Sem cristais cadastrados ou sem permissão de leitura (RLS).")

    with st.expander("Adicionar cristal"):
        with st.form(K("cristais","form")):
            name = st.text_input("Nome", value="Quartzo Rosa", key=K("cristais","nome"))
            chakra = st.multiselect("Chakras",
                                    ["raiz","sacral","plexo","cardiaco","laringeo","terceiro_olho","coronal"],
                                    key=K("cristais","chakras"))
            color = st.text_input("Cores (separe por vírgula)", value="rosa", key=K("cristais","cores"))
            kw    = st.text_input("Palavras-chave (vírgula)", value="acolhimento,autoamor", key=K("cristais","keywords"))
            bens  = st.text_area("Benefícios (um por linha)", value="Suaviza emoções\nApoia autocuidado", key=K("cristais","beneficios"))
            pair  = st.text_area("Pairing (JSON)", value='{"hz":[528,639],"solfeggio":["SOL528","SOL639"],"binaural":["alpha","theta"]}', key=K("cristais","pairing"))
            notas = st.text_area("Notas", key=K("cristais","notas"))
            if st.form_submit_button("Salvar cristal", use_container_width=True) and sb:
                sb.table("crystals").upsert({
                    "name":name,
                    "chakra":chakra,
                    "color":[c.strip() for c in (color or "").split(",") if c.strip()],
                    "keywords":[k.strip() for k in (kw or "").split(",") if k.strip()],
                    "benefits":[b.strip() for b in (bens or "").splitlines() if b.strip()],
                    "pairing_freq": json.loads(pair or "{}"),
                    "notes": notas or None
                }).execute()
                st.success("Cristal salvo."); st.cache_data.clear()

# ========== Financeiro ==========
with tabs[9]:
    st.subheader("Financeiro")
    prices = pd.DataFrame(sb_select("prices","id,item,valor_cents,ativo",order="item"))
    if not prices.empty:
        prices["valor"] = prices["valor_cents"]/100
        st.dataframe(prices[["item","valor","ativo"]] if set(["item","valor","ativo"]).issubset(prices.columns)
                     else prices, use_container_width=True, hide_index=True)
    else:
        st.info("Sem itens de preço cadastrados.")

    p_pac = sb_select("patients","id,nome",order="created_at",desc=True)
    mapa = {p["nome"]:p["id"] for p in (p_pac or [])}
    with st.form(K("financeiro","form")):
        c1,c2,c3 = st.columns(3)
        pac   = c1.selectbox("Paciente", list(mapa.keys()) or ["—"], key=K("fin","paciente"))
        items_list = prices["item"].tolist() if not prices.empty else ["Sessão"]
        item  = c2.selectbox("Item", items_list, key=K("fin","item"))
        valor_default = float(prices.loc[prices["item"]==item,"valor"].iloc[0]) if (not prices.empty and item in items_list) else 150.0
        valor = c3.number_input("Valor (R$)", 0.0, 9999.0, valor_default, 10.0, key=K("fin","valor"))
        metodo= st.selectbox("Método", ["PIX","Cartão","Dinheiro"], key=K("fin","metodo"))
        obs   = st.text_input("Obs", key=K("fin","obs"))
        if st.form_submit_button("Registrar pagamento", use_container_width=True) and sb:
            sb.table("payments").insert({
                "patient_id": mapa.get(pac),
                "item": item,
                "valor_cents": int(round(valor*100)),
                "metodo": metodo,
                "obs": obs or None
            }).execute()
            st.success("Pagamento lançado.")
    pays = pd.DataFrame(sb_select("payments","data,item,valor_cents,metodo,patients(nome)",order="data",desc=True,limit=50))
    if not pays.empty:
        pays["valor"] = pays["valor_cents"]/100
        pays["Paciente"]=pays.get("patients", [{}]).apply(lambda x:(x or {}).get("nome","—") if isinstance(x,dict) else "—")
        cols = [c for c in ["data","Paciente","item","valor","metodo"] if c in pays.columns]
        st.dataframe(pays[cols] if cols else pays, use_container_width=True, hide_index=True)
    else:
        st.info("Sem pagamentos ou sem permissão de leitura (RLS).")

# ========== Biblioteca ==========
with tabs[10]:
    st.subheader("Biblioteca de Tratamentos (Templates)")
    tpls = sb_select("therapy_templates","id,name,objetivo,roteiro_binaural,frequencias_suporte,cama_preset,phyto_plan,notas",order="name")
    if tpls:
        nomes=[t["name"] for t in tpls]; mapa={t["name"]:t for t in tpls}
        sel=st.selectbox("Template",nomes, key=K("biblioteca","template"))
        t=mapa[sel]
        st.markdown(f"**Objetivo:** {t.get('objetivo','')}")
        st.write("**Frequências de suporte:**", t.get("frequencias_suporte"))
        st.write("**Roteiro binaural:**", t.get("roteiro_binaural"))
        st.write("**Notas:**", t.get("notas"))
        st.caption("Aplique os itens nas abas específicas para tocar/editar.")
    else:
        st.info("Sem templates ou sem permissão de leitura (RLS).")

# ========== Emoções ==========
with tabs[11]:
    st.subheader("Emoções — mapa e recomendações")
    cols = "id,emocao,escala,polaridade,frequencias,binaural_beat_hz,carrier_hz,chakras,cores,cristais,afirmacoes,respiracao,notas"
    df = pd.DataFrame(sb_select("emotions_map", cols, order="escala"))
    if df.empty:
        st.info(
            "A tabela **emotions_map** está vazia. "
            "Você pode cadastrar abaixo ou importar um CSV com colunas: "
            "`emocao,escala,polaridade,frequencias(JSON),binaural_beat_hz,carrier_hz,chakras(JSON),cores(JSON),cristais(JSON),afirmacoes(JSON),respiracao,notas`."
        )
    else:
        # Filtros
        c1,c2,c3 = st.columns([2,1,1])
        termo = c1.text_input("Busca por emoção", key=K("emo","busca"))
        pols = sorted([p for p in df["polaridade"].dropna().unique().tolist()])
        pol  = c2.selectbox("Polaridade", ["(todas)"]+pols, index=0, key=K("emo","pol"))
        if "escala" in df.columns and df["escala"].notna().any():
            esc_min = int(df["escala"].min()); esc_max = int(df["escala"].max())
        else:
            esc_min, esc_max = 0, 100
        faixa = c3.slider("Faixa de escala", esc_min, esc_max, (esc_min, esc_max), key=K("emo","faixa"))

        dfv = df.copy()
        if termo:
            dfv = dfv[dfv["emocao"].fillna("").str.contains(termo, case=False, na=False)]
        if pol != "(todas)":
            dfv = dfv[dfv["polaridade"] == pol]
        if "escala" in dfv.columns:
            dfv = dfv[(dfv["escala"].fillna(0)>=faixa[0]) & (dfv["escala"].fillna(0)<=faixa[1])]

        cols_show = [c for c in ["emocao","escala","polaridade","frequencias","binaural_beat_hz","chakras","cristais"] if c in dfv.columns]
        st.dataframe(dfv[cols_show] if cols_show else dfv, use_container_width=True, hide_index=True)

        # Seleção
        nomes = dfv["emocao"].dropna().tolist()
        if nomes:
            sel = st.selectbox("Selecionar emoção", nomes, key=K("emo","sel"))
            row = dfv[dfv["emocao"]==sel].iloc[0]

            # Helper para normalizar lista JSON/CSV
            def as_list(val):
                if val is None: return []
                if isinstance(val, list): return val
                s = str(val).strip()
                if not s: return []
                try:
                    j = json.loads(s)
                    if isinstance(j, list): return j
                except Exception:
                    pass
                return [x.strip() for x in s.split(",") if x.strip()]

            freqs = as_list(row.get("frequencias"))
            chakras = as_list(row.get("chakras"))
            cores   = as_list(row.get("cores"))
            cristais= as_list(row.get("cristais"))
            afirms  = as_list(row.get("afirmacoes"))

            st.markdown("### Recomendações")
            cA,cB,cC = st.columns(3)
            cA.metric("Escala", f"{int(row.get('escala') or 0)}")
            cB.metric("Polaridade", row.get("polaridade") or "—")
            cC.metric("Binaural (beat)", f"{float(row.get('binaural_beat_hz') or 10.0):.1f} Hz")

            if freqs:
                st.markdown("**Frequências de suporte:** " + ", ".join([f"`{f}`" for f in freqs]))
            if chakras:
                st.markdown("**Chakras relacionados:** " + ", ".join(chakras))
            if cores:
                st.markdown("**Cores terapêuticas:** " + ", ".join(cores))
            if cristais:
                st.markdown("**Cristais:** " + ", ".join(cristais))
            if afirms:
                st.markdown("**Afirmações sugeridas:**")
                st.write("\n".join([f"- {a}" for a in afirms]))
            if row.get("respiracao"):
                st.markdown("**Respiração:** " + str(row.get("respiracao")))
            if row.get("notas"):
                st.caption(str(row.get("notas")))

            # Tocar imediatamente
            st.markdown("### Tocar agora")
            carrier = float(row.get("carrier_hz") or 220.0)
            beat    = float(row.get("binaural_beat_hz") or 10.0)
            dur     = st.slider("Duração (s)", 10, 600, 120, key=K("emo","dur"))
            st.components.v1.html(
                webaudio_binaural_html(carrier, beat, dur, bg_data_url=None, bg_gain=0.12),
                height=300
            )
            wav = synth_binaural_wav(carrier, beat, seconds=min(dur, 20), sr=44100, amp=0.2)
            st.audio(wav, format="audio/wav")
            st.download_button("Baixar WAV (binaural ~20s)", data=wav,
                               file_name=f"emo_{sel}_carrier{int(carrier)}_beat{beat:.1f}.wav",
                               mime="audio/wav", key=K("emo","dl"))

            # Se houver lista de Hz de suporte, tocar um tom puro opcional
            if freqs:
                st.markdown("### Tom puro (uma frequência de suporte)")
                hz_opt = st.selectbox("Escolha a frequência", [str(f) for f in freqs], key=K("emo","hzsel"))
                try:
                    hz_val = float(hz_opt)
                    st.components.v1.html(webaudio_tone_html(hz_val, seconds=30, gain=0.06, wave="sine"), height=160)
                except Exception:
                    st.caption("Frequência inválida para tom puro.")

    # Cadastrar / Editar emoção
    with st.expander("Adicionar/editar emoção"):
        with st.form(K("emo","form")):
            emocao = st.text_input("Emoção (única)", key=K("emo","nome"))
            escala = st.number_input("Escala (0-100)", 0, 100, 50, key=K("emo","escala"))
            polaridade = st.selectbox("Polaridade", ["negativa","neutra","positiva"], index=1, key=K("emo","pol_new"))
            frequencias = st.text_area("Frequências (JSON ou CSV)", value='[528,639]', key=K("emo","freqs"))
            binaural = st.number_input("Binaural beat (Hz)", 0.5, 45.0, 10.0, 0.5, key=K("emo","beat"))
            carrier  = st.number_input("Carrier (Hz)", 50.0, 1000.0, 220.0, 1.0, key=K("emo","carrier"))
            chakras  = st.text_input("Chakras (CSV)", value="cardiaco", key=K("emo","chakras"))
            cores    = st.text_input("Cores (CSV)", value="verde", key=K("emo","cores"))
            cristais = st.text_input("Cristais (CSV)", value="quartzo rosa", key=K("emo","cristais"))
            afirm    = st.text_area("Afirmações (uma por linha)", value="Eu me acolho\nEu estou segura/seguro", key=K("emo","afirm"))
            resp     = st.text_area("Respiração (texto)", value="4-7-8 por 3 ciclos", key=K("emo","resp"))
            notas    = st.text_area("Notas", key=K("emo","notas"))
            if st.form_submit_button("Salvar emoção", use_container_width=True) and sb:
                def to_json_list_csv(s):
                    s = (s or "").strip()
                    if not s: return []
                    try:
                        j = json.loads(s)
                        if isinstance(j, list): return j
                    except Exception:
                        pass
                    return [x.strip() for x in s.split(",") if x.strip()]

                payload = {
                    "emocao": emocao.strip() or None,
                    "escala": int(escala),
                    "polaridade": polaridade,
                    "frequencias": to_json_list_csv(frequencias),
                    "binaural_beat_hz": float(binaural),
                    "carrier_hz": float(carrier),
                    "chakras": to_json_list_csv(chakras),
                    "cores": to_json_list_csv(cores),
                    "cristais": to_json_list_csv(cristais),
                    "afirmacoes": [a.strip() for a in (afirm or "").splitlines() if a.strip()],
                    "respiracao": resp or None,
                    "notas": notas or None
                }
                sb.table("emotions_map").upsert(payload).execute()
                st.success("Emoção salva/atualizada.")
                st.cache_data.clear()
