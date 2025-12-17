# app.py — MVP Clínico Holístico (Anamnese avançada + Binaural + Cama + Fitoterapia + Cristais + Biblioteca + Emoções)
# Secrets/ENV: SUPABASE_URL, SUPABASE_KEY (anon)

import os, io, json, wave, base64, time, pathlib
from datetime import datetime, date
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import re

_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$")

def _is_uuid(x):
    s = str(x or "").strip()
    return bool(_UUID_RE.match(s))

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
    "Binaural","Cama de Cristal","Fitoterapia","Cristais",
    "Financeiro","Biblioteca","Emoções","Sessões salvas"
])


# ========== Pacientes ==========
with tabs[0]:
    st.subheader("Pacientes")
    if not sb:
        st.warning("Configure SUPABASE_URL/KEY para habilitar o CRUD de pacientes.")

    # -------- Cadastrar novo --------
    with st.form(K("pacientes","form_novo")):
        c1,c2,c3 = st.columns([2,1,1])
        nome  = c1.text_input("Nome", key=K("pacientes","novo","nome"))
        nasc  = c2.date_input("Nascimento", value=None, key=K("pacientes","novo","nascimento"))
        tel   = c3.text_input("Telefone", key=K("pacientes","novo","telefone"))
        email = st.text_input("E-mail", key=K("pacientes","novo","email"))
        notas = st.text_area("Notas", key=K("pacientes","novo","notas"))
        if st.form_submit_button("💾 Salvar novo paciente", use_container_width=True):
            if not sb:
                st.error("Supabase não configurado.")
            elif not (nome or "").strip():
                st.error("Informe o nome.")
            else:
                try:
                    sb.table("patients").insert({
                        "nome": (nome or "").strip(),
                        "nascimento": str(nasc) if nasc else None,
                        "telefone": tel or None,
                        "email": email or None,
                        # se a coluna 'notas' não existir, o banco ignora
                        "notas": (notas or None)
                    }).execute()
                    st.success("Paciente salvo.")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Erro ao salvar: {getattr(e, 'message', e)}")

    # -------- Lista (robusta às colunas que existirem) --------
    wanted_cols = ["id","nome","nascimento","telefone","email","notas","created_at"]
    pts = sb_select("patients", ",".join(wanted_cols), order="created_at", desc=True, limit=200) if sb else []
    df_pts = pd.DataFrame(pts)

    if df_pts.empty:
        st.info("Nenhum paciente cadastrado ainda.")
    else:
        show_cols = [c for c in ["nome","nascimento","telefone","email","notas","created_at"] if c in df_pts.columns]
        st.dataframe(df_pts[show_cols] if show_cols else df_pts, use_container_width=True, hide_index=True)

    # -------- Editar / Excluir --------
    st.markdown("### Gerenciar paciente")
    if df_pts.empty or "nome" not in df_pts.columns:
        st.caption("Cadastre alguém acima para editar/excluir. (A tabela precisa ter pelo menos 'nome'; ideal ter 'id').")
    else:
        # monta opções "Nome · #id" quando houver id; evita int() e lida com nulos
        opts = []
        for _, r in df_pts.iterrows():
            nm = str(r.get("nome") or "—")
            rid = r.get("id")
            label = f"{nm} · #{rid}" if pd.notnull(rid) else nm
            opts.append((label, rid, nm))
        labels = [o[0] for o in opts]
        sel_label = st.selectbox("Selecione o paciente", labels, key=K("pacientes","edit","sel"))

        # recupera id e nome selecionados
        sel_tuple = next((o for o in opts if o[0] == sel_label), None)
        sel_id = sel_tuple[1] if sel_tuple else None
        sel_nome = sel_tuple[2] if sel_tuple else None

        # escolhe a linha pela chave disponível (prioriza id)
        if sel_id is not None and "id" in df_pts.columns:
            row = df_pts[df_pts["id"] == sel_id].iloc[0]
        else:
            row = df_pts[df_pts["nome"] == sel_nome].iloc[0]

        def _parse_date_safe(v):
            try:
                if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip()):
                    return None
                return pd.to_datetime(v).date()
            except Exception:
                return None

        with st.form(K("pacientes","form_edit")):
            c1,c2,c3 = st.columns([2,1,1])
            e_nome  = c1.text_input("Nome", value=str(row.get("nome","")), key=K("pacientes","edit","nome"))
            e_nasc  = c2.date_input("Nascimento",
                                    value=_parse_date_safe(row.get("nascimento")),
                                    key=K("pacientes","edit","nasc"))
            e_tel   = c3.text_input("Telefone", value=str(row.get("telefone") or ""), key=K("pacientes","edit","tel"))
            e_email = st.text_input("E-mail", value=str(row.get("email") or ""), key=K("pacientes","edit","email"))
            if "notas" in df_pts.columns:
                e_notas = st.text_area("Notas", value=str(row.get("notas") or ""), key=K("pacientes","edit","notas"))
            else:
                e_notas = None
                st.caption("Coluna **notas** não existe na tabela — exibindo sem notas.")

            colU, colD = st.columns(2)
            upd = colU.form_submit_button("✏️ Atualizar", use_container_width=True)
            del_ok = colD.checkbox("Confirmar exclusão", key=K("pacientes","edit","confirm_del"))
            delete = colD.form_submit_button("🗑️ Excluir", use_container_width=True)

        # UPDATE/DELETE só se tivermos 'id'
        if upd and sb:
            payload = {
                "nome": (e_nome or "").strip(),
                "nascimento": str(e_nasc) if e_nasc else None,
                "telefone": e_tel or None,
                "email": e_email or None
            }
            if "notas" in df_pts.columns:
                payload["notas"] = (e_notas or None)

            try:
                if sel_id is None or "id" not in df_pts.columns:
                    st.warning("Não é possível atualizar sem coluna 'id' na tabela. Crie a coluna para habilitar UPDATE.")
                else:
                    sb.table("patients").update(payload).eq("id", sel_id).execute()
                    st.success("Paciente atualizado.")
                    st.cache_data.clear()
            except Exception as e:
                st.error(f"Erro ao atualizar: {getattr(e, 'message', e)}")

        if delete and sb:
            if not del_ok:
                st.warning("Marque 'Confirmar exclusão' para remover.")
            else:
                try:
                    if sel_id is None or "id" not in df_pts.columns:
                        st.warning("Não é possível excluir sem coluna 'id' na tabela.")
                    else:
                        sb.table("patients").delete().eq("id", sel_id).execute()
                        st.success("Paciente excluído.")
                        st.cache_data.clear()
                except Exception as e:
                    st.error(f"Erro ao excluir: {getattr(e, 'message', e)}")




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
    # ------------------------------------------
    # SESSÕES SALVAS (usar/aplicar/rodar)
    # ------------------------------------------
    st.markdown("### 💾 Sessões salvas do paciente")

    def _load_sessions_for_patient(pid: int):
        if not sb or not pid:
            return []
        try:
            # preferencial: filtra no backend
            q = sb.table("sessions").select(
                "id,patient_id,data,tipo,protocolo,status,patients(nome)"
            ).eq("patient_id", pid).order("data", desc=True).limit(50)
            return q.execute().data or []
        except Exception:
            # fallback: pega tudo e filtra aqui
            rows = sb_select("sessions", "id,patient_id,data,tipo,protocolo,status,patients(nome)",
                             order="data", desc=True, limit=200)
            return [r for r in (rows or []) if r.get("patient_id") == pid]

    # Mapeia labels das frequências para poder pré-selecionar no Planner
    code_to_label = {}
    try:
        for f in (freqs or []):
            lbl = f'{f["code"]} • {f.get("nome","")}'
            code_to_label[str(f.get("code"))] = lbl
    except Exception:
        pass

    pid_sel = mapa.get(psel)
    sess_rows = _load_sessions_for_patient(pid_sel) if pid_sel else []

    if not sess_rows:
        st.info("Nenhuma sessão para este paciente (ou sem permissão de leitura).")
    else:
        import pandas as _pd

        def _proto_summary(p: dict):
            if not isinstance(p, dict): return ("—", "—", "—", 0)
            # Frequências
            fqs = p.get("frequencias") or []
            nfreq = len(fqs)
            # Binaural
            bname = (p.get("binaural") or {}).get("nome") or "—"
            # Cama
            cname = (p.get("cama") or {}).get("nome") \
                    or ("preset" if isinstance(p.get("cama"), dict) else "—")
            # Fitoterapia
            pname = (p.get("fitoterapia_plan") or {}).get("name") or "—"
            return (bname, cname, pname, nfreq)

        table = []
        for r in sess_rows:
            b, c, p, nfreq = _proto_summary(r.get("protocolo") or {})
            data_str = r.get("data") or ""
            try:
                data_str = str(_pd.to_datetime(data_str))
            except Exception:
                pass
            table.append({
                "Quando": data_str,
                "Tipo": r.get("tipo") or "—",
                "Status": r.get("status") or "—",
                "Binaural": b,
                "Cama": c,
                "Plano Fito": p,
                "Frequências (#)": nfreq,
                "ID": r.get("id")
            })

        df_sess = _pd.DataFrame(table)
        st.dataframe(
            df_sess[["Quando","Tipo","Status","Binaural","Cama","Plano Fito","Frequências (#)"]],
            use_container_width=True, hide_index=True
        )

        st.markdown("#### Ações por sessão")
        for r in sess_rows:
            sid  = r.get("id")
            prot = r.get("protocolo") or {}
            with st.expander(f"🧭 Sessão #{sid} — {r.get('tipo','Misto')}  ·  {r.get('status','rascunho')}"):
                # --- EXECUTAR AGORA ---
                st.markdown("**▶️ Executar agora**")

                # Binaural (player embutido)
                b = prot.get("binaural") or {}
                try:
                    carrier = float(b.get("carrier_hz") or 220.0)
                    beat    = float(b.get("beat_hz") or 10.0)
                    dur_s   = int((b.get("duracao_min") or 10) * 60)
                    st.caption(f"Binaural: {b.get('nome','(preset)')} • carrier {carrier:.1f} Hz • beat {beat:.1f} Hz • {dur_s}s")
                    st.components.v1.html(webaudio_binaural_html(carrier, beat, seconds=min(dur_s, 300)), height=300)
                except Exception:
                    st.caption("Binaural: —")

                # Frequências (escolha rápida + player)
                fqs = prot.get("frequencias") or []
                codes = [ (fx.get("code") if isinstance(fx, dict) else str(fx)) for fx in fqs ]
                if codes:
                    opt = [ code_to_label.get(str(c), str(c)) for c in codes ]
                    sel_fx = st.selectbox("Tocar frequência do protocolo", opt, key=K("sess","fx",sid))
                    # extrai Hz se existir no catálogo carregado
                    try:
                        # acha no catálogo local df_f (aba Frequências). Se não existir, tenta pelos dados de suporte
                        hz_sel = None
                        code_sel = str(sel_fx).split(" • ")[0]
                        if 'df_f' in globals() and isinstance(df_f, _pd.DataFrame):
                            rowf = df_f[df_f["code"]==code_sel]
                            if not rowf.empty:
                                hz_sel = float(rowf.iloc[0].get("hz") or 0.0)
                        if hz_sel:
                            st.components.v1.html(webaudio_tone_html(hz_sel, seconds=20), height=160)
                        else:
                            st.caption("Frequência sem Hz no catálogo local — abra a aba **Frequências** para ouvir.")
                    except Exception:
                        pass
                else:
                    st.caption("Frequências: —")

                # Cama (grade)
                cama = prot.get("cama") or {}
                etapas = (cama.get("etapas") if isinstance(cama, dict) else []) or []
                if etapas:
                    _dfc = _pd.DataFrame(etapas, columns=["ordem","chakra","cor","min"]).copy()
                    _dfc["ordem"] = _pd.to_numeric(_dfc["ordem"], errors="coerce").fillna(0).astype(int)
                    _dfc = _dfc.sort_values("ordem").reset_index(drop=True)
                    _dfc["ordem"] = range(1, len(_dfc)+1)
                    st.caption(f"Cama: {cama.get('nome','(personalizada)')} • duração: {_dfc['min'].fillna(0).sum():.0f} min")
                    st.dataframe(_dfc, use_container_width=True, hide_index=True)
                else:
                    st.caption("Cama: —")

                # Plano fito (resumo)
                fito = prot.get("fitoterapia_plan") or {}
                if fito:
                    st.caption(f"Plano fito: {fito.get('name','—')}")
                notas = prot.get("notas") or ""
                if notas:
                    st.markdown(f"**Notas:** {notas}")

                st.download_button("⬇️ Exportar protocolo (JSON)",
                                   data=json.dumps(prot, ensure_ascii=False, indent=2),
                                   file_name=f"protocolo_sessao_{sid}.json",
                                   mime="application/json",
                                   key=K("sess","export",sid))

                st.markdown("---")

                # --- APLICAR NO PLANNER ---
                if st.button("📋 Aplicar no Planner (pré-preencher)", key=K("sess","apply",sid)):
                    try:
                        # Frequências -> multiselect do Planner
                        labels = [code_to_label.get(str(c.get("code") if isinstance(c, dict) else c),
                                                    str(c.get("code") if isinstance(c, dict) else c))
                                  for c in (prot.get("frequencias") or [])]
                        st.session_state[K("planner","freqs")] = [l for l in labels if l]

                        # Binaural preset / Cama preset / Plano fito (por nome)
                        bname = (prot.get("binaural") or {}).get("nome")
                        cname = (prot.get("cama") or {}).get("nome")
                        pname = (prot.get("fitoterapia_plan") or {}).get("name")

                        if bname and bname in mapa_pres: st.session_state[K("planner","binaural")] = bname
                        if cname and cname in mapa_cama: st.session_state[K("planner","cama")]     = cname
                        if pname and pname in mapa_plan: st.session_state[K("planner","fitoplan")] = pname

                        st.session_state[K("planner","notas")] = prot.get("notas","")

                        st.success("Protocolo aplicado ao Planner. (Abra/continue no topo desta aba.)")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Falha ao aplicar: {e}")

                colF, colD, colX = st.columns(3)
                # --- FINALIZAR ---
                if colF.button("✅ Finalizar sessão", key=K("sess","close",sid)) and sb:
                    try:
                        sb.table("sessions").update({"status":"finalizada"}).eq("id", sid).execute()
                        st.success("Sessão finalizada.")
                        st.cache_data.clear(); st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao finalizar: {getattr(e,'message',e)}")

                # --- DUPLICAR ---
                if colD.button("🧬 Duplicar como rascunho", key=K("sess","dup",sid)) and sb:
                    try:
                        sb.table("sessions").insert({
                            "patient_id": r.get("patient_id"),
                            "data": datetime.utcnow().isoformat(),
                            "tipo": r.get("tipo") or "Misto",
                            "protocolo": prot,
                            "status": "rascunho"
                        }).execute()
                        st.success("Sessão duplicada.")
                        st.cache_data.clear(); st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao duplicar: {getattr(e,'message',e)}")

                # --- EXCLUIR ---
                del_ok = colX.checkbox("Confirmar excluir", key=K("sess","confirmdel",sid))
                if colX.button("🗑️ Excluir sessão", key=K("sess","del",sid)) and sb:
                    if not del_ok:
                        st.warning("Marque 'Confirmar excluir' antes.")
                    else:
                        try:
                            sb.table("sessions").delete().eq("id", sid).execute()
                            st.success("Sessão excluída.")
                            st.cache_data.clear(); st.rerun()
                        except Exception as e:
                            st.error(f"Erro ao excluir: {getattr(e,'message',e)}")



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

    def _as_list(x):
        if isinstance(x, list):
            return x
        if x is None or (isinstance(x, str) and not x.strip()):
            return []
        return [x]

    # Carrega
    df = pd.DataFrame(sb_select(
        "crystals",
        "id,name,chakra,color,keywords,benefits,pairing_freq,notes",
        order="name"
    ))

    if df.empty:
        st.info("Sem cristais cadastrados ou sem permissão de leitura (RLS).")
    else:
        # visão geral
        base_cols = [c for c in ["name","chakra","color","keywords"] if c in df.columns]
        st.dataframe(df[base_cols] if base_cols else df, use_container_width=True, hide_index=True)

        # seleção
        nomes = df["name"].dropna().tolist() if "name" in df.columns else []
        if nomes:
            sel = st.selectbox("Editar cristal", nomes, key=K("cristais","sel"))
            row = df[df["name"]==sel].iloc[0]
            crystal_id = row.get("id")

            st.markdown("### Detalhes em grid")

            # Grids de listas simples
            CHAKRAS = ["raiz","sacral","plexo","cardiaco","laringeo","terceiro_olho","coronal"]

            grid_chak = pd.DataFrame({"chakra": _as_list(row.get("chakra"))})
            grid_col  = pd.DataFrame({"cor":    _as_list(row.get("color"))})
            grid_kw   = pd.DataFrame({"palavra":_as_list(row.get("keywords"))})
            grid_ben  = pd.DataFrame({"beneficio": _as_list(row.get("benefits"))})

            cA, cB = st.columns(2)
            with cA:
                st.caption("Chakras")
                grid_chak = st.data_editor(
                    grid_chak, num_rows="dynamic", use_container_width=True, hide_index=True,
                    column_config={"chakra": st.column_config.SelectboxColumn("Chakra", options=CHAKRAS)}
                )
                st.caption("Cores")
                grid_col = st.data_editor(
                    grid_col, num_rows="dynamic", use_container_width=True, hide_index=True
                )
            with cB:
                st.caption("Palavras-chave")
                grid_kw = st.data_editor(
                    grid_kw, num_rows="dynamic", use_container_width=True, hide_index=True
                )
                st.caption("Benefícios")
                grid_ben = st.data_editor(
                    grid_ben, num_rows="dynamic", use_container_width=True, hide_index=True
                )

            # Grid para pairing_freq (separa por tipos)
            p = row.get("pairing_freq") or {}
            grid_pair_hz  = st.data_editor(
                pd.DataFrame({"Hz": _as_list(p.get("hz"))}),
                num_rows="dynamic", use_container_width=True, hide_index=True,
                column_config={"Hz": st.column_config.NumberColumn("Hz", step=1.0)}
            )
            grid_pair_sol = st.data_editor(
                pd.DataFrame({"Solfeggio": _as_list(p.get("solfeggio"))}),
                num_rows="dynamic", use_container_width=True, hide_index=True
            )
            grid_pair_bin = st.data_editor(
                pd.DataFrame({"Binaural": _as_list(p.get("binaural"))}),
                num_rows="dynamic", use_container_width=True, hide_index=True
            )

            notas = st.text_area("Notas", value=str(row.get("notes") or ""), key=K("cristais","notes"))

            # Monta payload a partir dos grids
            def _clean_list(df_col, key):
                if df_col is None or df_col.empty or key not in df_col.columns:
                    return []
                vals = [v for v in df_col[key].tolist() if (isinstance(v, str) and v.strip()) or pd.notnull(v)]
                # normaliza strings
                vals = [v.strip() if isinstance(v, str) else v for v in vals]
                return vals

            payload = {
                "chakra":   _clean_list(grid_chak, "chakra"),
                "color":    _clean_list(grid_col,  "cor"),
                "keywords": _clean_list(grid_kw,   "palavra"),
                "benefits": _clean_list(grid_ben,  "beneficio"),
                "pairing_freq": {
                    "hz":        _clean_list(grid_pair_hz,  "Hz"),
                    "solfeggio": _clean_list(grid_pair_sol, "Solfeggio"),
                    "binaural":  _clean_list(grid_pair_bin, "Binaural"),
                },
                "notes": (notas or None)
            }

            # Botões
            c1, c2 = st.columns(2)
            if c1.button("💾 Salvar alterações", use_container_width=True, key=K("cristais","save")) and sb:
                try:
                    q = sb.table("crystals").update(payload)
                    if pd.notnull(crystal_id):
                        q = q.eq("id", int(crystal_id))
                    else:
                        q = q.eq("name", sel)
                    q.execute()
                    st.success("Cristal atualizado.")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Erro ao salvar: {getattr(e,'message',e)}")

    # ---- Adicionar cristal usando grids (sem JSON livre) ----
    with st.expander("Adicionar cristal"):
        with st.form(K("cristais","form_add")):
            novo_nome = st.text_input("Nome", value="Quartzo Rosa", key=K("cristais","novo","nome"))

            grid_new_chak = st.data_editor(
                pd.DataFrame({"chakra":[]}), num_rows="dynamic", use_container_width=True, hide_index=True,
                column_config={"chakra": st.column_config.SelectboxColumn("Chakra", options=["raiz","sacral","plexo","cardiaco","laringeo","terceiro_olho","coronal"])}
            )
            grid_new_cor  = st.data_editor(pd.DataFrame({"cor":[]}), num_rows="dynamic", use_container_width=True, hide_index=True)
            grid_new_kw   = st.data_editor(pd.DataFrame({"palavra":[]}), num_rows="dynamic", use_container_width=True, hide_index=True)
            grid_new_ben  = st.data_editor(pd.DataFrame({"beneficio":[]}), num_rows="dynamic", use_container_width=True, hide_index=True)

            st.markdown("**Pairing de frequência**")
            grid_new_hz  = st.data_editor(pd.DataFrame({"Hz":[]}), num_rows="dynamic", use_container_width=True, hide_index=True,
                                          column_config={"Hz": st.column_config.NumberColumn("Hz", step=1.0)})
            grid_new_sol = st.data_editor(pd.DataFrame({"Solfeggio":[]}), num_rows="dynamic", use_container_width=True, hide_index=True)
            grid_new_bin = st.data_editor(pd.DataFrame({"Binaural":[]}), num_rows="dynamic", use_container_width=True, hide_index=True)

            new_notas = st.text_area("Notas", key=K("cristais","novo","notas"))

            if st.form_submit_button("➕ Salvar novo cristal", use_container_width=True) and sb:
                try:
                    def _pick(df_, col):
                        return [x for x in (df_[col].tolist() if (df_ is not None and col in df_.columns) else [])
                                if (isinstance(x,str) and x.strip()) or pd.notnull(x)]

                    sb.table("crystals").upsert({
                        "name": (novo_nome or "").strip(),
                        "chakra":   _pick(grid_new_chak, "chakra"),
                        "color":    _pick(grid_new_cor,  "cor"),
                        "keywords": _pick(grid_new_kw,   "palavra"),
                        "benefits": _pick(grid_new_ben,  "beneficio"),
                        "pairing_freq": {
                            "hz": _pick(grid_new_hz, "Hz"),
                            "solfeggio": _pick(grid_new_sol, "Solfeggio"),
                            "binaural":  _pick(grid_new_bin, "Binaural"),
                        },
                        "notes": (new_notas or None)
                    }).execute()
                    st.success("Cristal criado.")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Erro ao criar: {getattr(e,'message',e)}")


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
    import pandas as pd
    import json
    import uuid

    st.subheader("Biblioteca — Templates")

    if not sb:
        st.warning("Configure SUPABASE_URL/KEY para usar a Biblioteca.")
        st.stop()

    # ---------------- helpers ----------------
    def _is_uuid(x):
        try:
            uuid.UUID(str(x))
            return True
        except Exception:
            return False

    def _norm_freqs(freqs):
        """
        freqs pode ser None, list[dict] ou dict. Normaliza para DataFrame:
        colunas: code, nome, hz, tipo, chakra
        """
        rows = []
        if isinstance(freqs, list):
            for x in freqs:
                if isinstance(x, dict):
                    rows.append({
                        "code": x.get("code") or "",
                        "nome": x.get("nome") or "",
                        "hz": x.get("hz"),
                        "tipo": x.get("tipo") or "",
                        "chakra": x.get("chakra") or "",
                    })
        elif isinstance(freqs, dict):
            # se vier dict solto, tenta achar lista em alguma chave comum
            lst = freqs.get("items") or freqs.get("frequencias") or []
            if isinstance(lst, list):
                for x in lst:
                    if isinstance(x, dict):
                        rows.append({
                            "code": x.get("code") or "",
                            "nome": x.get("nome") or "",
                            "hz": x.get("hz"),
                            "tipo": x.get("tipo") or "",
                            "chakra": x.get("chakra") or "",
                        })

        if not rows:
            rows = [{"code": "", "nome": "", "hz": None, "tipo": "", "chakra": ""}]

        dfX = pd.DataFrame(rows, columns=["code", "nome", "hz", "tipo", "chakra"])
        dfX["hz"] = pd.to_numeric(dfX["hz"], errors="coerce")
        return dfX

    def _norm_rb(rb):
        rows = []
        if isinstance(rb, list):
            for x in rb:
                if isinstance(x, dict):
                    rows.append({
                        "ordem": x.get("ordem", 0),
                        "carrier_hz": x.get("carrier_hz"),
                        "beat_hz": x.get("beat_hz"),
                        "dur_min": x.get("dur_min"),
                        "obs": x.get("obs") or "",
                    })

        if not rows:
            rows = [{"ordem": 1, "carrier_hz": 220.0, "beat_hz": 10.0, "dur_min": 10, "obs": ""}]

        dfX = pd.DataFrame(rows, columns=["ordem", "carrier_hz", "beat_hz", "dur_min", "obs"])
        dfX["ordem"] = pd.to_numeric(dfX["ordem"], errors="coerce").fillna(0).astype(int)
        dfX = dfX.sort_values("ordem").reset_index(drop=True)
        dfX["ordem"] = range(1, len(dfX) + 1)
        return dfX

    def _norm_cama(cp):
        # cp pode ser: uuid (id do preset), string (nome), dict {"etapas":[...]}, ou lista de etapas
        etapas = []

        if isinstance(cp, str) and cp.strip():
            s = cp.strip()

            # 1) Se parece UUID: busca por id
            if _is_uuid(s):
                try:
                    row = sb.table("cama_presets").select("nome,etapas").eq("id", s).limit(1).execute().data
                    if row:
                        etapas = row[0].get("etapas") or []
                except Exception:
                    etapas = []

            # 2) Senão: trata como nome
            if not etapas:
                try:
                    row = sb.table("cama_presets").select("nome,etapas").eq("nome", s).limit(1).execute().data
                    if row:
                        etapas = row[0].get("etapas") or []
                except Exception:
                    etapas = []

        elif isinstance(cp, dict):
            etapas = cp.get("etapas") or []
        elif isinstance(cp, list):
            etapas = cp

        rows = []
        for i, x in enumerate(etapas, start=1):
            if isinstance(x, dict):
                rows.append({
                    "ordem": x.get("ordem", i),
                    "chakra": (x.get("chakra") or ""),
                    "cor": (x.get("cor") or ""),
                    "min": int(x.get("min") or 5),
                })
            else:
                rows.append({"ordem": i, "chakra": "", "cor": "", "min": 5})

        if not rows:
            rows = [{"ordem": i, "chakra": v, "cor": "", "min": 5} for i, v in enumerate(
                ["raiz", "sacral", "plexo", "cardiaco", "laringeo", "terceiro_olho", "coronal"], start=1
            )]

        dfX = pd.DataFrame(rows, columns=["ordem", "chakra", "cor", "min"])
        dfX["ordem"] = pd.to_numeric(dfX["ordem"], errors="coerce").fillna(0).astype(int)
        dfX = dfX.sort_values("ordem").reset_index(drop=True)
        dfX["ordem"] = range(1, len(dfX) + 1)
        return dfX

    def _pack_freqs(dfX):
        out = []
        if dfX is None or dfX.empty:
            return out
        for _, r in dfX.iterrows():
            code = (str(r.get("code") or "").strip())
            nome = (str(r.get("nome") or "").strip())
            hz = r.get("hz")
            hz = float(hz) if pd.notnull(hz) else None
            tipo = (str(r.get("tipo") or "").strip())
            chakra = (str(r.get("chakra") or "").strip())
            if not code and hz is None and not nome and not tipo and not chakra:
                continue
            out.append({
                "code": code or None,
                "nome": nome or None,
                "hz": hz,
                "tipo": tipo or None,
                "chakra": chakra or None
            })
        return out

    def _pack_rb(dfX):
        out = []
        if dfX is None or dfX.empty:
            return out
        for _, r in dfX.iterrows():
            try:
                ordem = int(r.get("ordem") or 0)
            except Exception:
                ordem = 0
            carrier = r.get("carrier_hz"); carrier = float(carrier) if pd.notnull(carrier) else None
            beat    = r.get("beat_hz");    beat    = float(beat)    if pd.notnull(beat)    else None
            dur     = r.get("dur_min");    dur     = int(dur)       if pd.notnull(dur)     else None
            obs     = str(r.get("obs") or "").strip() or None
            if not carrier and not beat and not dur and not obs:
                continue
            out.append({"ordem": ordem or len(out) + 1, "carrier_hz": carrier, "beat_hz": beat, "dur_min": dur, "obs": obs})
        out.sort(key=lambda x: x.get("ordem") or 0)
        for i, x in enumerate(out, start=1):
            x["ordem"] = i
        return out

    def _save_cama_preset_and_get_id(preset_name: str, df_cama):
        """
        Salva/atualiza cama_presets e devolve o UUID.
        Não depende de UNIQUE em nome: se existir pelo nome, atualiza; senão, insere.
        """
        etapas = []
        if df_cama is not None and not df_cama.empty:
            _d = df_cama.copy()
            _d["ordem"] = pd.to_numeric(_d["ordem"], errors="coerce").fillna(0).astype(int)
            _d = _d.sort_values("ordem").reset_index(drop=True)
            _d["ordem"] = range(1, len(_d) + 1)

            for _, r in _d.iterrows():
                ch = str(r.get("chakra") or "").strip()
                if not ch:
                    continue
                cr = str(r.get("cor") or "").strip() or None
                mn = int(pd.to_numeric(r.get("min"), errors="coerce") or 5)
                etapas.append({"ordem": int(r.get("ordem")), "chakra": ch, "cor": cr, "min": max(1, mn)})

        dur = int(sum(e["min"] for e in etapas)) if etapas else 0

        payload = {
            "nome": preset_name,
            "etapas": etapas,
            "duracao_min": dur,
            "notas": "Auto-gerado pelo template"
        }

        # busca por nome
        try:
            got = sb.table("cama_presets").select("id").eq("nome", preset_name).limit(1).execute().data or []
        except Exception:
            got = []

        if got:
            cama_id = got[0].get("id")
            sb.table("cama_presets").update(payload).eq("id", cama_id).execute()
            return cama_id

        ins = sb.table("cama_presets").insert(payload).execute().data or []
        return ins[0].get("id") if ins else None

    # ---------------- dados ----------------
    tpls = sb_select(
        "therapy_templates",
        "id,name,objetivo,roteiro_binaural,frequencias_suporte,cama_preset,phyto_plan,notas",
        order="name"
    )
    nomes = [t.get("name") for t in (tpls or []) if t.get("name")]
    mapa  = {t.get("name"): t for t in (tpls or []) if t.get("name")}

    NEW_LABEL = "(+ Novo template…)"
    choices = [NEW_LABEL] + nomes
    sel = st.selectbox("Template", choices or [NEW_LABEL], key=K("biblioteca", "template"))

    # ---------------- UI ----------------
    if sel == NEW_LABEL:
        # ---------- NOVO TEMPLATE ----------
        with st.form(K("biblioteca", "novo_form")):
            c1, c2 = st.columns([2, 1])
            novo_nome = c1.text_input("Nome do template", value="Meu Template", key=K("biblioteca", "novo_nome"))
            objetivo  = c2.text_input("Objetivo (curto)", value="", key=K("biblioteca", "novo_obj"))
            st.caption("Edite os grids abaixo e clique em **Criar**.")

            st.markdown("### Frequências de suporte")
            df_freqs = st.data_editor(
                _norm_freqs([]),
                num_rows="dynamic", use_container_width=True, hide_index=True,
                key=K("biblioteca", "novo_freqs"),
                column_config={
                    "hz": st.column_config.NumberColumn("Hz", step=1.0),
                    "chakra": st.column_config.SelectboxColumn(
                        "Chakra",
                        options=["", "raiz", "sacral", "plexo", "cardiaco", "laringeo", "terceiro_olho", "coronal"]
                    ),
                }
            )

            st.markdown("### Roteiro binaural")
            df_rb = st.data_editor(
                _norm_rb([]),
                num_rows="dynamic", use_container_width=True, hide_index=True,
                key=K("biblioteca", "novo_rb"),
                column_config={
                    "ordem": st.column_config.NumberColumn("Ordem", min_value=1, step=1),
                    "carrier_hz": st.column_config.NumberColumn("Carrier (Hz)", step=1.0),
                    "beat_hz": st.column_config.NumberColumn("Beat (Hz)", step=0.5),
                    "dur_min": st.column_config.NumberColumn("Duração (min)", step=1),
                    "obs": st.column_config.TextColumn("Observações"),
                }
            )

            st.markdown("### Cama — etapas")
            df_cama = st.data_editor(
                _norm_cama([]),
                num_rows="dynamic", use_container_width=True, hide_index=True,
                key=K("biblioteca", "novo_cama"),
                column_config={
                    "ordem": st.column_config.NumberColumn("Ordem", min_value=1, step=1),
                    "chakra": st.column_config.SelectboxColumn(
                        "Chakra",
                        options=["raiz", "sacral", "plexo", "cardiaco", "laringeo", "terceiro_olho", "coronal"]
                    ),
                    "cor": st.column_config.TextColumn("Cor"),
                    "min": st.column_config.NumberColumn("Minutos", min_value=1, step=1),
                }
            )
            st.caption(f"Duração total prevista: **{int(pd.to_numeric(df_cama['min'], errors='coerce').fillna(0).sum())} min**")

            notas = st.text_area("Notas do template", key=K("biblioteca", "novo_notas"))

            criar = st.form_submit_button("➕ Criar template", use_container_width=True)

        if criar and sb:
            nome_ok = (novo_nome or "").strip()
            if not nome_ok:
                st.warning("Informe um **nome** para o template.")
                st.stop()
            if nome_ok in nomes:
                st.error("Já existe um template com esse nome. Escolha outro.")
                st.stop()

            cama_name = f"{nome_ok} — Cama"
            cama_id = _save_cama_preset_and_get_id(cama_name, df_cama)
            if not cama_id:
                st.error("Não foi possível criar/obter o preset da cama (cama_presets).")
                st.stop()

            payload = {
                "name": nome_ok,
                "objetivo": objetivo or None,
                "frequencias_suporte": _pack_freqs(df_freqs),
                "roteiro_binaural": _pack_rb(df_rb),
                "cama_preset": cama_id,  # ✅ UUID
                "phyto_plan": None,
                "notas": (notas or None),
            }

            try:
                sb.table("therapy_templates").insert(payload).execute()
                st.success(f"Template **{nome_ok}** criado.")
                st.cache_data.clear()
            except Exception as e:
                _api_err(e, "therapy_templates.insert")

    else:
        # ---------- EDITAR/CLONAR EXISTENTE ----------
        t = mapa.get(sel) or {}
        template_id = t.get("id")

        st.text_input("Objetivo", value=str(t.get("objetivo", "")), key=K("biblioteca", "objetivo"), disabled=True)
        st.caption("Edite os componentes abaixo em grid (as alterações serão salvas no template).")

        st.markdown("### Frequências de suporte (grid)")
        df_freqs = st.data_editor(
            _norm_freqs(t.get("frequencias_suporte")),
            num_rows="dynamic", use_container_width=True, hide_index=True,
            key=K("biblioteca", "freqs"),
            column_config={
                "hz": st.column_config.NumberColumn("Hz", step=1.0),
                "chakra": st.column_config.SelectboxColumn(
                    "Chakra",
                    options=["", "raiz", "sacral", "plexo", "cardiaco", "laringeo", "terceiro_olho", "coronal"]
                ),
            }
        )

        st.markdown("### Roteiro binaural (grid)")
        df_rb = st.data_editor(
            _norm_rb(t.get("roteiro_binaural")),
            num_rows="dynamic", use_container_width=True, hide_index=True,
            key=K("biblioteca", "rb"),
            column_config={
                "ordem": st.column_config.NumberColumn("Ordem", min_value=1, step=1),
                "carrier_hz": st.column_config.NumberColumn("Carrier (Hz)", step=1.0),
                "beat_hz": st.column_config.NumberColumn("Beat (Hz)", step=0.5),
                "dur_min": st.column_config.NumberColumn("Duração (min)", step=1),
                "obs": st.column_config.TextColumn("Observações"),
            }
        )

        st.markdown("### Cama — etapas (grid)")
        df_cama = st.data_editor(
            _norm_cama(t.get("cama_preset")),
            num_rows="dynamic", use_container_width=True, hide_index=True,
            key=K("biblioteca", "cama"),
            column_config={
                "ordem": st.column_config.NumberColumn("Ordem", min_value=1, step=1),
                "chakra": st.column_config.SelectboxColumn(
                    "Chakra",
                    options=["raiz", "sacral", "plexo", "cardiaco", "laringeo", "terceiro_olho", "coronal"]
                ),
                "cor": st.column_config.TextColumn("Cor"),
                "min": st.column_config.NumberColumn("Minutos", min_value=1, step=1),
            }
        )
        st.caption(f"Duração total prevista: **{int(pd.to_numeric(df_cama['min'], errors='coerce').fillna(0).sum())} min**")

        notas = st.text_area("Notas do template", value=str(t.get("notas") or ""), key=K("biblioteca", "notas"))

        c1, c2 = st.columns(2)
        if c1.button("💾 Salvar alterações do template", use_container_width=True, key=K("biblioteca", "save")) and sb:
            try:
                cama_name = f"{sel} — Cama"
                cama_id = _save_cama_preset_and_get_id(cama_name, df_cama)
                if not cama_id:
                    st.error("Não foi possível salvar o preset da cama (cama_presets).")
                    st.stop()

                payload_upd = {
                    "frequencias_suporte": _pack_freqs(df_freqs),
                    "roteiro_binaural": _pack_rb(df_rb),
                    "cama_preset": cama_id,  # ✅ UUID
                    "notas": (notas or None),
                }

                q = sb.table("therapy_templates").update(payload_upd)

                # id pode ser uuid ou int: trata ambos
                if template_id:
                    if _is_uuid(template_id):
                        q = q.eq("id", str(template_id))
                    else:
                        try:
                            q = q.eq("id", int(template_id))
                        except Exception:
                            q = q.eq("name", sel)
                else:
                    q = q.eq("name", sel)

                q.execute()
                st.success("Template atualizado.")
                st.cache_data.clear()
            except Exception as e:
                _api_err(e, "therapy_templates.update")

        with st.expander("🧬 Duplicar como…"):
            novo_nome = st.text_input("Novo nome", value=f"{sel} (cópia)", key=K("biblioteca", "dup_nome"))
            objetivo_dup = st.text_input("Objetivo", value=str(t.get("objetivo") or ""), key=K("biblioteca", "dup_obj"))

            if st.button("Duplicar", key=K("biblioteca", "dup_btn")) and sb:
                name_ok = (novo_nome or "").strip()
                if not name_ok:
                    st.warning("Informe um **novo nome** para a cópia.")
                elif name_ok in nomes:
                    st.error("Já existe um template com esse nome.")
                else:
                    try:
                        cama_name_dup = f"{name_ok} — Cama"
                        cama_id_dup = _save_cama_preset_and_get_id(cama_name_dup, df_cama)
                        if not cama_id_dup:
                            st.error("Não foi possível criar/obter o preset da cama (cama_presets).")
                            st.stop()

                        payload_ins = {
                            "name": name_ok,
                            "objetivo": objetivo_dup or None,
                            "frequencias_suporte": _pack_freqs(df_freqs),
                            "roteiro_binaural": _pack_rb(df_rb),
                            "cama_preset": cama_id_dup,  # ✅ UUID
                            "phyto_plan": t.get("phyto_plan") or None,
                            "notas": (notas or None),
                        }
                        sb.table("therapy_templates").insert(payload_ins).execute()
                        st.success(f"Criado **{name_ok}**.")
                        st.cache_data.clear()
                    except Exception as e:
                        _api_err(e, "therapy_templates.insert")



# ========== Emoções ==========
with tabs[11]:
    st.subheader("Emoções — mapa e recomendações")

    if not sb:
        st.warning("Configure SUPABASE_URL/KEY para carregar o mapa emocional.")
    else:
        # busca colunas reais do schema (sem 'escala' e sem 'id')
        cols = "emocao,nivel,hz_ref,polaridade,frequencias,binaural_beat_hz,carrier_hz,chakras,cores,cristais,afirmacoes,respiracao,notas"
        try:
            rows = sb.table("emotions_map").select(cols).order("nivel", desc=False).execute().data
        except Exception as e:
            st.error(f"[emotions_map] erro: {getattr(e, 'message', e)}")
            rows = []

        df = pd.DataFrame(rows)
        if df.empty:
            st.info(
                "A tabela `emotions_map` está vazia ou sem permissão de leitura.\n"
                "Colunas esperadas: emocao, nivel, hz_ref, polaridade, frequencias(JSON), "
                "binaural_beat_hz, carrier_hz, chakras(JSON), cores(JSON), cristais(JSON), "
                "afirmacoes(JSON), respiracao, notas."
            )
        else:
            # cria uma coluna 'escala' amigável (usa nivel; se não houver, cai pra hz_ref)
            if "nivel" in df.columns:
                df["escala"] = df["nivel"]
            elif "hz_ref" in df.columns:
                df["escala"] = df["hz_ref"]
            else:
                df["escala"] = None

            # helpers para “embelezar” colunas json/list
            def fmt_list_or_dict(x):
                if isinstance(x, list):
                    return ", ".join(map(str, x))
                if isinstance(x, dict):
                    # tenta deixar compacto: solfeggio/hz principais
                    s = []
                    if "solfeggio" in x and isinstance(x["solfeggio"], list):
                        s.append("Solfeggio: " + ", ".join(map(str, x["solfeggio"])))
                    if "hz" in x and isinstance(x["hz"], list):
                        s.append("Hz: " + ", ".join(map(lambda v: f"{v}", x["hz"])))
                    return " | ".join(s) if s else json.dumps(x, ensure_ascii=False)
                return x

            show = pd.DataFrame({
                "Emoção": df["emocao"],
                "Escala": df["escala"],
                "Polaridade": df["polaridade"].str.title(),
                "Frequências": df["frequencias"].apply(fmt_list_or_dict),
                "Beat (Hz)": df["binaural_beat_hz"],
                "Carrier (Hz)": df["carrier_hz"],
                "Chakras": df["chakras"].apply(fmt_list_or_dict),
                "Cores": df["cores"].apply(fmt_list_or_dict),
                "Cristais": df["cristais"].apply(fmt_list_or_dict),
                "Afirmações": df["afirmacoes"].apply(fmt_list_or_dict),
                "Respiração": df["respiracao"],
                "Notas": df["notas"],
            })

            st.dataframe(show, use_container_width=True, hide_index=True)

            # UI de aplicação rápida: escolher emoção e tocar/gerar recomendações
            st.markdown("### Aplicar e tocar rapidamente")
            c1, c2, c3 = st.columns([2,1,1])
            emo = c1.selectbox("Escolha a emoção", df["emocao"].tolist(), key=K("emo","sel"))
            dur = c2.number_input("Duração (s)", 10, 1800, 120, 5, key=K("emo","dur"))
            vol = c3.slider("Volume (binaural)", 0.0, 0.4, 0.12, 0.01, key=K("emo","vol"))

            row = df[df["emocao"] == emo].iloc[0]
            # extrai bate-estaca seguro
            beat = float(row.get("binaural_beat_hz") or 10.0)
            carrier = float(row.get("carrier_hz") or 220.0)

            # info lateral
            mL, mR, m3 = st.columns(3)
            mL.metric("Beat (Hz)", f"{beat:.2f}")
            mR.metric("Carrier (Hz)", f"{carrier:.2f}")
            m3.metric("Escala", str(row.get("escala")))

            # tocar binaural
            st.components.v1.html(
                webaudio_binaural_html(carrier, beat, int(dur), None, float(vol)),
                height=300
            )
            wav = synth_binaural_wav(carrier, beat, seconds=min(int(dur), 20))
            st.audio(wav, format="audio/wav")
            st.download_button(
                "Baixar WAV (20s)",
                data=wav,
                file_name=f"binaural_{int(carrier)}_{beat:.1f}.wav",
                mime="audio/wav",
                key=K("emo","dl")
            )

            with st.expander("Detalhes e recomendações"):
                st.write("**Frequências sugeridas:**", fmt_list_or_dict(row.get("frequencias")))
                st.write("**Chakras:**", fmt_list_or_dict(row.get("chakras")))
                st.write("**Cores:**", fmt_list_or_dict(row.get("cores")))
                st.write("**Cristais:**", fmt_list_or_dict(row.get("cristais")))
                # afirmações em lista
                af = row.get("afirmacoes")
                if isinstance(af, list):
                    st.markdown("**Afirmações:**")
                    for a in af:
                        st.markdown(f"- {a}")
                else:
                    st.write("**Afirmações:**", af)
                st.write("**Respiração:**", row.get("respiracao"))
                st.caption(row.get("notas") or "")
                


# ========== Emoções ==========
with tabs[12]:

    st.subheader("🗂️ Sessões salvas")

   # if not sb:
      #  st.warning("Configure SUPABASE_URL/KEY para habilitar a leitura das sessões.")
       # st.stop()

    # ---- Dados base (pacientes, frequências, presets) ----
    pts = sb_select("patients", "id,nome", order="created_at", desc=True, limit=1000)
    mapa_pid_nome = {}
    try:
        for p in pts or []:
            pid = p.get("id")
            if pid is not None and str(pid).strip() != "":
                mapa_pid_nome[int(pid)] = str(p.get("nome") or "—")
    except Exception:
        pass

    freqs = sb_select("frequencies", "code,nome,hz,tipo,chakra,cor", order="code", desc=False, limit=2000)
    freq_by_code = {str(f.get("code") or "").strip(): f for f in (freqs or [])}
    # labels usados no Planner (ex.: "SOL528 • Solfeggio 528 Hz")
    def _freq_label(f):
        code = str(f.get("code") or "").strip()
        nome = str(f.get("nome") or "").strip()
        return f"{code} • {nome}" if nome else code
    planner_labels_by_code = {c: _freq_label(f) for c, f in freq_by_code.items()}

    pres_bina = sb_select("binaural_presets", "id,nome,carrier_hz,beat_hz,duracao_min", order="nome")
    pres_bina_by_name = {str(p.get("nome") or "").strip(): p for p in (pres_bina or [])}

    pres_cama = sb_select("cama_presets", "id,nome,etapas,duracao_min,notas", order="nome")
    pres_cama_by_name = {str(c.get("nome") or "").strip(): c for c in (pres_cama or [])}

    plans_phyto = sb_select("phytotherapy_plans", "id,name", order="name")
    plan_by_name = {str(p.get("name") or "").strip(): p for p in (plans_phyto or [])}

    # ---- Carrega sessões ----
    # Tenta com campos explícitos; se a RLS/colunas falhar, sb_select cai para '*'
    rows = sb_select("sessions", "id,patient_id,data,tipo,status,protocolo,patients(nome)", order="data", desc=True, limit=400)
    df = pd.DataFrame(rows)

    if df.empty:
        st.info("Nenhuma sessão encontrada ainda.")
        st.stop()

    # Normalizações de colunas
    if "data" in df.columns:
        try:
            # Mostra em horário local (sem tz) quando possível
            df["data"] = pd.to_datetime(df["data"], errors="coerce").dt.tz_convert(None)
        except Exception:
            df["data"] = pd.to_datetime(df["data"], errors="coerce")

    # Nome do paciente (usa relacionamento ou mapeamento local)
    def _nome_paciente(r):
        # se veio o join: patients = {"nome": "..."}
        pjoin = r.get("patients")
        if isinstance(pjoin, dict) and "nome" in pjoin:
            return pjoin.get("nome") or "—"
        # senão tenta pelo patient_id
        pid = r.get("patient_id")
        try:
            if pd.notnull(pid):
                return mapa_pid_nome.get(int(pid), "—")
        except Exception:
            pass
        return "—"

    if "Paciente" not in df.columns:
        df["Paciente"] = df.apply(lambda r: _nome_paciente(r), axis=1)

    # ---- Filtros ----
    c1, c2, c3, c4 = st.columns([2,2,2,2])
    pac_opts = ["(todos)"] + sorted([p for p in df["Paciente"].dropna().unique().tolist() if p and p != "—"])
    f_pac = c1.selectbox("Paciente", pac_opts, index=0, key=K("sessoes", "f", "pac"))

    status_opts = ["(todos)"] + sorted([s for s in df.get("status", pd.Series([], dtype=str)).dropna().astype(str).unique().tolist() if s])
    f_sta = c2.selectbox("Status", status_opts, index=0, key=K("sessoes", "f", "status"))

    min_d, max_d = None, None
    if "data" in df.columns and not df["data"].dropna().empty:
        dser = pd.to_datetime(df["data"], errors="coerce").dropna()
        if not dser.empty:
            min_d = dser.min().date()
            max_d = dser.max().date()
    f_de = c3.date_input("De", value=min_d if min_d else date.today(), key=K("sessoes","f","de"))
    f_ate = c4.date_input("Até", value=max_d if max_d else date.today(), key=K("sessoes","f","ate"))

    dff = df.copy()
    if f_pac != "(todos)":
        dff = dff[dff["Paciente"] == f_pac]
    if f_sta != "(todos)" and "status" in dff.columns:
        dff = dff[dff["status"].astype(str) == f_sta]
    if "data" in dff.columns and (f_de or f_ate):
        dser = pd.to_datetime(dff["data"], errors="coerce")
        if f_de:
            dff = dff[dser.dt.date >= f_de]
        if f_ate:
            dff = dff[dser.dt.date <= f_ate]

    # ---- Lista em grid ----
    show_cols = [c for c in ["data","Paciente","tipo","status"] if c in dff.columns]
    st.dataframe(dff[show_cols] if show_cols else dff, use_container_width=True, hide_index=True)

    # ---- Seleção de uma sessão para ações ----
    # rótulo amigável: "#id · yyyy-mm-dd HH:MM · Paciente"
    def _lab(r):
        sid = r.get("id")
        dtv = r.get("data")
        try:
            dtv = pd.to_datetime(dtv, errors="coerce")
            dts = dtv.strftime("%Y-%m-%d %H:%M") if pd.notnull(dtv) else "—"
        except Exception:
            dts = str(dtv)
        return f"#{sid} · {dts} · {r.get('Paciente','—')}"

    opts = [(_lab(r), r.get("id")) for _, r in dff.iterrows() if pd.notnull(r.get("id"))]
    if not opts:
        st.stop()
    labels = [o[0] for o in opts]
    lab2id = {o[0]: o[1] for o in opts}
    sel_lab = st.selectbox("Selecionar sessão", labels, key=K("sessoes","sel"))
    sel_id = lab2id.get(sel_lab)

    row = df[df["id"] == sel_id].iloc[0] if sel_id in df.get("id", []) .values else None
    if row is None:
        st.warning("Não foi possível localizar os dados desta sessão.")
        st.stop()

    # ---- Detalhes do protocolo ----
    st.markdown("### 📋 Protocolo")
    prot = row.get("protocolo") or {}

    # Frequências
    freq_list = []
    for item in (prot.get("frequencias") or []):
        code = ""
        if isinstance(item, dict):
            code = str(item.get("code") or "").strip()
        else:
            code = str(item or "").strip()
        f = freq_by_code.get(code)
        if f:
            freq_list.append({
                "code": code,
                "nome": f.get("nome"),
                "hz": f.get("hz"),
                "tipo": f.get("tipo"),
                "chakra": f.get("chakra"),
            })
        else:
            if code:
                freq_list.append({"code": code, "nome": None, "hz": None, "tipo": None, "chakra": None})
    df_freq = pd.DataFrame(freq_list)
    if not df_freq.empty:
        st.caption("Frequências de suporte")
        st.dataframe(df_freq, use_container_width=True, hide_index=True)
    else:
        st.caption("Sem frequências cadastradas neste protocolo.")

    # Binaural
    st.caption("Binaural")
    b = prot.get("binaural") or {}
    colB1, colB2, colB3 = st.columns(3)
    try:
        colB1.metric("Carrier (Hz)", f"{float(b.get('carrier_hz') or 220.0):.2f}")
        colB2.metric("Beat (Hz)", f"{float(b.get('beat_hz') or 10.0):.2f}")
        dur_min = int(b.get("duracao_min") or b.get("dur_min") or 10)
        colB3.metric("Duração (min)", f"{dur_min}")
    except Exception:
        colB1.metric("Carrier (Hz)", "—")
        colB2.metric("Beat (Hz)", "—")
        colB3.metric("Duração (min)", "—")

    # Cama
    st.caption("Cama — etapas")
    cama = prot.get("cama") or {}
    etapas = []
    if isinstance(cama, dict):
        # pode ser {"nome": "...", "etapas":[...]} ou direto {"etapas":[...]}
        etapas = cama.get("etapas") or []
    elif isinstance(cama, list):
        etapas = cama
    df_cama = pd.DataFrame(etapas) if etapas else pd.DataFrame(columns=["ordem","chakra","cor","min"])
    if not df_cama.empty:
        # normaliza e ordena
        for col in ["ordem","chakra","cor","min"]:
            if col not in df_cama.columns:
                df_cama[col] = None
        df_cama["ordem"] = pd.to_numeric(df_cama["ordem"], errors="coerce")
        df_cama = df_cama.sort_values("ordem", na_position="last").reset_index(drop=True)
        df_cama["ordem"] = range(1, len(df_cama)+1)
        st.dataframe(df_cama[["ordem","chakra","cor","min"]], use_container_width=True, hide_index=True)
        try:
            st.caption(f"🕒 Duração total prevista: **{int(pd.to_numeric(df_cama['min'], errors='coerce').fillna(0).sum())} min**")
        except Exception:
            pass
    else:
        st.caption("Sem etapas de cama neste protocolo.")

    # Fitoterapia
    st.caption("Fitoterapia (plano associado)")
    phyto = prot.get("fitoterapia_plan") or {}
    if phyto:
        st.write({k: phyto.get(k) for k in ["id","name","objetivo","cadencia","duracao_sem"] if k in phyto})
    else:
        st.write("—")

    if prot.get("notas"):
        st.markdown("**Notas do protocolo:**")
        st.write(prot.get("notas"))

    st.markdown("---")

    # ---- Ações rápidas ----
    cA1, cA2, cA3, cA4 = st.columns([1,1,1,2])

    # Atualizar status
    status_atual = str(row.get("status") or "rascunho")
    novo_status = cA1.selectbox("Status", ["rascunho","em andamento","concluida","cancelada"],
                                index=["rascunho","em andamento","concluida","cancelada"].index(status_atual)
                                if status_atual in ["rascunho","em andamento","concluida","cancelada"] else 0,
                                key=K("sessoes","status_sel"))
    if cA2.button("💾 Atualizar status", use_container_width=True, key=K("sessoes","btn_status")):
        try:
            sb.table("sessions").update({"status": novo_status}).eq("id", int(sel_id)).execute()
            st.success("Status atualizado.")
            st.cache_data.clear()
        except Exception as e:
            st.error(f"Erro ao atualizar status: {getattr(e,'message',e)}")

    # Duplicar sessão
    if cA3.button("🧬 Duplicar como rascunho", use_container_width=True, key=K("sessoes","btn_dup")):
        try:
            payload = {
                "patient_id": row.get("patient_id"),
                "data": datetime.utcnow().isoformat(),
                "tipo": row.get("tipo") or "Misto",
                "status": "rascunho",
                "protocolo": prot,
            }
            sb.table("sessions").insert(payload).execute()
            st.success("Sessão duplicada como rascunho.")
            st.cache_data.clear()
        except Exception as e:
            st.error(f"Erro ao duplicar: {getattr(e,'message',e)}")

    # Excluir sessão
    with cA4:
        colX1, colX2 = st.columns([2,1])
        ok_del = colX1.checkbox("Confirmar exclusão", key=K("sessoes","ok_del"))
        if colX2.button("🗑️ Excluir", use_container_width=True, key=K("sessoes","btn_del")):
            if not ok_del:
                st.warning("Marque 'Confirmar exclusão' para remover.")
            else:
                try:
                    sb.table("sessions").delete().eq("id", int(sel_id)).execute()
                    st.success("Sessão excluída.")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Erro ao excluir: {getattr(e,'message',e)}")

    st.markdown("---")

    # ---- Aplicar no Planner (preenche campos nas outras abas) ----
    st.markdown("### 🚀 Aplicar no Planner")
    st.caption("Isso pré-seleciona paciente, frequências e tenta alinhar presets nas abas correspondentes.")

    if st.button("Aplicar seleção no Planner", key=K("sessoes","aplicar_planner"), use_container_width=True):
        try:
            # Paciente
            pac_label = row.get("Paciente") or mapa_pid_nome.get(int(row.get("patient_id") or 0), None)
            if pac_label:
                st.session_state[K("planner","paciente")] = pac_label

            # Frequências (monta labels iguais aos do Planner)
            prot_freqs = [ (x.get("code") if isinstance(x,dict) else str(x)) for x in (prot.get("frequencias") or []) ]
            labels_ok = [planner_labels_by_code[c] for c in prot_freqs if c in planner_labels_by_code]
            if labels_ok:
                st.session_state[K("planner","freqs")] = labels_ok

            # Binaural (tenta setar pelo nome do preset se existir)
            bnome = str((prot.get("binaural") or {}).get("nome") or "").strip()
            if bnome and bnome in pres_bina_by_name:
                st.session_state[K("planner","binaural")] = bnome

            # Cama (preset por nome)
            cnome = str((prot.get("cama") or {}).get("nome") or "").strip()
            if cnome and cnome in pres_cama_by_name:
                st.session_state[K("planner","cama")] = cnome

            # Plano fito (por name)
            pname = str((prot.get("fitoterapia_plan") or {}).get("name") or "").strip()
            if pname and pname in plan_by_name:
                st.session_state[K("planner","fitoplan")] = pname

            # Notas
            if prot.get("notas"):
                st.session_state[K("planner","notas")] = str(prot.get("notas"))

            st.success("Aplicado! Abra a aba “Sessão (Planner)” para conferir os campos preenchidos.")
        except Exception as e:
            st.error(f"Não foi possível aplicar no Planner: {getattr(e,'message',e)}")

    st.caption("Dica: você pode tocar imediatamente o binaural acima na aba **Binaural** usando os mesmos parâmetros.")

