# app.py — MVP Clínico Holístico (com keys únicas)
# Funcionalidades: Pacientes, Anamnese, Agenda, Planner de Sessão,
# Frequências, Binaural (com música de fundo), Cama de Cristal,
# Fitoterapia, Cristais, Financeiro, Biblioteca.
# Variáveis de ambiente (Secrets): SUPABASE_URL, SUPABASE_KEY (anon)

import os, io, json, wave, base64, time, pathlib
from datetime import datetime, timedelta, date
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

st.set_page_config(page_title="Programa Terapias Claudia", layout="wide")

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
    if not sb: return []
    q = sb.table(table).select(cols)
    if order: q = q.order(order, desc=desc)
    if limit: q = q.limit(limit)
    return q.execute().data or []

def synth_binaural_wav(fc: float, beat: float, seconds: float=20.0, sr: int=44100, amp: float=0.2) -> bytes:
    bt = abs(float(beat)); fl = max(1.0, float(fc)-bt/2); fr = float(fc)+bt/2
    t = np.linspace(0, seconds, int(sr*seconds), endpoint=False)
    left = np.sin(2*np.pi*fl*t); right = np.sin(2*np.pi*fr*t)
    # fade in/out
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

def bytes_to_data_url(raw: bytes, filename: str|None) -> tuple[str, str]:
    if not raw: return None, None
    name = (filename or "").lower()
    mime = "audio/mpeg"
    if name.endswith(".wav"): mime="audio/wav"
    elif name.endswith(".ogg") or name.endswith(".oga"): mime="audio/ogg"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}", mime

def webaudio_binaural_html(fc: float, beat: float, seconds: int=60,
                           bg_data_url: str|None=None, bg_gain: float=0.12):
    """Player binaural + música de fundo estável usando HTMLAudioElement (menos bugs que decodeAudioData)."""
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
      // Alguns navegadores só tocam após gesto do usuário — aqui já veio do clique
      await bgAudio.play().catch(()=>{{ /* se falhar, tentaremos via node mesmo assim */ }});
      bgNode = ctx.createMediaElementSource(bgAudio);
      bgGain = ctx.createGain(); bgGain.gain.value = {g:.4f};

      // Forçar MONO: somar L/R e mandar aos dois canais
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
      // Se o play inicial falhou (autoplay policy), tentar novamente agora que temos contexto ativo
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

# ----------------- UI -----------------
st.title("🌿 Doce Conexão - Frequencias/Cama de Cristal/Fito")

tabs = st.tabs([
    "Pacientes","Anamnese","Agenda","Sessão (Planner)","Frequências",
    "Binaural","Cama de Cristal","Fitoterapia","Cristais","Financeiro","Biblioteca"
])

# ========== Pacientes ==========
with tabs[0]:
    st.subheader("Pacientes")
    if not sb: st.warning("Configure SUPABASE_URL/KEY.")

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
                    "nome":nome,
                    "nascimento":str(nasc) if nasc else None,
                    "telefone":tel,
                    "email":email
                }).execute()
                st.success("Paciente salvo.")
                st.cache_data.clear()

    pts = sb_select("patients","id,nome,nascimento,telefone,email,created_at",order="created_at",desc=True,limit=50)
    if pts:
        st.dataframe(pd.DataFrame(pts),use_container_width=True,hide_index=True)

# ========== Anamnese ==========
with tabs[1]:
    st.subheader("Anamnese — Avançada")

    # Paciente
    pts = sb_select("patients","id,nome",order="created_at",desc=True)
    mapa = {p["nome"]:p["id"] for p in pts}
    psel = st.selectbox("Paciente", list(mapa.keys()) or ["—"], key=K("anv","paciente"))

    # Carrega perguntas ativas do banco
    qs = sb_select("anamnesis_questions","section,qkey,label,qtype,min_val,max_val,options,weight,required,ord,active",order="section")
    if not qs:
        st.warning("Nenhuma pergunta cadastrada. Execute o SQL de perguntas (anamnesis_questions).")
    else:
        # agrupar por seção
        dfq = pd.DataFrame(qs)
        dfq = dfq[dfq["active"].fillna(True)]
        sections = list(dfq["section"].dropna().unique())

        # formulário com tabs por seção
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
                            val = st.slider(label, int(mn or 0), int(mx or 10), int((mn or 0 + (mx or 10))//2), key=key)
                        elif qt == "bool":
                            val = st.checkbox(label, key=key)
                        elif qt == "multi":
                            val = st.multiselect(label, opts, key=key)
                        elif qt == "number":
                            val = st.number_input(label, value=float(mn or 0), min_value=float(mn or 0), max_value=float(mx or 9999), step=1.0, key=key)
                        else:
                            val = st.text_area(label, key=key) if (mx or 0) > 200 else st.text_input(label, key=key)

                        respostas[qk] = val

            # cálculo de score (chakras + indicadores)
            def score_from_answers(ans: dict) -> dict:
                # inicia com 0
                chakras = {k:0.0 for k in ["raiz","sacral","plexo","cardiaco","laringeo","terceiro_olho","coronal"]}
                flags = set()
                # aplica pesos das perguntas
                for _, row in dfq.iterrows():
                    qk = row["qkey"]; w = row.get("weight") or {}
                    if qk not in ans: continue
                    val = ans[qk]
                    # flags
                    if isinstance(w, dict) and "flags" in w:
                        if bool(val):
                            for fl in (w["flags"] or []):
                                flags.add(fl)
                    # chakra weights
                    if isinstance(w, dict) and "chakra" in w:
                        for ch, wt in (w["chakra"] or {}).items():
                            try:
                                v = float(val) if isinstance(val,(int,float)) else (1.0 if val else 0.0)
                            except Exception:
                                v = 0.0
                            chakras[ch] = chakras.get(ch,0.0) + (float(wt) * v)

                # normalizações simples
                # Equilíbrio autoavaliado (0–10) podemos reverter para “déficit” (10-v)
                for ch_key, eq_key in [
                    ("raiz","raiz_eq"),("sacral","sacral_eq"),("plexo","plexo_eq"),
                    ("cardiaco","cardiaco_eq"),("laringeo","laringeo_eq"),
                    ("terceiro_olho","terceiro_olho_eq"),("coronal","coronal_eq")
                ]:
                    if eq_key in respostas and isinstance(respostas[eq_key], (int,float)):
                        chakras[ch_key] += max(0.0, 10.0 - float(respostas[eq_key]))

                # índices diretos
                idx = {
                    "sono": float(respostas.get("sono_qualidade") or 0),
                    "estresse": float(respostas.get("estresse_nivel") or 0),
                    "ansiedade": float(respostas.get("ansiedade") or 0),
                }
                return {"chakras": chakras, "flags": sorted(list(flags)), "indices": idx}

            score = score_from_answers(respostas)

            # resumo e recomendações iniciais (sugestão)
            st.markdown("### Resumo")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Chakras (maior = mais atenção):**")
                st.json(score["chakras"])
            with c2:
                st.write("**Flags:**", ", ".join(score["flags"]) if score["flags"] else "—")
                st.write("**Índices:**", score["indices"])

            recs = []
            ch_order = sorted(score["chakras"].items(), key=lambda x: x[1], reverse=True)
            if ch_order:
                top_ch, _ = ch_order[0]
                # mapeamento simples p/ frequências iniciais
                mapa_freq = {
                    "raiz":["SOL396","CHAKRA_RAIZ"],
                    "sacral":["SOL417","CHAKRA_SACRAL"],
                    "plexo":["SOL432","SOL417"],
                    "cardiaco":["SOL528","SOL639","CHAKRA_CARDIACO"],
                    "laringeo":["SOL741","CHAKRA_LARINGEO"],
                    "terceiro_olho":["SOL852","CHAKRA_TERCEIRO_OLHO"],
                    "coronal":["SOL963","CHAKRA_CORONAL"]
                }
                recs.append({"frequencias_sugeridas": mapa_freq.get(top_ch, ["SOL528"])})

            # contraindicações que afetam binaural/herbais
            if "epilepsia" in score["flags"]:
                recs.append({"binaural": "evitar batidas altas (>10Hz); preferir sem binaural"})
            if "gravidez" in score["flags"]:
                recs.append({"fitoterapia": "evitar ervas com contraindicação na gestação; revisar plano"})
            if "marcapasso" in score["flags"]:
                recs.append({"eletromagnético": "evitar campos/ímanes; utilizar apenas luz/sons seguros"})

            if recs:
                st.markdown("### Recomendações iniciais (automáticas)")
                st.json(recs)

            # Salvar
            if st.form_submit_button("Salvar anamnese", use_container_width=True):
                payload = {"patient_id": mapa.get(psel),
                           "respostas": respostas,
                           "score": score}
                sb.table("anamneses").insert(payload).execute()
                st.success("Anamnese salva com sucesso.")

# ========== Agenda ==========
with tabs[2]:
    st.subheader("Agenda")
    pts = sb_select("patients","id,nome",order="created_at",desc=True)
    mapa = {p["nome"]:p["id"] for p in pts}
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
            "notas":notas
        }).execute()
        st.success("Agendado!")

    ag = sb_select("appointments","id,patient_id,inicio,tipo,notas,patients(nome)",order="inicio",desc=False,limit=100)
    if ag:
        df=pd.DataFrame(ag)
        df["inicio"]=pd.to_datetime(df["inicio"]).dt.tz_convert(None)
        df["Paciente"]=df["patients"].apply(lambda x: (x or {}).get("nome","—"))
        st.dataframe(df[["inicio","Paciente","tipo","notas"]],use_container_width=True,hide_index=True)

# ========== Sessão (Planner) ==========
with tabs[3]:
    st.subheader("Planner de Sessão")
    pts = sb_select("patients","id,nome",order="created_at",desc=True)
    mapa = {p["nome"]:p["id"] for p in pts}
    psel = st.selectbox("Paciente", list(mapa.keys()) or ["—"], key=K("planner","paciente"))

    st.markdown("**Escolha rapidamente componentes da sessão:**")
    colA,colB,colC,colD = st.columns(4)

    freqs = sb_select("frequencies","code,nome,hz,tipo,chakra,cor",order="code")
    opt_freq = [f'{f["code"]} • {f["nome"]}' for f in freqs]
    sel_freq = colA.multiselect("Frequências", opt_freq, key=K("planner","freqs"))

    pres = sb_select("binaural_presets","id,nome,carrier_hz,beat_hz,duracao_min",order="nome")
    mapa_pres = {p["nome"]:p for p in pres}
    sel_bina = colB.selectbox("Preset Binaural", list(mapa_pres.keys()) or ["(opcional)"], key=K("planner","binaural"))

    camas = sb_select("cama_presets","id,nome,etapas,duracao_min",order="nome")
    mapa_cama = {c["nome"]:c for c in camas}
    sel_cama = colC.selectbox("Preset Cama", list(mapa_cama.keys()) or ["(opcional)"], key=K("planner","cama"))

    plans = sb_select("phytotherapy_plans","id,name",order="name")
    mapa_plan = {p["name"]:p for p in plans}
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
    df = pd.DataFrame(sb_select("frequencies","code,nome,hz,tipo,chakra,cor,descricao",order="code"))
    if not df.empty:
        st.dataframe(df,use_container_width=True,hide_index=True)

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
                    "chakra":(chakra or None),"cor":(cor or None),"descricao":desc
                }).execute()
                st.success("Salvo."); st.cache_data.clear()

# ========== Binaural ==========
with tabs[5]:
    st.subheader("Binaural — player rápido")
    c1,c2,c3=st.columns(3)
    carrier = c1.number_input("Carrier (Hz)",50.0,1000.0,220.0,1.0, key=K("binaural","carrier"))
    beat    = c2.number_input("Batida (Hz)",0.5,40.0,10.0,0.5, key=K("binaural","beat"))
    dur     = int(c3.number_input("Duração (s)",10,900,120,5, key=K("binaural","dur")))

    st.markdown("🎵 Música de fundo (opcional)")
    bg_up   = st.file_uploader("MP3/WAV/OGG",type=["mp3","wav","ogg"], key=K("binaural","bg_file"))
    bg_gain = st.slider("Volume do fundo",0.0,0.4,0.12,0.01, key=K("binaural","bg_gain"))
    raw = None
    if bg_up:
        raw = bg_up.read()
        st.audio(raw)  # prévia
    bg_url,_ = bytes_to_data_url(raw, bg_up.name if bg_up else None)

    st.components.v1.html(webaudio_binaural_html(carrier,beat,dur,bg_url,bg_gain), height=260)

    wav = synth_binaural_wav(carrier,beat,20,44100,0.2)
    st.audio(wav, format="audio/wav")
    st.download_button("Baixar WAV (20s)", data=wav,
                       file_name=f"binaural_{int(carrier)}_{beat:.1f}.wav",
                       mime="audio/wav", key=K("binaural","dl_wav"))

# ========== Cama de Cristal ==========
with tabs[6]:
    st.subheader("Cama — presets de 7 luzes")
    camas = sb_select("cama_presets","id,nome,etapas,duracao_min,notas",order="nome")
    nomes = [c["nome"] for c in camas]
    sel = st.selectbox("Preset", nomes or ["—"], key=K("cama","sel"))
    if nomes:
        c = [x for x in camas if x["nome"]==sel][0]
        try:
            etapas = pd.DataFrame(c["etapas"])
            st.dataframe(etapas,use_container_width=True,hide_index=True)
        except Exception:
            st.write(c.get("etapas"))
        st.caption(f"Duração: {c.get('duracao_min','?')} min — {c.get('notas','')}")
    with st.expander("Criar/editar preset"):
        nome = st.text_input("Nome do preset", value="Chakras 7x5", key=K("cama","nome"))
        etapas_json = st.text_area("Etapas (JSON)", value='[{"ordem":1,"chakra":"raiz","cor":"vermelho","min":5}]', key=K("cama","etapas"))
        dur_min = st.number_input("Duração total", 5, 180, 35, key=K("cama","dur"))
        notas = st.text_input("Notas", key=K("cama","notas"))
        if st.button("Salvar preset", key=K("cama","btn_salvar")) and sb:
            sb.table("cama_presets").upsert({
                "nome":nome,"etapas":json.loads(etapas_json),
                "duracao_min":int(dur_min),"notas":notas
            }).execute()
            st.success("Preset salvo."); st.cache_data.clear()

# ========== Fitoterapia ==========
with tabs[7]:
    st.subheader("Planos fitoterápicos")
    df = pd.DataFrame(sb_select("phytotherapy_plans","id,name,objetivo,posologia,duracao_sem,cadencia,notas",order="name"))
    if not df.empty:
        st.dataframe(df,use_container_width=True,hide_index=True)
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
                    "name":name,"objetivo":obj,"posologia":json.loads(pos),
                    "duracao_sem":int(durw),"cadencia":cad,"notas":notas
                }).execute()
                st.success("Plano salvo."); st.cache_data.clear()

# ========== Cristais ==========
with tabs[8]:
    st.subheader("Cristais")
    df = pd.DataFrame(sb_select("crystals","id,name,chakra,color,keywords,benefits,pairing_freq,notes",order="name"))
    if not df.empty:
        st.dataframe(df[["name","chakra","color","keywords"]],use_container_width=True,hide_index=True)
        sel = st.selectbox("Detalhes", df["name"].tolist(), key=K("cristais","sel"))
        row = df[df["name"]==sel].iloc[0]
        st.write("**Benefícios:**", row.get("benefits"))
        st.write("**Combinações de frequência:**", row.get("pairing_freq"))
        st.caption(row.get("notes") or "")
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
                    "color":[c.strip() for c in color.split(",") if c.strip()],
                    "keywords":[k.strip() for k in kw.split(",") if k.strip()],
                    "benefits":[b.strip() for b in bens.splitlines() if b.strip()],
                    "pairing_freq": json.loads(pair),
                    "notes": notas or None
                }).execute()
                st.success("Cristal salvo."); st.cache_data.clear()

# ========== Financeiro ==========
with tabs[9]:
    st.subheader("Financeiro")
    prices = pd.DataFrame(sb_select("prices","id,item,valor_cents,ativo",order="item"))
    if not prices.empty:
        prices["valor"] = prices["valor_cents"]/100
        st.dataframe(prices[["item","valor","ativo"]],use_container_width=True,hide_index=True)

    p_pac = sb_select("patients","id,nome",order="created_at",desc=True)
    mapa = {p["nome"]:p["id"] for p in p_pac}
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
                "obs": obs
            }).execute()
            st.success("Pagamento lançado.")
    pays = pd.DataFrame(sb_select("payments","data,item,valor_cents,metodo,patients(nome)",order="data",desc=True,limit=50))
    if not pays.empty:
        pays["valor"] = pays["valor_cents"]/100
        pays["Paciente"]=pays["patients"].apply(lambda x:(x or {}).get("nome","—"))
        st.dataframe(pays[["data","Paciente","item","valor","metodo"]],use_container_width=True,hide_index=True)

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
