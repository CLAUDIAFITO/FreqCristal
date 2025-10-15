# app.py — MVP do zero: Pacientes, Anamnese, Agenda, Sessão (Planner),
# Frequências, Binaural, Cama de Cristal, Fitoterapia, Cristais, Financeiro, Biblioteca


import os, io, json, wave, base64
from datetime import datetime, timedelta, date
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# --------- ENV / SUPABASE ----------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
try:
    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
except Exception:
    sb = None

st.set_page_config(page_title="Clínica Holística — MVP", layout="wide")

# --------- HELPERS ----------
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
    env = np.linspace(0,1,int(sr*0.02)); env2 = np.linspace(1,0,int(sr*0.02))
    left[:len(env)]*=env; right[:len(env)]*=env; left[-len(env2):]*=env2; right[-len(env2):]*=env2
    stereo = np.vstack([left,right]).T * float(amp)
    y = np.int16(np.clip(stereo,-1,1)*32767)
    buf = io.BytesIO()
    with wave.open(buf,"wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(y.tobytes())
    return buf.getvalue()

def file_to_data_url(up):
    if not up: return None, None
    raw = up.read(); name = up.name.lower()
    mime = "audio/mpeg"
    if name.endswith(".wav"): mime="audio/wav"
    elif name.endswith(".ogg"): mime="audio/ogg"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}", mime

def webaudio_binaural_html(fc: float, beat: float, seconds: int=60, bg_data_url: str|None=None, bg_gain: float=0.12):
    bt = abs(float(beat)); fl = max(20.0, float(fc)-bt/2); fr = float(fc)+bt/2; sec = int(max(5,seconds))
    bg = json.dumps(bg_data_url) if bg_data_url else "null"; g = float(bg_gain)
    return f"""
<div style="padding:.6rem;border:1px solid #eee;border-radius:10px;">
  <b>Binaural</b> — L {fl:.2f} Hz • R {fr:.2f} Hz • {sec}s {'<span style="margin-left:6px;">🎵 fundo</span>' if bg_data_url else ''}<br/>
  <button id="bplay">▶️ Tocar</button> <button id="bstop">⏹️ Parar</button>
  <div style="font-size:.9rem;color:#666">Use fones · volume moderado</div>
</div>
<script>
let ctx=null, l=null, r=null, gL=null, gR=null, mix=null, timer=null, bgS=null, bgG=null;
async function start(){{
  if(ctx) return; ctx=new (window.AudioContext||window.webkitAudioContext)();
  l=ctx.createOscillator(); r=ctx.createOscillator(); l.type='sine'; r.type='sine';
  l.frequency.value={fl:.6f}; r.frequency.value={fr:.6f};
  gL=ctx.createGain(); gR=ctx.createGain(); gL.gain.value=0.05; gR.gain.value=0.05;
  const merger=ctx.createChannelMerger(2);
  l.connect(gL).connect(merger,0,0); r.connect(gR).connect(merger,0,1); merger.connect(ctx.destination);
  l.start(); r.start();
  const bg={bg}; if(bg){{
    try {{
      bgG=ctx.createGain(); bgG.gain.value={g:.4f};
      const res=await fetch(bg); const arr=await res.arrayBuffer(); const buf=await ctx.decodeAudioData(arr);
      bgS=ctx.createBufferSource(); bgS.buffer=buf; bgS.loop=true;
      const split=ctx.createChannelSplitter(2); const mono=ctx.createChannelMerger(2);
      const a=ctx.createGain(); const b=ctx.createGain(); a.gain.value=0.5; b.gain.value=0.5;
      bgS.connect(split); split.connect(a,0); split.connect(b,1); a.connect(mono,0,0); b.connect(mono,0,0);
      mono.connect(bgG).connect(ctx.destination); bgS.start();
    }} catch(e){{ console.warn(e); }}
  }}
  timer=setTimeout(stop,{sec*1000});
}}
function stop(){{
  if(!ctx) return; try{{l.stop();r.stop();}}catch(e){{}}
  if(bgS) try{{bgS.stop();}}catch(e){{}}
  try{{ctx.close();}}catch(e){{}} ctx=null;l=r=gL=gR=mix=bgS=bgG=null; if(timer) clearTimeout(timer);
}}
document.getElementById('bplay').onclick=start; document.getElementById('bstop').onclick=stop;
</script>"""

# --------- UI ---------
st.title("🌿 Clínica Holística — MVP")

tabs = st.tabs(["Pacientes","Anamnese","Agenda","Sessão (Planner)","Frequências","Binaural","Cama de Cristal","Fitoterapia","Cristais","Financeiro","Biblioteca"])

# ========== Pacientes ==========
with tabs[0]:
    st.subheader("Pacientes")
    if not sb: st.warning("Configure SUPABASE_URL/KEY.")
    with st.form("f_pac"):
        c1,c2,c3=st.columns([2,1,1])
        nome=c1.text_input("Nome")
        nasc=c2.date_input("Nascimento", value=None)
        tel=c3.text_input("Telefone")
        email=st.text_input("E-mail")
        notas=st.text_area("Notas")
        if st.form_submit_button("Salvar"):
            sb.table("patients").insert({"nome":nome,"nascimento":str(nasc) if nasc else None,"telefone":tel,"email":email}).execute()
            st.success("Paciente salvo.")
            st.cache_data.clear()
    pts = sb_select("patients","id,nome,nascimento,telefone,email,created_at",order="created_at",desc=True,limit=50)
    if pts: st.dataframe(pd.DataFrame(pts),use_container_width=True,hide_index=True)

# ========== Anamnese ==========
with tabs[1]:
    st.subheader("Anamnese")
    pts = sb_select("patients","id,nome",order="created_at",desc=True)
    mapa = {p["nome"]:p["id"] for p in pts}
    psel = st.selectbox("Paciente", list(mapa.keys()) or ["—"])
    col1,col2=st.columns(2)
    q_sono = col1.slider("Qualidade do sono",0,10,6)
    q_estresse = col2.slider("Nível de estresse",0,10,5)
    chks = st.multiselect("Queixas principais",["Ansiedade","Insônia","Cansaço","Dores","Falta de foco"])
    obs = st.text_area("Observações")
    if st.button("Salvar anamnese"):
        sb.table("anamneses").insert({"patient_id": mapa.get(psel),"respostas":{
            "sono":q_sono,"estresse":q_estresse,"queixas":chks,"obs":obs}}).execute()
        st.success("Anamnese salva.")

# ========== Agenda ==========
with tabs[2]:
    st.subheader("Agenda")
    pts = sb_select("patients","id,nome",order="created_at",desc=True)
    mapa = {p["nome"]:p["id"] for p in pts}
    c1,c2,c3=st.columns(3)
    psel = c1.selectbox("Paciente", list(mapa.keys()) or ["—"], key="ag_p")
    start = c2.date_input("Data",value=date.today())
    hora = c3.time_input("Hora")
    tipo = st.selectbox("Tipo",["Cama","Binaural","Fitoterapia","Misto"])
    notas = st.text_input("Notas")
    if st.button("Agendar"):
        dt = datetime.combine(start, hora)
        sb.table("appointments").insert({"patient_id": mapa.get(psel),"inicio":dt.isoformat(),"tipo":tipo,"notas":notas}).execute()
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
    psel = st.selectbox("Paciente", list(mapa.keys()) or ["—"], key="pl_p")
    st.markdown("**Escolha rapidamente componentes da sessão:**")
    colA,colB,colC,colD = st.columns(4)
    # Frequências de suporte
    freqs = sb_select("frequencies","code,nome,hz,tipo,chakra,cor",order="code")
    opt_freq = [f'{f["code"]} • {f["nome"]}' for f in freqs]
    sel_freq = colA.multiselect("Frequências", opt_freq)
    # Binaural preset
    pres = sb_select("binaural_presets","id,nome,carrier_hz,beat_hz,duracao_min",order="nome")
    mapa_pres = {p["nome"]:p for p in pres}
    sel_bina = colB.selectbox("Preset Binaural", list(mapa_pres.keys()) or ["(opcional)"])
    # Cama preset
    camas = sb_select("cama_presets","id,nome,etapas,duracao_min",order="nome")
    mapa_cama = {c["nome"]:c for c in camas}
    sel_cama = colC.selectbox("Preset Cama", list(mapa_cama.keys()) or ["(opcional)"])
    # Plano fitoterápico
    plans = sb_select("phytotherapy_plans","id,name",order="name")
    mapa_plan = {p["name"]:p for p in plans}
    sel_plan = colD.selectbox("Plano Fitoterápico", list(mapa_plan.keys()) or ["(opcional)"])
    notas = st.text_area("Notas da sessão")
    if st.button("Salvar sessão"):
        prot = {
            "frequencias":[{"code": s.split(" • ")[0]} for s in sel_freq],
            "binaural": mapa_pres.get(sel_bina),
            "cama": mapa_cama.get(sel_cama),
            "fitoterapia_plan": mapa_plan.get(sel_plan),
            "notas": notas
        }
        sb.table("sessions").insert({"patient_id": mapa.get(psel),
                                     "data": datetime.utcnow().isoformat(),
                                     "tipo":"Misto","protocolo":prot,"status":"rascunho"}).execute()
        st.success("Sessão salva!")

# ========== Frequências ==========
with tabs[4]:
    st.subheader("Catálogo de Frequências")
    df = pd.DataFrame(sb_select("frequencies","code,nome,hz,tipo,chakra,cor,descricao",order="code"))
    if not df.empty: st.dataframe(df,use_container_width=True,hide_index=True)
    with st.expander("Adicionar/editar"):
        with st.form("f_freq"):
            code = st.text_input("code (único)", value="SOL528")
            nome = st.text_input("nome", value="Solfeggio 528 Hz")
            hz = st.number_input("hz", 1.0, 2000.0, 528.0, 1.0)
            tipo = st.selectbox("tipo",["solfeggio","chakra","custom"],index=0)
            chakra = st.selectbox("chakra",["","raiz","sacral","plexo","cardiaco","laringeo","terceiro_olho","coronal"],index=0)
            cor = st.text_input("cor")
            desc = st.text_area("descrição")
            if st.form_submit_button("Upsert"):
                sb.table("frequencies").upsert({"code":code,"nome":nome,"hz":hz,"tipo":tipo,"chakra":(chakra or None),"cor":(cor or None),"descricao":desc}).execute()
                st.success("Salvo."); st.cache_data.clear()

# ========== Binaural ==========
with tabs[5]:
    st.subheader("Binaural — player rápido")
    c1,c2,c3=st.columns(3)
    carrier = c1.number_input("Carrier (Hz)",50.0,1000.0,220.0,1.0)
    beat = c2.number_input("Batida (Hz)",0.5,40.0,10.0,0.5)
    dur = int(c3.number_input("Duração (s)",10,900,120,5))
    st.markdown("🎵 Música de fundo (opcional)")
    bg_file = st.file_uploader("MP3/WAV/OGG",type=["mp3","wav","ogg"])
    bg_gain = st.slider("Volume do fundo",0.0,0.4,0.12,0.01)
    bg_url,_ = file_to_data_url(bg_file) if bg_file else (None,None)
    st.components.v1.html(webaudio_binaural_html(carrier,beat,dur,bg_url,bg_gain), height=260)
    wav = synth_binaural_wav(carrier,beat,20,44100,0.2)
    st.audio(wav, format="audio/wav")
    st.download_button("Baixar WAV (20s)", data=wav, file_name=f"binaural_{int(carrier)}_{beat:.1f}.wav", mime="audio/wav")

# ========== Cama de Cristal ==========
with tabs[6]:
    st.subheader("Cama — presets de 7 luzes")
    camas = sb_select("cama_presets","id,nome,etapas,duracao_min,notas",order="nome")
    nomes = [c["nome"] for c in camas]
    sel = st.selectbox("Preset", nomes or ["—"])
    if nomes:
        c = [x for x in camas if x["nome"]==sel][0]
        etapas = pd.DataFrame(c["etapas"])
        st.dataframe(etapas,use_container_width=True,hide_index=True)
        st.caption(f"Duração: {c.get('duracao_min','?')} min — {c.get('notas','')}")
    with st.expander("Criar/editar preset"):
        nome = st.text_input("Nome do preset", value="Chakras 7x5")
        etapas_json = st.text_area("Etapas (JSON)", value='[{"ordem":1,"chakra":"raiz","cor":"vermelho","min":5}]')
        dur_min = st.number_input("Duração total", 5, 180, 35)
        notas = st.text_input("Notas")
        if st.button("Salvar preset"):
            sb.table("cama_presets").upsert({"nome":nome,"etapas":json.loads(etapas_json),"duracao_min":int(dur_min),"notas":notas}).execute()
            st.success("Preset salvo."); st.cache_data.clear()

# ========== Fitoterapia ==========
with tabs[7]:
    st.subheader("Planos fitoterápicos")
    df = pd.DataFrame(sb_select("phytotherapy_plans","id,name,objetivo,posologia,duracao_sem,cadencia,notas",order="name"))
    if not df.empty: st.dataframe(df,use_container_width=True,hide_index=True)
    with st.expander("Novo plano"):
        name = st.text_input("Nome do plano", value="Calma Suave")
        obj = st.text_area("Objetivo")
        pos = st.text_area("Posologia (JSON)", value='[{"erva":"Camomila","forma":"infusão","dose":"200 ml","frequencia":"2x/dia","duracao":"15 dias"}]')
        durw = st.number_input("Duração (semanas)",1,52,3)
        cad = st.text_input("Cadência", value="uso diário")
        if st.button("Salvar plano"):
            sb.table("phytotherapy_plans").upsert({"name":name,"objetivo":obj,"posologia":json.loads(pos),"duracao_sem":int(durw),"cadencia":cad}).execute()
            st.success("Plano salvo."); st.cache_data.clear()

# ========== Cristais ==========
with tabs[8]:
    st.subheader("Cristais")
    df = pd.DataFrame(sb_select("crystals","id,name,chakra,color,keywords,benefits,pairing_freq,notes",order="name"))
    if not df.empty:
        st.dataframe(df[["name","chakra","color","keywords"]],use_container_width=True,hide_index=True)
        sel = st.selectbox("Detalhes", df["name"].tolist())
        row = df[df["name"]==sel].iloc[0]
        st.write("**Benefícios:**", row.get("benefits"))
        st.write("**Combinações de frequência:**", row.get("pairing_freq"))
        st.caption(row.get("notes") or "")
    with st.expander("Adicionar cristal"):
        name = st.text_input("Nome", value="Quartzo Rosa")
        chakra = st.multiselect("Chakras", ["raiz","sacral","plexo","cardiaco","laringeo","terceiro_olho","coronal"])
        color = st.text_input("Cores (separe por vírgula)", value="rosa")
        kw = st.text_input("Palavras-chave (vírgula)", value="acolhimento,autoamor")
        bens = st.text_area("Benefícios (um por linha)", value="Suaviza emoções\nApoia autocuidado")
        pair = st.text_area("Pairing (JSON)", value='{"hz":[528,639],"solfeggio":["SOL528","SOL639"],"binaural":["alpha","theta"]}')
        if st.button("Salvar cristal"):
            sb.table("crystals").upsert({
                "name":name,"chakra":chakra,"color":[c.strip() for c in color.split(",") if c.strip()],
                "keywords":[k.strip() for k in kw.split(",") if k.strip()],
                "benefits":[b.strip() for b in bens.splitlines() if b.strip()],
                "pairing_freq": json.loads(pair)
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
    with st.form("pay"):
        c1,c2,c3 = st.columns(3)
        pac = c1.selectbox("Paciente", list(mapa.keys()) or ["—"])
        item = c2.selectbox("Item", prices["item"].tolist() if not prices.empty else ["Sessão"])
        valor = c3.number_input("Valor (R$)", 0.0, 9999.0, float(prices.loc[prices["item"]==item,"valor"].iloc[0]) if not prices.empty else 150.0, 10.0)
        metodo = st.selectbox("Método", ["PIX","Cartão","Dinheiro"])
        obs = st.text_input("Obs")
        if st.form_submit_button("Registrar pagamento"):
            sb.table("payments").insert({"patient_id": mapa.get(pac),"item":item,"valor_cents":int(round(valor*100)),"metodo":metodo,"obs":obs}).execute()
            st.success("Pagamento lançado.")
    pays = pd.DataFrame(sb_select("payments","data,item,valor_cents,metodo,patients(nome)",order="data",desc=True,limit=50))
    if not pays.empty:
        pays["valor"] = pays["valor_cents"]/100
        pays["Paciente"]=pays["patients"].apply(lambda x:(x or {}).get("nome","—"))
        st.dataframe(pays[["data","Paciente","item","valor","metodo"]],use_container_width=True,hide_index=True)

# ========== Biblioteca (templates prontos) ==========
with tabs[10]:
    st.subheader("Biblioteca de Tratamentos (Templates)")
    tpls = sb_select("therapy_templates","id,name,objetivo,roteiro_binaural,frequencias_suporte,cama_preset,phyto_plan,notas",order="name")
    if tpls:
        nomes=[t["name"] for t in tpls]; mapa={t["name"]:t for t in tpls}
        sel=st.selectbox("Template",nomes)
        t=mapa[sel]
        st.markdown(f"**Objetivo:** {t.get('objetivo','')}")
        st.write("**Frequências de suporte:**", t.get("frequencias_suporte"))
        st.write("**Roteiro binaural:**", t.get("roteiro_binaural"))
        st.write("**Notas:**", t.get("notas"))
        st.caption("Aplique os itens nas abas específicas para tocar/editar.")
