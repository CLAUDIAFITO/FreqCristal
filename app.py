import os
import json
import base64
from datetime import date, timedelta
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import io
import wave

# -------------
# Config
# -------------
st.set_page_config(page_title="claudiafito_v2", layout="wide")

def _get_env_or_secret(key: str) -> Optional[str]:
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key)

DATABASE_URL = _get_env_or_secret("DATABASE_URL")
SUPABASE_URL = _get_env_or_secret("SUPABASE_URL")
SUPABASE_KEY = _get_env_or_secret("SUPABASE_SERVICE_ROLE_KEY") or _get_env_or_secret("SUPABASE_KEY")

BACKEND = "postgres" if DATABASE_URL else ("supabase" if (SUPABASE_URL and SUPABASE_KEY) else "none")
if BACKEND == "none":
    st.error("Defina DATABASE_URL **OU** SUPABASE_URL + SUPABASE_KEY (preferencialmente SUPABASE_SERVICE_ROLE_KEY).")
    st.stop()

# -------------
# DB helpers
# -------------
if BACKEND == "postgres":
    import psycopg2
    import psycopg2.extras

    def get_conn():
        return psycopg2.connect(DATABASE_URL)

    def qall(sql: str, params=None):
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                return cur.fetchall()

    def qone(sql: str, params=None):
        rows = qall(sql, params)
        return rows[0] if rows else None

    def qexec(sql: str, params=None):
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.commit()

else:
    from supabase import create_client, Client
    sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    def sb_select(table: str, columns: str="*", eq: Dict[str, Any]=None, order: Optional[Tuple[str,bool]]=None, limit: Optional[int]=None):
        q = sb.table(table).select(columns)
        if eq:
            for k,v in eq.items():
                q = q.eq(k, v)
        if order:
            col, asc = order
            q = q.order(col, desc=not asc)
        if limit is not None:
            q = q.limit(limit)
        res = q.execute()
        return res.data or []

    def sb_insert(table: str, payload: Dict[str, Any]):
        res = sb.table(table).insert(payload).execute()
        data = res.data or []
        return data[0] if data else None

    def sb_upsert(table: str, payload: Dict[str, Any], on_conflict: Optional[str]=None):
        q = sb.table(table).upsert(payload)
        if on_conflict:
            q = q.on_conflict(on_conflict)
        res = q.execute()
        data = res.data or []
        return data[0] if data else None

# -------------
# Domain model (anamnese simples)
# -------------
DOMAINS = ["sono", "ansiedade", "humor_baixo", "exaustao", "pertencimento", "tensao", "ruminacao"]

QUESTIONS = [
    {"id": "sono_q1", "label": "Dificuldade para pegar no sono", "domain": "sono", "weight": 1.0},
    {"id": "sono_q2", "label": "Acorda no meio da noite / sono leve", "domain": "sono", "weight": 1.0},
    {"id": "ans_q1", "label": "Ansiedade / agitação no dia a dia", "domain": "ansiedade", "weight": 1.2},
    {"id": "ans_q2", "label": "Sintomas físicos de ansiedade (aperto, inquietação)", "domain": "ansiedade", "weight": 1.0},
    {"id": "hum_q1", "label": "Tristeza / desânimo frequente", "domain": "humor_baixo", "weight": 1.2},
    {"id": "hum_q2", "label": "Perda de prazer / motivação", "domain": "humor_baixo", "weight": 1.0},
    {"id": "exa_q1", "label": "Cansaço / exaustão por responsabilidades", "domain": "exaustao", "weight": 1.2},
    {"id": "exa_q2", "label": "Pouco tempo para si / autocuidado", "domain": "exaustao", "weight": 1.0},
    {"id": "per_q1", "label": "Sensação de não pertencimento / desconexão", "domain": "pertencimento", "weight": 1.2},
    {"id": "per_q2", "label": "Vergonha / autojulgamento", "domain": "pertencimento", "weight": 1.0},
    {"id": "ten_q1", "label": "Tensão muscular / dores recorrentes", "domain": "tensao", "weight": 1.0},
    {"id": "ten_q2", "label": "Mandíbula/ombros travados / corpo em alerta", "domain": "tensao", "weight": 1.0},
    {"id": "rum_q1", "label": "Mente acelerada / ruminação", "domain": "ruminacao", "weight": 1.2},
    {"id": "rum_q2", "label": "Dificuldade de foco por pensamentos repetitivos", "domain": "ruminacao", "weight": 1.0},
]

FLAGS = [
    {"id": "flag_preg", "label": "Gestação / amamentação"},
    {"id": "flag_meds", "label": "Uso de medicamentos (ansiolíticos/antidepressivos/sedativos)"},
    {"id": "flag_allergy", "label": "Alergias / sensibilidades"},
    {"id": "flag_sound", "label": "Sensibilidade a som (binaural)"},
    {"id": "flag_light", "label": "Sensibilidade à luz (cama de cristal)"},
]

DOMAIN_TO_PROTOCOL = {
    "ansiedade": "FOCO – Ansiedade / Agitação",
    "sono": "FOCO – Sono Profundo",
    "exaustao": "FOCO – Exaustão / Sobrecarga",
    "pertencimento": "FOCO – Pertencimento / Vergonha",
    "humor_baixo": "FOCO – Exaustão / Sobrecarga",
    "tensao": "FOCO – Ansiedade / Agitação",
    "ruminacao": "FOCO – Ansiedade / Agitação",
}
BASE_PROTOCOL = "BASE – Aterramento + Regulação"

def compute_scores(answers: Dict[str, int]) -> Dict[str, float]:
    sums = {d: 0.0 for d in DOMAINS}
    maxs = {d: 0.0 for d in DOMAINS}
    for q in QUESTIONS:
        v = float(answers.get(q["id"], 0))
        w = float(q["weight"])
        d = q["domain"]
        sums[d] += v * w
        maxs[d] += 4.0 * w
    return {d: round((sums[d] / maxs[d] * 100.0) if maxs[d] else 0.0, 1) for d in DOMAINS}

def pick_focus(scores: Dict[str, float], top_n=3) -> List[Tuple[str, float]]:
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

def sessions_from_scores(scores: Dict[str, float]) -> Tuple[int, int]:
    top = sorted(scores.values(), reverse=True)
    max_score = top[0] if top else 0.0
    strong = sum(1 for s in top[:3] if s >= 70)
    if strong <= 0:
        strong = sum(1 for s in top[:3] if s >= 60)
    qty = 4 if strong <= 1 else (6 if strong == 2 else 8)
    cadence = 7 if max_score >= 80 else (10 if max_score >= 60 else 14)
    return qty, cadence

def load_protocols() -> Dict[str, Dict[str, Any]]:
    if BACKEND == "postgres":
        rows = qall("select name, domain, rules_json, content_json from public.protocol_library where active = true")
        return {r["name"]: {"name": r["name"], "domain": r["domain"], "rules": r["rules_json"], "content": r["content_json"]} for r in rows}
    rows = sb_select("protocol_library", columns="name,domain,rules_json,content_json,active", eq={"active": True}, order=("domain", True))
    out = {}
    for r in rows:
        out[r["name"]] = {"name": r["name"], "domain": r["domain"], "rules": r.get("rules_json") or {}, "content": r.get("content_json") or {}}
    return out

def load_binaural_presets() -> List[Dict[str, Any]]:
    if BACKEND == "postgres":
        return qall("select id, nome, carrier_hz, beat_hz, duracao_min, notas from public.binaural_presets order by nome")
    return sb_select("binaural_presets", columns="id,nome,carrier_hz,beat_hz,duracao_min,notas", order=("nome", True))

def load_frequencies(tipo: Optional[str]=None) -> List[Dict[str, Any]]:
    if BACKEND == "postgres":
        if tipo:
            return qall("select code, nome, hz, tipo, chakra, cor, descricao from public.frequencies where tipo=%s order by code", (tipo,))
        return qall("select code, nome, hz, tipo, chakra, cor, descricao from public.frequencies order by code")
    if tipo:
        return sb_select("frequencies", columns="code,nome,hz,tipo,chakra,cor,descricao", eq={"tipo": tipo}, order=("code", True))
    return sb_select("frequencies", columns="code,nome,hz,tipo,chakra,cor,descricao", order=("code", True))

def select_protocols(scores: Dict[str, float], protocols: Dict[str, Dict[str, Any]]) -> List[str]:
    selected = [BASE_PROTOCOL] if BASE_PROTOCOL in protocols else []
    for dom, sc in scores.items():
        if sc >= 60:
            pname = DOMAIN_TO_PROTOCOL.get(dom)
            if pname and pname in protocols and pname not in selected:
                selected.append(pname)
    if BASE_PROTOCOL in protocols and BASE_PROTOCOL not in selected:
        selected.insert(0, BASE_PROTOCOL)
    return selected

def merge_plan(selected_names: List[str], protocols: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    chakras, emocoes, cristais, fito, alerts = [], [], [], [], []
    def add_unique(lst, item):
        if item not in lst:
            lst.append(item)
    for name in selected_names:
        c = protocols.get(name, {}).get("content", {}) or {}
        for ch in c.get("chakras_foco", []): add_unique(chakras, ch)
        for e in c.get("emocoes_foco", []): add_unique(emocoes, e)
        for cr in c.get("cristais", []):
            if cr not in cristais: cristais.append(cr)
        for f in c.get("fito", []):
            if f not in fito: fito.append(f)
        if c.get("alertas"): add_unique(alerts, c["alertas"])
    return {"chakras_prioritarios": chakras,"emocoes_prioritarias": emocoes,"cristais_sugeridos": cristais,"fito_sugerida": fito,"alertas": alerts}

def build_session_scripts(qty: int, cadence_days: int, focus: List[Tuple[str, float]],
                          selected_names: List[str], protocols: Dict[str, Dict[str, Any]],
                          audio_block: Dict[str, Any], extra_freq_codes: List[str]) -> List[Dict[str, Any]]:
    focus_domains = [d for d,_ in focus]
    focus_cards = []
    for dom in focus_domains:
        pname = DOMAIN_TO_PROTOCOL.get(dom)
        if pname and pname in selected_names and pname != BASE_PROTOCOL:
            focus_cards.append(pname)
    if not focus_cards:
        focus_cards = [n for n in selected_names if n != BASE_PROTOCOL][:1]

    scripts = []
    today = date.today()
    for i in range(1, qty+1):
        session_date = today + timedelta(days=cadence_days*(i-1))
        focus_card = focus_cards[(i-1) % len(focus_cards)] if focus_cards else None
        parts = []
        base = protocols.get(BASE_PROTOCOL, {}).get("content", {}) or {}
        if base:
            parts.append({"card": BASE_PROTOCOL, "binaural": base.get("binaural"), "cama": base.get("cama_cristal"),
                          "cristais": base.get("cristais"), "fito": base.get("fito"),
                          "roteiro": base.get("roteiro_sessao")})

        if focus_card:
            fc = protocols.get(focus_card, {}).get("content", {}) or {}
            parts.append({"card": focus_card, "binaural": fc.get("binaural"), "cama": fc.get("cama_cristal"),
                          "cristais": fc.get("cristais"), "fito": fc.get("fito"),
                          "roteiro": fc.get("roteiro_sessao")})

        scripts.append({
            "session_n": i,
            "scheduled_date": str(session_date),
            "status": "AGENDADA",
            "audio": audio_block,
            "frequencias": [{"code": c} for c in extra_freq_codes],
            "parts": parts
        })
    return scripts

# -------------------------
# WAV binaural (preview/download)
# -------------------------
def synth_binaural_wav(carrier_hz: float, beat_hz: float, seconds: int = 20, sr: int = 44100, amp: float = 0.2) -> bytes:
    t = np.linspace(0, seconds, int(sr*seconds), endpoint=False)
    l = np.sin(2*np.pi*(carrier_hz - beat_hz/2.0)*t)
    r = np.sin(2*np.pi*(carrier_hz + beat_hz/2.0)*t)
    stereo = np.stack([l, r], axis=1) * float(amp)
    stereo_i16 = np.int16(np.clip(stereo, -1, 1) * 32767)

    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(stereo_i16.tobytes())
    return bio.getvalue()

# -------------------------
# CRUD: patients/intakes/pla

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

ns/sessions_nova
# -------------------------
def list_patients():
    if BACKEND == "postgres":
        return qall("select id, nome, nascimento from public.patients order by nome asc")
    return sb_select("patients", columns="id,nome,nascimento", order=("nome", True))

def insert_patient(nome: str, telefone: str, email: str, nascimento, notas: str) -> str:
    payload = {"nome": nome, "telefone": telefone or None, "email": email or None, "nascimento": str(nascimento) if nascimento else None, "notas": notas or None}
    if BACKEND == "postgres":
        row = qone("""insert into public.patients (nome, telefone, email, nascimento, notas)
                      values (%s,%s,%s,%s,%s) returning id""",
                   (nome, telefone or None, email or None, nascimento, notas or None))
        return row["id"]
    row = sb_insert("patients", payload)
    return row["id"]

def insert_intake(patient_id: str, complaint: str, answers: Dict[str, int], scores: Dict[str, float],
                 flags: Dict[str, bool], notes: str) -> str:
    payload = {
        "patient_id": patient_id,
        "complaint": complaint or None,
        "answers_json": answers,
        "scores_json": scores,
        "flags_json": flags,
        "notes": notes or None,
    }
    if BACKEND == "postgres":
        row = qone("""insert into public.intakes (patient_id, complaint, answers_json, scores_json, flags_json, notes)
                      values (%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s)
                      returning id""",
                   (patient_id, complaint, json.dumps(answers, ensure_ascii=False),
                    json.dumps(scores, ensure_ascii=False),
                    json.dumps(flags, ensure_ascii=False),
                    notes))
        return row["id"]
    row = sb_insert("intakes", payload)
    return row["id"]

def insert_plan(patient_id: str, intake_id: str, focus: List[Tuple[str, float]], selected_names: List[str],
                sessions_qty: int, cadence_days: int, plan_json: Dict[str, Any]) -> str:
    payload = {
        "intake_id": intake_id,
        "patient_id": patient_id,
        "focus_json": {"top": focus},
        "selected_protocols": selected_names,
        "sessions_qty": sessions_qty,
        "cadence_days": cadence_days,
        "plan_json": plan_json,
    }
    if BACKEND == "postgres":
        row = qone("""insert into public.plans (intake_id, patient_id, focus_json, selected_protocols, sessions_qty, cadence_days, plan_json)
                      values (%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s::jsonb)
                      returning id""",
                   (intake_id, patient_id,
                    json.dumps({"top": focus}, ensure_ascii=False),
                    json.dumps(selected_names, ensure_ascii=False),
                    sessions_qty, cadence_days,
                    json.dumps(plan_json, ensure_ascii=False)))
        return row["id"]
    row = sb_insert("plans", payload)
    return row["id"]

def insert_session_nova(plan_id: str, patient_id: str, session_n: int, scheduled_date_str: str, status: str, script: Dict[str, Any]):
    payload = {
        "plan_id": plan_id,
        "patient_id": patient_id,
        "session_n": session_n,
        "scheduled_date": scheduled_date_str,
        "status": status,
        "script_json": script,
    }
    if BACKEND == "postgres":
        qexec("""insert into public.sessions_nova (plan_id, patient_id, session_n, scheduled_date, status, script_json)
                 values (%s,%s,%s,%s,%s,%s::jsonb)""",
              (plan_id, patient_id, session_n, scheduled_date_str, status, json.dumps(script, ensure_ascii=False)))
    else:
        sb_insert("sessions_nova", payload)

# -------------------------
# UI
# -------------------------
st.title("claudiafito_v2 — Atendimento + Binaural (como no app antigo)")
st.caption("Inclui presets Gamma/Theta/Alpha/Delta, Solfeggio, Chakras e upload de música de fundo do computador.")

tabs = st.tabs(["Atendimento", "Binaural"])

# Shared audio settings (igual o antigo)
st.session_state.setdefault("binaural_carrier", 220.0)
st.session_state.setdefault("binaural_beat", 10.0)
st.session_state.setdefault("binaural_dur_min", 15)
st.session_state.setdefault("binaural_bg_gain", 0.12)
st.session_state.setdefault("extra_freq_codes", [])

with tabs[1]:
    st.subheader("Binaural — player rápido")

    band_map = {
        "Delta (1–4 Hz)": 3.0,
        "Theta (4–8 Hz)": 6.0,
        "Alpha (8–12 Hz)": 10.0,
        "Beta baixa (12–18 Hz)": 15.0,
        "Gamma (30–45 Hz)": 40.0,
    }
    bcol1, bcol2 = st.columns([2,1])
    faixa = bcol1.selectbox("Faixa de ondas (atalho)", list(band_map.keys()), index=2)
    if bcol2.button("Aplicar faixa"):
        st.session_state["binaural_beat"] = float(band_map[faixa])
        st.success(f"Batida ajustada para {band_map[faixa]} Hz")

    presets = load_binaural_presets()
    mapa_pres = {p["nome"]: p for p in (presets or [])}
    cols_top = st.columns([2,1])
    preset_escolhido = cols_top[0].selectbox("Tratamento pré-definido (binaural_presets)", list(mapa_pres.keys()) or ["(nenhum)"])
    if cols_top[1].button("Aplicar preset") and preset_escolhido in mapa_pres:
        p = mapa_pres[preset_escolhido]
        st.session_state["binaural_carrier"] = float(p.get("carrier_hz") or 220.0)
        st.session_state["binaural_beat"] = float(p.get("beat_hz") or 10.0)
        st.session_state["binaural_dur_min"] = int(p.get("duracao_min") or 15)
        st.success("Preset aplicado.")

    c1, c2, c3 = st.columns(3)
    carrier = c1.number_input("Carrier (Hz)", 80.0, 600.0, float(st.session_state["binaural_carrier"]), 1.0)
    beat = c2.number_input("Beat (Hz)", 0.5, 60.0, float(st.session_state["binaural_beat"]), 0.5)
    dur = c3.number_input("Duração (min)", 1, 60, int(st.session_state["binaural_dur_min"]), 1)

    st.session_state["binaural_carrier"] = carrier
    st.session_state["binaural_beat"] = beat
    st.session_state["binaural_dur_min"] = dur

    l_hz = carrier - beat/2.0
    r_hz = carrier + beat/2.0
    st.info(f"Cálculo: L = {l_hz:.2f} Hz e R = {r_hz:.2f} Hz (beat ~ {beat:.2f} Hz)")

    st.markdown("🎵 Música de fundo (opcional) — do seu computador (como antes)")
    bg_up = st.file_uploader("MP3/WAV/OGG (até 12MB)", type=["mp3","wav","ogg"])
    bg_gain = st.slider("Volume do fundo", 0.0, 0.4, float(st.session_state["binaural_bg_gain"]), 0.01)
    st.session_state["binaural_bg_gain"] = bg_gain

    bg_raw = None
    bg_name = None
    if bg_up:
        bg_raw = bg_up.read()
        bg_name = bg_up.name
        st.audio(bg_raw)
        # Player com botões Tocar/Parar (igual ao app antigo)
        bg_data_url = None
        try:
            mime = getattr(bg_up, "type", None) or "audio/mpeg"
            bg_data_url = f"data:{mime};base64,{base64.b64encode(bg_raw).decode('utf-8')}"
        except Exception:
            bg_data_url = None

    # Renderiza o player WebAudio (binaural + fundo)
    carrier_hz = float(st.session_state.get("binaural_carrier", 220.0))
    beat_hz = float(st.session_state.get("binaural_beat", 10.0))
    dur_min = int(st.session_state.get("binaural_dur_min", 15) or 15)
    seconds = int(max(10, min(120, dur_min * 60)))  # limita para não ficar pesado no navegador

    st.markdown("▶️ **Player (Tocar/Parar)** — binaural + fundo")
    components.html(
        webaudio_binaural_html(carrier_hz, beat_hz, seconds=seconds, bg_data_url=bg_data_url, bg_gain=float(bg_gain)),
        height=110,
    )

    st.markdown("🔔 Frequências auxiliares (Solfeggio + Chakras)")
    sol = load_frequencies("solfeggio")
    chak = load_frequencies("chakra")
    sol_opts = [f'{r["code"]} — {r.get("nome") or ""}'.strip() for r in sol]
    chak_opts = [f'{r["code"]} — {r.get("nome") or ""}'.strip() for r in chak]
    sol_map = {sol_opts[i]: sol[i]["code"] for i in range(len(sol_opts))}
    chak_map = {chak_opts[i]: chak[i]["code"] for i in range(len(chak_opts))}
    c4, c5 = st.columns(2)
    sel_sol = c4.multiselect("Solfeggio", sol_opts, default=[])
    sel_chak = c5.multiselect("Chakras", chak_opts, default=[])
    extra_codes = [sol_map[x] for x in sel_sol] + [chak_map[x] for x in sel_chak]
    custom_code = st.text_input("Custom code (opcional)", value="")
    if custom_code.strip():
        extra_codes.append(custom_code.strip().upper())

    # dedupe preserve order
    seen=set(); extra_codes=[c for c in extra_codes if not (c in seen or seen.add(c))]
    st.session_state["extra_freq_codes"] = extra_codes

    wav = synth_binaural_wav(float(carrier), float(beat), seconds=min(int(dur*60), 20), sr=44100, amp=0.2)
    st.audio(wav, format="audio/wav")
    st.download_button("Baixar WAV (até 20s)", data=wav,
                       file_name=f"binaural_{int(carrier)}_{beat:.1f}.wav",
                       mime="audio/wav")

    with st.expander("Sugestões rápidas por objetivo"):
        st.markdown(
            """
- **Relaxar/ansiedade** → **Theta 5–6 Hz** (15–20 min) e fechar em **Alpha 10 Hz** (5–10 min).  
- **Sono** → **Delta 2–3 Hz** (10–20 min) → **Theta 5–6 Hz** (10–15 min).  
- **Foco calmo** → **Alpha 10 Hz** (10–15 min).  
- **Gamma 40 Hz** → estimulação breve (5–12 min), volume baixo.  
            """
        )

with tabs[0]:
    st.subheader("Atendimento (gera plano + sessões_nova)")

    with st.sidebar:
        st.header("Paciente")
        patients = list_patients()
        # label includes nascimento to avoid duplicates
        def lab(p):
            nasc = p.get("nascimento")
            tail = str(p["id"])[-4:]
            return f'{p["nome"]} — {nasc or "s/n"} — {tail}'
        labels = ["— Novo paciente —"] + [lab(p) for p in patients]
        sel = st.selectbox("Selecionar", labels, index=0)

        if sel == "— Novo paciente —":
            nome = st.text_input("Nome")
            telefone = st.text_input("Telefone (opcional)")
            email = st.text_input("E-mail (opcional)")
            nascimento = st.date_input("Nascimento (opcional)", value=None)
            pnotas = st.text_area("Notas (opcional)")
            if st.button("Criar paciente", type="primary", use_container_width=True):
                if not nome.strip():
                    st.warning("Informe o nome.")
                else:
                    st.session_state["patient_id"] = insert_patient(nome.strip(), telefone.strip(), email.strip(), nascimento, pnotas.strip())
                    st.success("Paciente criado!")
                    st.rerun()
        else:
            # map label -> id
            idx = labels.index(sel) - 1
            st.session_state["patient_id"] = patients[idx]["id"]

    patient_id = st.session_state.get("patient_id")
    if not patient_id:
        st.stop()

    col1, col2 = st.columns([2,1])
    with col1:
        complaint = st.text_input("Queixa principal (curta)")
    with col2:
        atend_date = st.date_input("Data", value=date.today())

    st.markdown("**Anamnese (0–4)**")
    answers = {}
    cols = st.columns(2)
    for i, q in enumerate(QUESTIONS):
        with cols[i % 2]:
            answers[q["id"]] = st.slider(q["label"], 0, 4, 0, key=f'att_{q["id"]}')

    st.markdown("**Sinais de atenção**")
    flags = {}
    fcols = st.columns(2)
    for i, f in enumerate(FLAGS):
        with fcols[i % 2]:
            flags[f["id"]] = st.checkbox(f["label"], value=False, key=f'att_{f["id"]}')

    notes = st.text_area("Notas do terapeuta (opcional)", height=100)

    scores = compute_scores(answers)
    focus = pick_focus(scores, top_n=3)
    qty, cadence = sessions_from_scores(scores)

    protocols = load_protocols()
    selected_names = select_protocols(scores, protocols)
    plan = merge_plan(selected_names, protocols)

    audio_block = {
        "binaural": {
            "carrier_hz": float(st.session_state["binaural_carrier"]),
            "beat_hz": float(st.session_state["binaural_beat"]),
            "duracao_min": int(st.session_state["binaural_dur_min"]),
        },
        "bg": {
            "filename": None,  # arquivo local não persiste no banco
            "gain": float(st.session_state["binaural_bg_gain"]),
            "note": "música de fundo é selecionada no computador (não é salva no banco)."
        }
    }
    extra_freq_codes = st.session_state.get("extra_freq_codes") or []

    scripts = build_session_scripts(qty, cadence, focus, selected_names, protocols, audio_block, extra_freq_codes)

    st.divider()
    left, right = st.columns(2)
    with left:
        df = pd.DataFrame([{"dominio": k, "score": v} for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.write("Foco:", focus)
        st.write("Sessões sugeridas:", {"qty": qty, "cadence_days": cadence})
        st.write("Frequências extras:", extra_freq_codes)
    with right:
        st.write("Protocolos:", selected_names)
        st.write("Plano:", plan)
        st.write("Áudio (binaural):", audio_block["binaural"])

    st.subheader("Sessões pré-definidas")
    st.dataframe(pd.DataFrame([{"sessao": s["session_n"], "data": s["scheduled_date"]} for s in scripts]),
                 use_container_width=True, hide_index=True)

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Salvar anamnese (intake)", use_container_width=True):
            intake_id = insert_intake(patient_id, complaint, answers, scores, flags, notes)
            st.session_state["last_intake_id"] = intake_id
            st.success("Anamnese salva!")
    with b2:
        if st.button("Gerar plano + criar sessões (sessions_nova)", type="primary", use_container_width=True):
            intake_id = st.session_state.get("last_intake_id")
            if not intake_id:
                intake_id = insert_intake(patient_id, complaint, answers, scores, flags, notes)
                st.session_state["last_intake_id"] = intake_id

            plan_id = insert_plan(
                patient_id=patient_id,
                intake_id=intake_id,
                focus=focus,
                selected_names=selected_names,
                sessions_qty=qty,
                cadence_days=cadence,
                plan_json={"date": str(atend_date), "complaint": complaint, "scores": scores,
                           "focus": focus, "selected_protocols": selected_names,
                           "plan": plan, "audio": audio_block, "frequencias": [{"code": c} for c in extra_freq_codes]},
            )
            for s in scripts:
                insert_session_nova(plan_id, patient_id, int(s["session_n"]), s["scheduled_date"], s["status"], s)

            st.success(f"Plano criado e sessões geradas em sessions_nova! plan_id={plan_id}")
