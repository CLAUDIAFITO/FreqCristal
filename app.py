import os
import io
import json
import wave
import base64
from datetime import date, timedelta
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Optional backends:
# 1) DATABASE_URL -> psycopg2 (Postgres direto)
# 2) SUPABASE_URL + SUPABASE_KEY (ou SUPABASE_SERVICE_ROLE_KEY) -> Supabase REST (supabase-py)

st.set_page_config(page_title="claudiafito_v2", layout="wide")

# -------------------------
# Keys únicas p/ widgets
# -------------------------
def K(*parts: str) -> str:
    """Gera uma key única estável para widgets: K('aba','secao','campo')."""
    return "k_" + "_".join(str(p).strip().lower().replace(" ", "_") for p in parts if p)

# -------------------------
# Áudio: helpers (binaural + fundo)
# -------------------------

# Limite p/ data-URL (evita erro de tamanho no Streamlit)
MAX_BG_MB = 12  # ~12MB (vira ~16MB base64)

def synth_binaural_wav(fc: float, beat: float, seconds: float = 20.0, sr: int = 44100, amp: float = 0.2) -> bytes:
    """Gera um WAV estéreo com binaural (L/R) para download/preview rápido."""
    bt = abs(float(beat))
    fl = max(1.0, float(fc) - bt / 2)
    fr = float(fc) + bt / 2
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    left = np.sin(2 * np.pi * fl * t)
    right = np.sin(2 * np.pi * fr * t)
    ramp = int(sr * 0.02)  # fade 20ms
    if ramp > 0:
        left[:ramp] *= np.linspace(0, 1, ramp)
        right[:ramp] *= np.linspace(0, 1, ramp)
        left[-ramp:] *= np.linspace(1, 0, ramp)
        right[-ramp:] *= np.linspace(1, 0, ramp)
    stereo = np.vstack([left, right]).T * float(amp)
    y = np.int16(np.clip(stereo, -1, 1) * 32767)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(y.tobytes())
    return buf.getvalue()

def bytes_to_data_url_safe(raw: bytes, filename: Optional[str], max_mb: int = MAX_BG_MB):
    """Converte bytes em data URL, mas recusa arquivos grandes para evitar erro de tamanho."""
    if not raw:
        return None, None, None
    size_mb = len(raw) / (1024 * 1024)
    name = (filename or "").lower()
    mime = "audio/mpeg"
    if name.endswith(".wav"):
        mime = "audio/wav"
    elif name.endswith(".ogg") or name.endswith(".oga"):
        mime = "audio/ogg"

    if size_mb > max_mb:
        return None, mime, f"Arquivo de {size_mb:.1f} MB excede o limite de {max_mb} MB para tocar embutido. Use arquivo menor ou uma URL."
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}", mime, None

def webaudio_binaural_html(
    fc: float,
    beat: float,
    seconds: int = 60,
    bg_url: Optional[str] = None,
    bg_gain: float = 0.12,
    binaural_gain: float = 0.05,
):
    """Player binaural + música de fundo (opcional) via WebAudio + <audio>."""
    bt = abs(float(beat))
    fl = max(20.0, float(fc) - bt / 2)
    fr = float(fc) + bt / 2
    sec = int(max(5, seconds))
    bg = json.dumps(bg_url) if bg_url else "null"
    g_bg = float(bg_gain)
    g_bin = float(max(0.0, min(0.2, binaural_gain)))

    return f"""
<div style="padding:.6rem;border:1px solid #eee;border-radius:10px;">
  <b>Binaural</b> — L {fl:.2f} Hz • R {fr:.2f} Hz • {sec}s {'<span style="margin-left:6px;">🎵 fundo</span>' if bg_url else ''}<br/>
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
  gL.gain.value = {g_bin:.4f}; gR.gain.value = {g_bin:.4f};
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
      bgGain = ctx.createGain(); bgGain.gain.value = {g_bg:.4f};

      // Forçar MONO (para não competir com o L/R do binaural)
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

function stop(){{ cleanup(); }}

document.getElementById('bplay').onclick = start;
document.getElementById('bstop').onclick  = stop;
</script>
"""

# -------------------------
# Config / Backend selection
# -------------------------
def _get_env_or_secret(key: str) -> Optional[str]:
    # Streamlit secrets wins; fallback env
    val = None
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            val = st.secrets[key]
    except Exception:
        pass
    return val or os.getenv(key)

# Opcional (igual ao seu app antigo): se você preferir "hardcode" no arquivo,
# preencha aqui (ou deixe vazio e use st.secrets/env no deploy).
SUPABASE_URL_FALLBACK = ""
SUPABASE_KEY_FALLBACK = ""

DATABASE_URL = _get_env_or_secret("DATABASE_URL")
SUPABASE_URL = _get_env_or_secret("SUPABASE_URL") or (SUPABASE_URL_FALLBACK or None)
SUPABASE_KEY = (
    _get_env_or_secret("SUPABASE_SERVICE_ROLE_KEY")
    or _get_env_or_secret("SUPABASE_KEY")
    or (SUPABASE_KEY_FALLBACK or None)
)

BACKEND = "postgres" if DATABASE_URL else ("supabase" if (SUPABASE_URL and SUPABASE_KEY) else "none")

if BACKEND == "none":
    st.error(
        "Configuração faltando.\n\n"
        "Defina **DATABASE_URL** (Postgres direto) **OU** defina **SUPABASE_URL** + **SUPABASE_KEY** "
        "(preferencialmente usando SUPABASE_SERVICE_ROLE_KEY no lugar da anon key)."
    )
    st.stop()

# -------------------------
# Postgres backend (psycopg2)
# -------------------------
if BACKEND == "postgres":
    import psycopg2
    import psycopg2.extras

    def get_conn():
        return psycopg2.connect(DATABASE_URL)

    def qexec(sql: str, params=None):
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.commit()

    def qall(sql: str, params=None):
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                return cur.fetchall()

    def qone(sql: str, params=None):
        rows = qall(sql, params)
        return rows[0] if rows else None

# -------------------------
# Supabase backend (REST via supabase-py)
# -------------------------
else:
    from supabase import create_client, Client

    sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    def sb_select(table: str, columns: str = "*", eq: Dict[str, Any] = None,
                  order: Optional[Tuple[str, bool]] = None, limit: Optional[int] = None):
        q = sb.table(table).select(columns)
        if eq:
            for k, v in eq.items():
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
        # supabase returns inserted rows
        data = res.data or []
        return data[0] if data else None

# -------------------------
# Anamnese (0–4)
# -------------------------
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
    else:
        rows = sb_select("protocol_library", columns="name,domain,rules_json,content_json,active", eq={"active": True}, order=("domain", True))
        out = {}
        for r in rows:
            out[r["name"]] = {"name": r["name"], "domain": r["domain"], "rules": r.get("rules_json") or {}, "content": r.get("content_json") or {}}
        return out

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
        for ch in c.get("chakras_foco", []):
            add_unique(chakras, ch)
        for e in c.get("emocoes_foco", []):
            add_unique(emocoes, e)
        for cr in c.get("cristais", []):
            if cr not in cristais:
                cristais.append(cr)
        for f in c.get("fito", []):
            if f not in fito:
                fito.append(f)
        if c.get("alertas"):
            add_unique(alerts, c["alertas"])

    return {
        "chakras_prioritarios": chakras,
        "emocoes_prioritarias": emocoes,
        "cristais_sugeridos": cristais,
        "fito_sugerida": fito,
        "alertas": alerts,
    }

def build_session_scripts(qty: int, cadence_days: int, focus: List[Tuple[str, float]],
                          selected_names: List[str], protocols: Dict[str, Dict[str, Any]],
                          audio_cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    focus_domains = [d for d, _ in focus]
    focus_cards = []
    for dom in focus_domains:
        pname = DOMAIN_TO_PROTOCOL.get(dom)
        if pname and pname in selected_names and pname != BASE_PROTOCOL:
            focus_cards.append(pname)
    if not focus_cards:
        focus_cards = [n for n in selected_names if n != BASE_PROTOCOL][:1]

    scripts = []
    today = date.today()
    for i in range(1, qty + 1):
        session_date = today + timedelta(days=cadence_days * (i - 1))
        focus_card = focus_cards[(i - 1) % len(focus_cards)] if focus_cards else None
        parts = []

        base = protocols.get(BASE_PROTOCOL, {}).get("content", {}) or {}
        if base:
            parts.append({
                "card": BASE_PROTOCOL,
                "binaural": base.get("binaural"),
                "cama": base.get("cama_cristal"),
                "cristais": base.get("cristais"),
                "fito": base.get("fito"),
                "roteiro": base.get("roteiro_sessao"),
            })

        if focus_card:
            fc = protocols.get(focus_card, {}).get("content", {}) or {}
            parts.append({
                "card": focus_card,
                "binaural": fc.get("binaural"),
                "cama": fc.get("cama_cristal"),
                "cristais": fc.get("cristais"),
                "fito": fc.get("fito"),
                "roteiro": fc.get("roteiro_sessao"),
            })

        scripts.append({
            "session_n": i,
            "scheduled_date": str(session_date),
            "status": "AGENDADA",
            "parts": parts,
        })
    return scripts

# -------------------------
# CRUD (patients existente + sessions_nova)
# -------------------------

# -------------------------
# CRUD: Áudio (dispositivos + som de fundo)
# -------------------------
def list_audio_devices(kind: str):
    if BACKEND == "postgres":
        return qall("select id, name, kind, connection from public.audio_devices where active=true and kind=%s order by name asc", (kind,))
    else:
        return sb_select("audio_devices", columns="id,name,kind,connection,active", eq={"active": True, "kind": kind}, order=("name", True))

def list_background_tracks():
    if BACKEND == "postgres":
        return qall("select id, name, category, url from public.background_tracks where active=true order by name asc")
    else:
        return sb_select("background_tracks", columns="id,name,category,url,active", eq={"active": True}, order=("name", True))

def insert_audio_device(name: str, kind: str, connection: str, notes: str = ""):
    payload = {"name": name, "kind": kind, "connection": connection or None, "notes": notes or None, "active": True}
    if BACKEND == "postgres":
        row = qone("""insert into public.audio_devices (name, kind, connection, notes, active)
                      values (%s,%s,%s,%s,true)
                      on conflict (name) do update set kind=excluded.kind, connection=excluded.connection, notes=excluded.notes, active=true
                      returning id""",
                   (name, kind, connection or None, notes or None))
        return row["id"]
    else:
        # supabase REST: upsert by name is not trivial without RPC; we'll try insert and let it error if duplicate
        row = sb_insert("audio_devices", payload)
        return row["id"]

def insert_background_track(name: str, category: str, url: str, notes: str = ""):
    payload = {"name": name, "category": category or None, "url": url or None, "notes": notes or None, "active": True}
    if BACKEND == "postgres":
        row = qone("""insert into public.background_tracks (name, category, url, notes, active)
                      values (%s,%s,%s,%s,true)
                      on conflict (name) do update set category=excluded.category, url=excluded.url, notes=excluded.notes, active=true
                      returning id""",
                   (name, category or None, url or None, notes or None))
        return row["id"]
    else:
        row = sb_insert("background_tracks", payload)
        return row["id"]
def list_patients():
    if BACKEND == "postgres":
        return qall("select id, nome from public.patients order by nome asc")
    else:
        return sb_select("patients", columns="id,nome", order=("nome", True))

def insert_patient(nome: str, telefone: str, email: str, nascimento, notas: str) -> str:
    payload = {"nome": nome, "telefone": telefone or None, "email": email or None, "nascimento": str(nascimento) if nascimento else None, "notas": notas or None}
    if BACKEND == "postgres":
        row = qone("""insert into public.patients (nome, telefone, email, nascimento, notas)
                      values (%s,%s,%s,%s,%s) returning id""",
                   (nome, telefone or None, email or None, nascimento, notas or None))
        return row["id"]
    else:
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
    else:
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
    else:
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
st.title("claudiafito_v2")
if BACKEND == "postgres":
    st.caption("Backend: Postgres (DATABASE_URL).")
else:
    st.caption("Backend: Supabase REST (SUPABASE_URL + SUPABASE_KEY).")

with st.sidebar:
    st.header("Paciente")
    patients = list_patients()
    labels = ["— Novo paciente —"] + [p["nome"] for p in patients]
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
        st.session_state["patient_id"] = next(p["id"] for p in patients if p["nome"] == sel)

    st.divider()
    st.header("Áudio")
    binaural_devices = list_audio_devices("BINAURAL")
    bg_tracks = list_background_tracks()

    bd_labels = ["— Selecionar —"] + [f'{d["name"]} ({d.get("connection") or "-"})' for d in binaural_devices]
    bt_labels = ["— Nenhum —"] + [f'{t["name"]} [{t.get("category") or "-"}]' for t in bg_tracks]

    bd_sel = st.selectbox("Dispositivo binaural (fone)", bd_labels, index=1 if len(bd_labels)>1 else 0)
    bt_sel = st.selectbox("Som de fundo (opcional)", bt_labels, index=0)

    bin_vol = st.slider("Volume binaural (sugestão)", 0, 100, 60)
    bg_vol = st.slider("Volume som de fundo (sugestão)", 0, 100, 25)

    # resolve ids
    binaural_device = None
    if bd_sel != "— Selecionar —" and binaural_devices:
        idx = bd_labels.index(bd_sel) - 1
        if idx >= 0:
            binaural_device = binaural_devices[idx]

    bg_track = None
    if bt_sel != "— Nenhum —" and bg_tracks:
        idx = bt_labels.index(bt_sel) - 1
        if idx >= 0:
            bg_track = bg_tracks[idx]

    st.session_state["audio_cfg"] = {
        "binaural_device": binaural_device,
        "background_track": bg_track,
        "mix": {"binaural_volume": bin_vol, "background_volume": bg_vol},
    }

    with st.expander("➕ Cadastrar dispositivo / trilha"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Novo dispositivo**")
            nd_name = st.text_input("Nome do dispositivo", key="nd_name")
            nd_kind = st.selectbox("Tipo", ["BINAURAL", "BACKGROUND", "MIXER"], key="nd_kind")
            nd_conn = st.text_input("Conexão (P2/BT/USB)", key="nd_conn")
            nd_notes = st.text_input("Notas (opcional)", key="nd_notes")
            if st.button("Salvar dispositivo", key="save_dev"):
                if nd_name.strip():
                    insert_audio_device(nd_name.strip(), nd_kind, nd_conn.strip(), nd_notes.strip())
                    st.success("Dispositivo salvo!")
                    st.rerun()
                else:
                    st.warning("Informe o nome do dispositivo.")
        with c2:
            st.markdown("**Nova trilha de fundo**")
            nt_name = st.text_input("Nome da trilha", key="nt_name")
            nt_cat = st.text_input("Categoria (nature/noise/ambient)", key="nt_cat")
            nt_url = st.text_input("URL (opcional, mp3/stream)", key="nt_url")
            nt_notes = st.text_input("Notas (opcional)", key="nt_notes")
            if st.button("Salvar trilha", key="save_track"):
                if nt_name.strip():
                    insert_background_track(nt_name.strip(), nt_cat.strip(), nt_url.strip(), nt_notes.strip())
                    st.success("Trilha salva!")
                    st.rerun()
                else:
                    st.warning("Informe o nome da trilha.")

    if bg_track and (bg_track.get("url") or "").strip():
        st.caption("Prévia do som de fundo:")
        st.audio(bg_track["url"])

patient_id = st.session_state.get("patient_id")
if not patient_id:
    st.stop()

tab_atend, tab_binaural = st.tabs(["Atendimento", "Binaural"])

with tab_atend:
    st.subheader("Atendimento (anamnese → plano → sessões)")

    col1, col2 = st.columns([2, 1])
    with col1:
        complaint = st.text_input("Queixa principal (curta)")
    with col2:
        atend_date = st.date_input("Data", value=date.today())

    st.subheader("Anamnese (0–4)")
    answers = {}
    cols = st.columns(2)
    for i, q in enumerate(QUESTIONS):
        with cols[i % 2]:
            answers[q["id"]] = st.slider(q["label"], 0, 4, 0, key=q["id"])

    st.subheader("Sinais de atenção")
    flags = {}
    fcols = st.columns(2)
    for i, f in enumerate(FLAGS):
        with fcols[i % 2]:
            flags[f["id"]] = st.checkbox(f["label"], value=False, key=f["id"])

    notes = st.text_area("Notas do terapeuta (opcional)", height=100)

    scores = compute_scores(answers)
    focus = pick_focus(scores, top_n=3)
    qty, cadence = sessions_from_scores(scores)

    protocols = load_protocols()
    selected_names = select_protocols(scores, protocols)
    plan = merge_plan(selected_names, protocols)
    audio_cfg = st.session_state.get("audio_cfg")
    scripts = build_session_scripts(qty, cadence, focus, selected_names, protocols, audio_cfg)

    st.divider()
    left, right = st.columns(2)
    with left:
        df = pd.DataFrame([{"dominio": k, "score": v} for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.write("Foco:", focus)
        st.write("Sessões sugeridas:", {"qty": qty, "cadence_days": cadence})
    with right:
        st.write("Protocolos:", selected_names)
        st.write("Plano:", plan)

    st.subheader("Sessões pré-definidas (gravará em sessions_nova)")
    st.dataframe(pd.DataFrame([{
                    "sessao": s["session_n"],
                    "data": s["scheduled_date"],
                    "binaural_device": (s.get("audio") or {}).get("binaural_device", {}).get("name") if (s.get("audio") or {}).get("binaural_device") else None,
                    "som_fundo": (s.get("audio") or {}).get("background_track", {}).get("name") if (s.get("audio") or {}).get("background_track") else None,
                } for s in scripts]),
                 use_container_width=True, hide_index=True)

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Salvar anamnese (intake)", use_container_width=True):
            intake_id = insert_intake(patient_id, complaint, answers, scores, flags, notes)
            st.session_state["last_intake_id"] = intake_id
            st.success("Anamnese salva!")

    with b2:
        if st.button("Gerar plano + criar sessões", type="primary", use_container_width=True):
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
                plan_json={"date": str(atend_date), "complaint": complaint, "scores": scores, "focus": focus,
                          "selected_protocols": selected_names, "plan": plan, "audio_cfg": audio_cfg},
            )

            for s in scripts:
                insert_session_nova(plan_id, patient_id, int(s["session_n"]), s["scheduled_date"], s["status"], s)

            st.success(f"Plano criado e sessões geradas em sessions_nova! plan_id={plan_id}")

with tab_binaural:
    st.subheader("Binaural (igual ao app antigo)")

    audio_cfg = st.session_state.get("audio_cfg") or {}
    dev = audio_cfg.get("binaural_device")
    bg = audio_cfg.get("background_track")
    mix = audio_cfg.get("mix") or {}

    if dev:
        st.caption(f"🎧 Dispositivo selecionado (sidebar): **{dev.get('name')}** ({dev.get('connection') or '-'})")
    else:
        st.caption("🎧 Selecione um dispositivo binaural (fone) na sidebar, se quiser registrar no atendimento.")

    # controles (mesma UI do antigo)
    c1, c2, c3 = st.columns(3)
    carrier = c1.number_input("Carrier (Hz)", 50.0, 1000.0,
                              float(st.session_state.get(K("binaural", "carrier"), 220.0)),
                              1.0, key=K("binaural", "carrier"))
    beat = c2.number_input("Batida (Hz)", 0.5, 45.0,
                           float(st.session_state.get(K("binaural", "beat"), 10.0)),
                           0.5, key=K("binaural", "beat"))
    dur = int(c3.number_input("Duração (s)", 10, 3600,
                              int(st.session_state.get(K("binaural", "dur"), 120)),
                              5, key=K("binaural", "dur")))

    bt = abs(float(beat))
    fL = max(20.0, float(carrier) - bt / 2.0)
    fR = float(carrier) + bt / 2.0
    mL, mR = st.columns(2)
    mL.metric("Esquerdo (L)", f"{fL:.2f} Hz")
    mR.metric("Direito (R)", f"{fR:.2f} Hz")

    with st.expander("Como funciona?"):
        st.markdown(
            """
**Binaural** = duas frequências **puras** diferentes em cada ouvido → o cérebro percebe a **diferença** como um tom de batida (**beat**).  
**Cálculo:** `L = carrier − beat/2` e `R = carrier + beat/2`.  
Ex.: carrier 220 Hz e beat 10 Hz ⇒ L = **215 Hz**, R = **225 Hz** ⇒ batida percebida **~10 Hz**.

**Faixas úteis (guia rápido):**
- **Delta** (1–4 Hz): sono profundo, reparo  
- **Theta** (4–8 Hz): introspecção/relaxamento  
- **Alpha** (8–12 Hz): foco calmo  
- **Beta baixa** (12–18 Hz): atenção/alerta leve  
- **Gamma** (30–45 Hz): estimulação breve  
            """
        )

    st.markdown("🎵 Música de fundo (opcional)")
    st.caption("Você pode usar a trilha da sidebar (URL) ou enviar um arquivo (até 12MB).")
    bg_up = st.file_uploader("MP3/WAV/OGG (até 12MB)", type=["mp3", "wav", "ogg"], key=K("binaural", "bg_file"))

    # ganhos (0..0.4), com defaults parecidos com o app antigo
    def _vol_to_gain(v: int, scale: float) -> float:
        try:
            return float(max(0.0, min(0.4, float(v) / scale)))
        except Exception:
            return 0.12

    bg_gain_default = _vol_to_gain(int(mix.get("background_volume") or 25), 200.0)
    bin_gain_default = float(max(0.01, min(0.2, float(int(mix.get("binaural_volume") or 60)) / 1200.0)))

    bg_gain = st.slider("Volume do fundo", 0.0, 0.4, float(st.session_state.get(K("binaural", "bg_gain"), bg_gain_default)), 0.01, key=K("binaural", "bg_gain"))
    bin_gain = st.slider("Volume do binaural", 0.01, 0.2, float(st.session_state.get(K("binaural", "bin_gain"), bin_gain_default)), 0.01, key=K("binaural", "bin_gain"))

    raw = None
    filename = None
    if bg_up:
        raw = bg_up.read()
        filename = bg_up.name
        st.audio(raw)  # prévia

    bg_url = None
    if raw:
        data_url, _mime, err = bytes_to_data_url_safe(raw, filename)
        if err:
            st.warning(f"⚠️ {err}")
        else:
            bg_url = data_url
    elif bg and (bg.get("url") or "").strip():
        bg_url = bg.get("url")

    st.components.v1.html(
        webaudio_binaural_html(carrier, beat, dur, bg_url, bg_gain, bin_gain),
        height=300
    )

    wav = synth_binaural_wav(carrier, beat, seconds=min(float(dur), 20.0), sr=44100, amp=0.2)
    st.audio(wav, format="audio/wav")
    st.download_button(
        "Baixar WAV (20s)",
        data=wav,
        file_name=f"binaural_{int(carrier)}_{float(beat):.1f}.wav",
        mime="audio/wav",
        key=K("binaural", "dl_wav")
    )

    with st.expander("Sugestões rápidas por objetivo"):
        st.markdown(
            """
- **Relaxar/ansiedade** → **Theta 5–6 Hz** (15–20 min) e fechar em **Alpha 10 Hz** (5–10 min).
- **Sono** → **Delta 2–3 Hz** (10–20 min) → **Theta 5–6 Hz** (10–15 min).
- **Foco calmo** → **Alpha 10 Hz** (10–15 min).
- **Gamma 40 Hz** → estimulação breve (5–12 min), volume baixo.

> Use com fones, volume moderado. Se houver desconforto (tontura/dor de cabeça), reduza volume ou pare.
            """
        )
