import os
import json
from datetime import date, timedelta
from typing import Dict, List, Tuple, Any

import pandas as pd
import streamlit as st
import psycopg2
import psycopg2.extras

st.set_page_config(page_title="claudiafito_v2", layout="wide")

# -------------------------
# DB
# -------------------------
def get_conn():
    dsn = st.secrets.get("DATABASE_URL", None) if hasattr(st, "secrets") else None
    dsn = dsn or os.getenv("DATABASE_URL")
    if not dsn:
        st.error("Defina DATABASE_URL (env ou st.secrets).")
        st.stop()
    return psycopg2.connect(dsn)

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
# Anamnese (0–4)
# -------------------------
DOMAINS = ["sono","ansiedade","humor_baixo","exaustao","pertencimento","tensao","ruminacao"]

QUESTIONS = [
    {"id":"sono_q1","label":"Dificuldade para pegar no sono", "domain":"sono", "weight":1.0},
    {"id":"sono_q2","label":"Acorda no meio da noite / sono leve", "domain":"sono", "weight":1.0},
    {"id":"ans_q1","label":"Ansiedade / agitação no dia a dia", "domain":"ansiedade", "weight":1.2},
    {"id":"ans_q2","label":"Sintomas físicos de ansiedade (aperto, inquietação)", "domain":"ansiedade", "weight":1.0},
    {"id":"hum_q1","label":"Tristeza / desânimo frequente", "domain":"humor_baixo", "weight":1.2},
    {"id":"hum_q2","label":"Perda de prazer / motivação", "domain":"humor_baixo", "weight":1.0},
    {"id":"exa_q1","label":"Cansaço / exaustão por responsabilidades", "domain":"exaustao", "weight":1.2},
    {"id":"exa_q2","label":"Pouco tempo para si / autocuidado", "domain":"exaustao", "weight":1.0},
    {"id":"per_q1","label":"Sensação de não pertencimento / desconexão", "domain":"pertencimento", "weight":1.2},
    {"id":"per_q2","label":"Vergonha / autojulgamento", "domain":"pertencimento", "weight":1.0},
    {"id":"ten_q1","label":"Tensão muscular / dores recorrentes", "domain":"tensao", "weight":1.0},
    {"id":"ten_q2","label":"Mandíbula/ombros travados / corpo em alerta", "domain":"tensao", "weight":1.0},
    {"id":"rum_q1","label":"Mente acelerada / ruminação", "domain":"ruminacao", "weight":1.2},
    {"id":"rum_q2","label":"Dificuldade de foco por pensamentos repetitivos", "domain":"ruminacao", "weight":1.0},
]

FLAGS = [
    {"id":"flag_preg","label":"Gestação / amamentação"},
    {"id":"flag_meds","label":"Uso de medicamentos (ansiolíticos/antidepressivos/sedativos)"},
    {"id":"flag_allergy","label":"Alergias / sensibilidades"},
    {"id":"flag_sound","label":"Sensibilidade a som (binaural)"},
    {"id":"flag_light","label":"Sensibilidade à luz (cama de cristal)"},
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
    sums = {d:0.0 for d in DOMAINS}
    maxs = {d:0.0 for d in DOMAINS}
    for q in QUESTIONS:
        v = float(answers.get(q["id"], 0))
        w = float(q["weight"])
        d = q["domain"]
        sums[d] += v * w
        maxs[d] += 4.0 * w
    return {d: round((sums[d]/maxs[d]*100.0) if maxs[d] else 0.0, 1) for d in DOMAINS}

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
    rows = qall("select name, domain, rules_json, content_json from public.protocol_library where active = true")
    return {r["name"]: {"name": r["name"], "domain": r["domain"], "rules": r["rules_json"], "content": r["content_json"]} for r in rows}

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
        c = protocols.get(name, {}).get("content", {})
        for ch in c.get("chakras_foco", []): add_unique(chakras, ch)
        for e in c.get("emocoes_foco", []): add_unique(emocoes, e)
        for cr in c.get("cristais", []):
            if cr not in cristais: cristais.append(cr)
        for f in c.get("fito", []):
            if f not in fito: fito.append(f)
        if c.get("alertas"): add_unique(alerts, c["alertas"])
    return {"chakras_prioritarios": chakras,"emocoes_prioritarias": emocoes,"cristais_sugeridos": cristais,"fito_sugerida": fito,"alertas": alerts}

def build_session_scripts(qty: int, cadence_days: int, focus: List[Tuple[str, float]],
                          selected_names: List[str], protocols: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
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

        base = protocols.get(BASE_PROTOCOL, {}).get("content", {})
        if base:
            parts.append({"card": BASE_PROTOCOL, "binaural": base.get("binaural"), "cama": base.get("cama_cristal"),
                          "cristais": base.get("cristais"), "fito": base.get("fito"),
                          "roteiro": base.get("roteiro_sessao")})

        if focus_card:
            fc = protocols.get(focus_card, {}).get("content", {})
            parts.append({"card": focus_card, "binaural": fc.get("binaural"), "cama": fc.get("cama_cristal"),
                          "cristais": fc.get("cristais"), "fito": fc.get("fito"),
                          "roteiro": fc.get("roteiro_sessao")})

        scripts.append({"session_n": i, "scheduled_date": str(session_date), "status": "AGENDADA", "parts": parts})
    return scripts

# -------------------------
# Persistência (patients: nome/nascimento/telefone/email/notas)
# -------------------------
def insert_patient(nome: str, telefone: str, email: str, nascimento, notas: str):
    row = qone(
        "insert into public.patients (nome, telefone, email, nascimento, notas) values (%s,%s,%s,%s,%s) returning id",
        (nome, telefone, email, nascimento, notas)
    )
    return row["id"]

def insert_intake(patient_id: str, complaint: str, answers: Dict[str, int], scores: Dict[str, float],
                 flags: Dict[str, bool], notes: str):
    row = qone("""insert into public.intakes (patient_id, complaint, answers_json, scores_json, flags_json, notes)
                  values (%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s)
                  returning id""",
               (patient_id, complaint, json.dumps(answers, ensure_ascii=False),
                json.dumps(scores, ensure_ascii=False),
                json.dumps(flags, ensure_ascii=False),
                notes))
    return row["id"]

def insert_plan_and_sessions(patient_id: str, intake_id: str, focus, selected_names, plan_json, qty, cadence, scripts):
    row = qone("""insert into public.plans (intake_id, patient_id, focus_json, selected_protocols, sessions_qty, cadence_days, plan_json)
                  values (%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s::jsonb)
                  returning id""",
               (intake_id, patient_id,
                json.dumps({"top": focus}, ensure_ascii=False),
                json.dumps(selected_names, ensure_ascii=False),
                qty, cadence,
                json.dumps(plan_json, ensure_ascii=False)))
    plan_id = row["id"]

    for s in scripts:
        qexec("""insert into public.sessions (plan_id, patient_id, session_n, scheduled_date, status, script_json)
                 values (%s,%s,%s,%s,%s,%s::jsonb)""",
              (plan_id, patient_id, int(s["session_n"]), s["scheduled_date"], s["status"],
               json.dumps(s, ensure_ascii=False)))
    return plan_id

# -------------------------
# UI (1 aba)
# -------------------------
st.title("claudiafito_v2 — Atendimento (1 aba)")
st.caption("Compatível com sua tabela public.patients (nome/nascimento/telefone/email/notas).")

with st.sidebar:
    st.header("Paciente")
    patients = qall("select id, nome from public.patients order by nome asc")
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
        # se houver nomes repetidos, este pega o primeiro; podemos evoluir depois para mostrar nome + data/id
        st.session_state["patient_id"] = next(p["id"] for p in patients if p["nome"] == sel)

patient_id = st.session_state.get("patient_id")
if not patient_id:
    st.stop()

col1, col2 = st.columns([2,1])
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
scripts = build_session_scripts(qty, cadence, focus, selected_names, protocols)

st.divider()
left, right = st.columns(2)
with left:
    st.write("Scores:", scores)
    st.write("Foco:", focus)
    st.write("Sessões:", {"qty": qty, "cadence_days": cadence})
with right:
    st.write("Protocolos:", selected_names)
    st.write("Plano:", plan)

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
    if st.button("Gerar plano + criar sessões", type="primary", use_container_width=True):
        intake_id = st.session_state.get("last_intake_id")
        if not intake_id:
            intake_id = insert_intake(patient_id, complaint, answers, scores, flags, notes)
            st.session_state["last_intake_id"] = intake_id
        plan_id = insert_plan_and_sessions(
            patient_id, intake_id, focus, selected_names,
            {"date": str(atend_date), "complaint": complaint, "scores": scores,
             "focus": focus, "selected_protocols": selected_names, "plan": plan},
            qty, cadence, scripts
        )
        st.success(f"Plano criado e sessões geradas! plan_id={plan_id}")
