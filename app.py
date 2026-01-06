import os
import json
from datetime import date, timedelta
from typing import Dict, List, Tuple, Any

import pandas as pd
import streamlit as st
import psycopg2
import psycopg2.extras

st.set_page_config(page_title="claudiafito_v2", layout="wide")

# =========================
# Conexão / Helpers SQL
# =========================
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

# =========================
# Modelo de Anamnese (enxuto)
# Cada pergunta: 0–4
# =========================
DOMAINS = [
    "sono",
    "ansiedade",
    "humor_baixo",
    "exaustao",
    "pertencimento",
    "tensao",
    "ruminacao",
]

QUESTIONS = [
    # Sono
    {"id":"sono_q1","label":"Dificuldade para pegar no sono", "domain":"sono", "weight":1.0},
    {"id":"sono_q2","label":"Acorda no meio da noite / sono leve", "domain":"sono", "weight":1.0},
    # Ansiedade
    {"id":"ans_q1","label":"Ansiedade / agitação no dia a dia", "domain":"ansiedade", "weight":1.2},
    {"id":"ans_q2","label":"Sintomas físicos de ansiedade (aperto, taquicardia, inquietação)", "domain":"ansiedade", "weight":1.0},
    # Humor baixo
    {"id":"hum_q1","label":"Tristeza / desânimo frequente", "domain":"humor_baixo", "weight":1.2},
    {"id":"hum_q2","label":"Perda de prazer / motivação", "domain":"humor_baixo", "weight":1.0},
    # Exaustão
    {"id":"exa_q1","label":"Cansaço / exaustão por responsabilidades", "domain":"exaustao", "weight":1.2},
    {"id":"exa_q2","label":"Pouco tempo para si / autocuidado", "domain":"exaustao", "weight":1.0},
    # Pertencimento
    {"id":"per_q1","label":"Sensação de não pertencimento / desconexão", "domain":"pertencimento", "weight":1.2},
    {"id":"per_q2","label":"Vergonha / autojulgamento", "domain":"pertencimento", "weight":1.0},
    # Tensão
    {"id":"ten_q1","label":"Tensão muscular / dores recorrentes", "domain":"tensao", "weight":1.0},
    {"id":"ten_q2","label":"Mandíbula/ombros travados / corpo em alerta", "domain":"tensao", "weight":1.0},
    # Ruminação
    {"id":"rum_q1","label":"Mente acelerada / ruminação", "domain":"ruminacao", "weight":1.2},
    {"id":"rum_q2","label":"Dificuldade de foco por pensamentos repetitivos", "domain":"ruminacao", "weight":1.0},
]

FLAGS = [
    {"id":"flag_preg","label":"Gestação / amamentação"},
    {"id":"flag_meds","label":"Uso de medicamentos (especialmente ansiolíticos/antidepressivos/sedativos)"},
    {"id":"flag_allergy","label":"Alergias / sensibilidades"},
    {"id":"flag_sound","label":"Sensibilidade a som (binaural)"},
    {"id":"flag_light","label":"Sensibilidade à luz (cama de cristal)"},
]

# Map domínios -> protocolo(s)
DOMAIN_TO_PROTOCOL = {
    "ansiedade": "FOCO – Ansiedade / Agitação",
    "sono": "FOCO – Sono Profundo",
    "exaustao": "FOCO – Exaustão / Sobrecarga",
    "pertencimento": "FOCO – Pertencimento / Vergonha",
    # humor_baixo, tensao, ruminacao podem ser atendidos por combinações
    "humor_baixo": "FOCO – Exaustão / Sobrecarga",
    "tensao": "FOCO – Ansiedade / Agitação",
    "ruminacao": "FOCO – Ansiedade / Agitação",
}

BASE_PROTOCOL = "BASE – Aterramento + Regulação"

# =========================
# Cálculo de scores
# =========================
def compute_scores(answers: Dict[str, int]) -> Dict[str, float]:
    # soma ponderada por domínio, normaliza 0–100
    sums = {d:0.0 for d in DOMAINS}
    maxs = {d:0.0 for d in DOMAINS}
    for q in QUESTIONS:
        v = float(answers.get(q["id"], 0))
        w = float(q["weight"])
        d = q["domain"]
        sums[d] += v * w
        maxs[d] += 4.0 * w
    scores = {}
    for d in DOMAINS:
        scores[d] = round((sums[d] / maxs[d] * 100.0) if maxs[d] > 0 else 0.0, 1)
    return scores

def pick_focus(scores: Dict[str, float], top_n=3) -> List[Tuple[str, float]]:
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return items[:top_n]

def sessions_from_scores(scores: Dict[str, float]) -> Tuple[int, int]:
    # sessões + cadência em dias
    top = sorted(scores.values(), reverse=True)
    max_score = top[0] if top else 0.0

    strong = sum(1 for s in top[:3] if s >= 70)
    if strong <= 0:
        strong = sum(1 for s in top[:3] if s >= 60)

    if strong <= 1:
        qty = 4
    elif strong == 2:
        qty = 6
    else:
        qty = 8

    if max_score >= 80:
        cadence = 7
    elif max_score >= 60:
        cadence = 10
    else:
        cadence = 14

    return qty, cadence

# =========================
# Protocolos do banco
# =========================
def load_protocols() -> Dict[str, Dict[str, Any]]:
    rows = qall("select name, domain, rules_json, content_json from public.protocol_library where active = true")
    prot = {}
    for r in rows:
        prot[r["name"]] = {
            "name": r["name"],
            "domain": r["domain"],
            "rules": r["rules_json"],
            "content": r["content_json"],
        }
    return prot

def select_protocols(scores: Dict[str, float], protocols: Dict[str, Dict[str, Any]]) -> List[str]:
    selected = [BASE_PROTOCOL] if BASE_PROTOCOL in protocols else []
    for dom, sc in scores.items():
        if sc >= 60:
            pname = DOMAIN_TO_PROTOCOL.get(dom)
            if pname and pname in protocols and pname not in selected:
                selected.append(pname)
    if BASE_PROTOCOL not in selected and BASE_PROTOCOL in protocols:
        selected.insert(0, BASE_PROTOCOL)
    return selected

def merge_plan(selected_names: List[str], protocols: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    # consolida chakras/emocoes/cristais/fito e mantém cards para referência
    chakras = []
    emocoes = []
    cristais = []
    fito = []
    alerts = []
    cards = []

    def _add_unique(lst, item):
        if item not in lst:
            lst.append(item)

    for name in selected_names:
        c = protocols.get(name, {}).get("content", {})
        cards.append({"name": name, "content": c})

        for ch in c.get("chakras_foco", []):
            _add_unique(chakras, ch)
        for e in c.get("emocoes_foco", []):
            _add_unique(emocoes, e)

        for cr in c.get("cristais", []):
            if cr not in cristais:
                cristais.append(cr)

        for f in c.get("fito", []):
            if f not in fito:
                fito.append(f)

        a = c.get("alertas")
        if a:
            _add_unique(alerts, a)

    return {
        "chakras_prioritarios": chakras,
        "emocoes_prioritarias": emocoes,
        "cristais_sugeridos": cristais,
        "fito_sugerida": fito,
        "alertas": alerts,
        "cards": cards,
    }

def build_session_scripts(qty: int, cadence_days: int, focus: List[Tuple[str, float]],
                          selected_names: List[str], protocols: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    # alterna foco principal; BASE entra sempre
    # ordem de foco: top 3 domínios
    focus_domains = [d for d, _ in focus]
    scripts = []
    today = date.today()

    # escolher 1 card de foco por sessão (cicla)
    focus_cards = []
    for dom in focus_domains:
        pname = DOMAIN_TO_PROTOCOL.get(dom)
        if pname and pname in selected_names and pname != BASE_PROTOCOL:
            focus_cards.append(pname)
    if not focus_cards:
        focus_cards = [n for n in selected_names if n != BASE_PROTOCOL][:1]

    for i in range(1, qty+1):
        session_date = today + timedelta(days=cadence_days*(i-1))
        focus_card = focus_cards[(i-1) % len(focus_cards)] if focus_cards else None

        parts = []
        # BASE content
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

        scripts.append({
            "session_n": i,
            "scheduled_date": str(session_date),
            "status": "AGENDADA",
            "parts": parts,
            "roteiro_unificado": [
                "Check-in (2 min)",
                "Intenção (1 min)",
                "Cama de cristal (15–25 min conforme sequência)",
                "Binaural (15–30 min conforme objetivo)",
                "Cristais (ancoragem 3–5 min)",
                "Fito (orientação sem posologia automática)",
                "Fechamento (respiração + afirmação)"
            ]
        })
    return scripts

# =========================
# Persistência
# =========================
def upsert_patient(full_name: str, phone: str, email: str, birth_date, notes: str):
    # tenta achar por nome+email (simples)
    row = qone("select id from public.patients where full_name=%s and coalesce(email,'')=coalesce(%s,'') limit 1",
               (full_name, email))
    if row:
        pid = row["id"]
        qexec("""update public.patients
                 set phone=%s, email=%s, birth_date=%s, notes=%s
                 where id=%s""", (phone, email, birth_date, notes, pid))
        return pid
    row = qone("""insert into public.patients (full_name, phone, email, birth_date, notes)
                  values (%s,%s,%s,%s,%s) returning id""",
               (full_name, phone, email, birth_date, notes))
    return row["id"]

def insert_intake(patient_id: str, complaint: str, answers: Dict[str, int], scores: Dict[str, float], flags: Dict[str, bool], notes: str):
    row = qone("""insert into public.intakes (patient_id, complaint, answers_json, scores_json, flags_json, notes)
                  values (%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s)
                  returning id""",
               (patient_id, complaint, json.dumps(answers, ensure_ascii=False),
                json.dumps(scores, ensure_ascii=False),
                json.dumps(flags, ensure_ascii=False),
                notes))
    return row["id"]

def insert_plan_and_sessions(patient_id: str, intake_id: str, focus: List[Tuple[str, float]],
                             selected_names: List[str], plan_json: Dict[str, Any],
                             qty: int, cadence: int, scripts: List[Dict[str, Any]]):
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

# =========================
# UI (1 aba)
# =========================
st.title("claudiafito_v2 — Atendimento (1 aba)")
st.caption("Binaural + Cama de Cristal + Fito + Cristais — sugestões de bem-estar (ajuste profissional sempre).")

# Pacientes
with st.sidebar:
    st.header("Paciente")
    patients = qall("select id, full_name from public.patients order by full_name asc")
    options = ["— Novo paciente —"] + [p["full_name"] for p in patients]
    sel = st.selectbox("Selecionar", options, index=0)

    if sel == "— Novo paciente —":
        full_name = st.text_input("Nome completo")
        phone = st.text_input("Telefone (opcional)")
        email = st.text_input("E-mail (opcional)")
        birth_date = st.date_input("Data de nascimento (opcional)", value=None)
        pnotes = st.text_area("Observações do paciente (opcional)")
        if st.button("Salvar paciente", use_container_width=True, type="primary"):
            if not full_name.strip():
                st.warning("Informe o nome.")
            else:
                pid = upsert_patient(full_name.strip(), phone.strip(), email.strip(), birth_date, pnotes.strip())
                st.success("Paciente salvo!")
                st.session_state["patient_id"] = pid
                st.rerun()
    else:
        pid = next(p["id"] for p in patients if p["full_name"] == sel)
        st.session_state["patient_id"] = pid

patient_id = st.session_state.get("patient_id")
if not patient_id:
    st.info("Selecione ou cadastre um paciente na barra lateral.")
    st.stop()

# Dados do atendimento
col1, col2, col3 = st.columns([2,2,1])
with col1:
    complaint = st.text_input("Queixa principal (curta)", placeholder="Ex.: ansiedade + insônia + sobrecarga")
with col2:
    atend_date = st.date_input("Data do atendimento", value=date.today())
with col3:
    st.write("")
    st.write("")
    if st.button("Ver histórico", use_container_width=True):
        st.session_state["show_history"] = True

# Anamnese
st.subheader("Anamnese (0–4) — quanto isso te impacta hoje?")
answers = {}
cols = st.columns(2)
for i, q in enumerate(QUESTIONS):
    with cols[i % 2]:
        answers[q["id"]] = st.slider(q["label"], 0, 4, 0, key=q["id"])

st.subheader("Sinais de atenção (checklist)")
flags = {}
flag_cols = st.columns(2)
for i, f in enumerate(FLAGS):
    with flag_cols[i % 2]:
        flags[f["id"]] = st.checkbox(f["label"], value=False, key=f["id"])

notes = st.text_area("Notas do terapeuta (opcional)", height=100)

# Motor
scores = compute_scores(answers)
focus = pick_focus(scores, top_n=3)
qty, cadence = sessions_from_scores(scores)

protocols = load_protocols()
if BASE_PROTOCOL not in protocols:
    st.warning("Protocolo BASE não encontrado no banco. Rode db/seed.sql.")

selected_names = select_protocols(scores, protocols)
plan_json = merge_plan(selected_names, protocols)
scripts = build_session_scripts(qty, cadence, focus, selected_names, protocols)

# Visualização
st.divider()
left, right = st.columns([1,1])

with left:
    st.subheader("Scores (0–100)")
    df = pd.DataFrame([{"dominio":k, "score":v} for k,v in sorted(scores.items(), key=lambda x: x[1], reverse=True)])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Focos detectados (Top 3)")
    st.write([{"dominio":d, "score":s} for d,s in focus])

    st.subheader("Pacote sugerido")
    st.write({"sessoes": qty, "cadencia_dias": cadence})

with right:
    st.subheader("Plano consolidado (obrigatório + focos)")
    st.write({
        "protocolos": selected_names,
        "chakras": plan_json["chakras_prioritarios"],
        "emocoes": plan_json["emocoes_prioritarias"],
        "cristais": plan_json["cristais_sugeridos"],
        "fito": plan_json["fito_sugerida"],
        "alertas": plan_json["alertas"],
    })

st.subheader("Roteiro das sessões (pré-definido)")
st.dataframe(pd.DataFrame([{
    "Sessão": s["session_n"],
    "Data sugerida": s["scheduled_date"],
    "Cards": " + ".join([p["card"] for p in s["parts"] if p.get("card")]),
    "Binaural": "; ".join([json.dumps(p.get("binaural", {}), ensure_ascii=False) for p in s["parts"]]),
} for s in scripts]), use_container_width=True, hide_index=True)

st.caption("Dica: você pode ajustar manualmente qualquer item antes de 'Criar sessões' (no código, ou criando um editor depois).")

# Ações
st.divider()
a1, a2, a3 = st.columns([1,1,1])

with a1:
    if st.button("Salvar anamnese (gera intake)", type="secondary", use_container_width=True):
        intake_id = insert_intake(patient_id, complaint, answers, scores, flags, notes)
        st.session_state["last_intake_id"] = intake_id
        st.success("Anamnese salva (intake criado).")

with a2:
    if st.button("Gerar plano + criar sessões", type="primary", use_container_width=True):
        intake_id = st.session_state.get("last_intake_id")
        if not intake_id:
            intake_id = insert_intake(patient_id, complaint, answers, scores, flags, notes)
            st.session_state["last_intake_id"] = intake_id
        plan_id = insert_plan_and_sessions(patient_id, intake_id, focus, selected_names, {
            "atendimento_data": str(atend_date),
            "complaint": complaint,
            "scores": scores,
            "focus": focus,
            "selected_protocols": selected_names,
            "plan": plan_json,
        }, qty, cadence, scripts)
        st.success(f"Plano criado e sessões geradas! plan_id={plan_id}")

with a3:
    with st.expander("Exportar plano (JSON)"):
        export = {
            "patient_id": patient_id,
            "date": str(atend_date),
            "complaint": complaint,
            "answers": answers,
            "scores": scores,
            "focus": focus,
            "selected_protocols": selected_names,
            "sessions_qty": qty,
            "cadence_days": cadence,
            "plan": plan_json,
            "sessions": scripts,
            "disclaimer": "Sugestões de bem-estar. Ajustar conforme contraindicações/alergias/medicamentos e avaliação profissional."
        }
        st.code(json.dumps(export, ensure_ascii=False, indent=2), language="json")

# Histórico
if st.session_state.get("show_history"):
    st.divider()
    st.subheader("Histórico (últimos planos)")
    rows = qall("""
        select p.created_at, p.sessions_qty, p.cadence_days, p.selected_protocols, p.focus_json
          from public.plans p
         where p.patient_id = %s
         order by p.created_at desc
         limit 20
    """, (patient_id,))
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
