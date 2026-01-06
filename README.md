
# claudiafito_v2 (enxuto)

Este projeto cria um app Streamlit com **1 aba**:
- Cadastro/seleção de paciente
- Anamnese pontuável (0–4)
- Motor de regras: gera **Plano** (Binaural + Cama de Cristal + Fito + Cristais)
- Cria automaticamente as **Sessões** no banco

## 1) Banco (Supabase)
1. Abra o **SQL Editor** no Supabase
2. Rode: `db/schema.sql`
3. Rode: `db/seed.sql`

## 2) Variáveis de ambiente / secrets
O app usa `DATABASE_URL` (padrão do Postgres), exemplo:

postgresql://USER:PASSWORD@HOST:5432/DBNAME

No Streamlit Cloud, coloque em **Secrets**:
```
DATABASE_URL = "postgresql://..."
```

## 3) Rodar local
```
pip install -r requirements.txt
streamlit run app.py
```

## Aviso
O sistema gera **sugestões de bem-estar**. Você sempre ajusta conforme:
- contraindicações, alergias, medicamentos
- sensibilidade a som/luz
- julgamento terapêutico profissional

- Defina as variáveis de ambiente `SUPABASE_URL` e `SUPABASE_KEY` no painel de deploy.
- Garanta que o banco já está com o schema criado e RLS ativo.
