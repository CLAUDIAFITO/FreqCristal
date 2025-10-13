
# Sistema de Frequências Terapêuticas — Cama de Cristal (Streamlit + Supabase)

Projeto do zero para gerar **protocolos de frequências** para sessões de **cama de cristal**,
com cadastro de pacientes e registro de sessões.

## Stack
- **Frontend/App**: Streamlit
- **Banco**: Postgres no **Supabase**
- **ORM/Client**: `supabase-py`
- **Auth**: Supabase Auth (email/senha ou magic link)
- **Armazenamento**: tabelas no Postgres (RLS ativada)

## Funcionalidades (MVP)
- Catálogo de frequências (Solfeggio, Chakras, Cores).
- Gerador de protocolo baseado em **intenção**, **chakra alvo**, **cor**, **tempo de sessão**.
- Cadastro de **Pacientes**, **Sessões** e **Frequências da Sessão**.
- Histórico de sessões por paciente.
- Exportação do protocolo em **CSV** (e pronto para PDF posteriormente).

## Como rodar localmente
1) Crie um projeto no **Supabase**. Copie `SUPABASE_URL` e `SUPABASE_ANON_KEY`.
2) Crie as tabelas rodando o SQL de `supabase_schema.sql` e **ative o RLS** (já incluso no arquivo).
3) (Opcional) Rode os *seeds* com `seed_frequencies.csv` usando o próprio app (página **Admin**) ou inserindo manualmente.
4) Configure variáveis de ambiente:
   ```bash
   export SUPABASE_URL="https://xxxxx.supabase.co"
   export SUPABASE_KEY="eyJhbGciOiJ..."
   export SUPABASE_SERVICE_ROLE=""  # opcional, apenas se for necessário seed programático
   ```
5) Instale dependências e rode o app:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Deploy (Streamlit Cloud ou servidor próprio)
- Defina as variáveis de ambiente `SUPABASE_URL` e `SUPABASE_KEY` no painel de deploy.
- Garanta que o banco já está com o schema criado e RLS ativo.
