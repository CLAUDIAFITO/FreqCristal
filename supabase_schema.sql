
create extension if not exists "uuid-ossp";

create table if not exists public.users_app (
  id uuid primary key default uuid_generate_v4(),
  auth_user_id uuid unique,
  email text unique not null,
  nome text,
  created_at timestamp with time zone default now()
);

create table if not exists public.patients (
  id uuid primary key default uuid_generate_v4(),
  created_by uuid references public.users_app(id) on delete set null,
  nome text not null,
  nascimento date,
  notas text,
  created_at timestamp with time zone default now()
);

do $$ begin
  create type freq_type as enum ('solfeggio','chakra','cor','custom');
exception when duplicate_object then null; end $$;

do $$ begin
  create type chakra as enum ('raiz','sacral','plexo','cardiaco','laringeo','terceiro_olho','coronal');
exception when duplicate_object then null; end $$;

create table if not exists public.frequencies (
  id uuid primary key default uuid_generate_v4(),
  code text unique,
  nome text not null,
  hz numeric not null check (hz > 0),
  tipo freq_type not null,
  chakra chakra,
  cor text,
  descricao text,
  created_at timestamp with time zone default now()
);

do $$ begin
  create type session_status as enum ('rascunho','agendada','em_andamento','finalizada','cancelada');
exception when duplicate_object then null; end $$;

create table if not exists public.sessions (
  id uuid primary key default uuid_generate_v4(),
  created_by uuid references public.users_app(id) on delete set null,
  patient_id uuid references public.patients(id) on delete cascade,
  data timestamp with time zone default now(),
  duracao_min integer check (duracao_min between 5 and 180),
  intencao text,
  chakra_alvo chakra,
  cor_alvo text,
  status session_status default 'rascunho',
  protocolo jsonb,
  created_at timestamp with time zone default now()
);

create table if not exists public.session_frequencies (
  session_id uuid references public.sessions(id) on delete cascade,
  freq_id uuid references public.frequencies(id) on delete restrict,
  ordem integer not null,
  duracao_seg integer not null check (duracao_seg > 0),
  amplitude numeric,
  forma_onda text,
  observacoes text,
  primary key (session_id, ordem)
);

alter table public.users_app enable row level security;
alter table public.patients enable row level security;
alter table public.sessions enable row level security;
alter table public.session_frequencies enable row level security;
alter table public.frequencies enable row level security;

create policy "users_app read" on public.users_app
  for select using (true);
create policy "users_app insert" on public.users_app
  for insert with check (true);

create policy "patients by owner" on public.patients
  for all using (auth.uid() is not null) with check (auth.uid() is not null);

create policy "sessions by owner" on public.sessions
  for all using (auth.uid() is not null) with check (auth.uid() is not null);

create policy "session_frequencies by owner" on public.session_frequencies
  for all using (
    auth.uid() is not null
    and exists (
      select 1 from public.sessions s where s.id = session_frequencies.session_id
    )
  ) with check (auth.uid() is not null);

create policy "frequencies read" on public.frequencies
  for select using (true);
create policy "frequencies block write" on public.frequencies
  for insert with check (false);
create policy "frequencies block update" on public.frequencies
  for update using (false) with check (false);
