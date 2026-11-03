-- Live data store (Supabase Postgres). Run once in the Supabase SQL editor.

create table if not exists stations (
    station_id  integer primary key,
    name        text not null,
    address     text,
    lat         double precision,
    lon         double precision,
    capacity    integer,
    updated_at  timestamptz default now()
);

-- One row per station per reading. The feed repeats a reading until the station reports
-- again, so the primary key quietly drops duplicates.
create table if not exists snapshots (
    station_id    integer not null,
    last_reported timestamptz not null,
    collected_at  timestamptz not null default now(),
    bikes         smallint not null,
    docks         smallint not null,
    is_installed  boolean,
    is_renting    boolean,
    is_returning  boolean,
    capacity      smallint,
    primary key (station_id, last_reported)
);
create index if not exists snapshots_time_idx on snapshots (last_reported);

-- Every prediction the live model makes, so we can later check whether it came true.
create table if not exists predictions (
    made_at         timestamptz not null,
    station_id      integer not null,
    horizon_min     smallint not null,
    target_time     timestamptz not null,
    bikes_now       smallint,
    predicted_bikes real,
    p_empty         real,
    p_full          real,
    model_version   text,
    primary key (made_at, station_id, horizon_min)
);
create index if not exists predictions_target_idx on predictions (target_time);

-- Each prediction next to what actually happened: the last reading at or before target_time,
-- as long as it is no more than 10 minutes old.
create or replace view prediction_accuracy as
select p.*,
       s.bikes                                  as actual_bikes,
       abs(p.predicted_bikes - s.bikes)         as abs_error,
       abs(p.bikes_now - s.bikes)               as persistence_abs_error,
       (s.bikes = 0)                            as actual_empty
from predictions p
join lateral (
    select bikes from snapshots s
    where s.station_id = p.station_id
      and s.last_reported <= p.target_time
      and s.last_reported >  p.target_time - interval '10 minutes'
    order by s.last_reported desc
    limit 1
) s on true;

-- Supabase exposes the public schema through its REST API. Turning on row-level security with
-- no policies blocks that API completely; the collector connects as the table owner, which
-- RLS does not apply to, so it keeps working.
alter table stations    enable row level security;
alter table snapshots   enable row level security;
alter table predictions enable row level security;
