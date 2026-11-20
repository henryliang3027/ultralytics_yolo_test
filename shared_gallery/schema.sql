-- Shared gallery: per-camera person IDs (from track-update batches) -> global person IDs.
-- Applied by gallery_store.py on start (idempotent).

-- One row per global person that still exists (merged ones are deleted; see global_alias).
CREATE TABLE IF NOT EXISTS global_person (
    global_person_id SERIAL PRIMARY KEY,
    created_ts       TIMESTAMPTZ NOT NULL,  -- tick that created it
    last_seen_ts     TIMESTAMPTZ NOT NULL
);

-- (camera_id, person_id) -> global_person_id. person_id is the camera's own (separate gallery) ID; it restarts at 1
-- when the pipeline restarts, so clear the gallery (POST /reset) when a camera's pipeline restarts.
-- first/last_seen_ts: when this local person was visible; two local persons of one camera whose spans overlap are
-- different people, so their global persons are never merged.
CREATE TABLE IF NOT EXISTS person_map (
    camera_id        TEXT        NOT NULL,
    person_id        INTEGER     NOT NULL,
    global_person_id INTEGER     NOT NULL REFERENCES global_person ON DELETE CASCADE,
    first_seen_ts    TIMESTAMPTZ NOT NULL,
    last_seen_ts     TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (camera_id, person_id)
);
CREATE INDEX IF NOT EXISTS person_map_global ON person_map (global_person_id);

-- Appearance bank: newest N embeddings per global person per camera.
CREATE TABLE IF NOT EXISTS embedding_bank (
    id               BIGSERIAL PRIMARY KEY,
    global_person_id INTEGER     NOT NULL REFERENCES global_person ON DELETE CASCADE,
    camera_id        TEXT        NOT NULL,
    person_id        INTEGER     NOT NULL,
    ts               TIMESTAMPTZ NOT NULL,
    vector           REAL[]      NOT NULL CHECK (array_length(vector, 1) = 256)
);
CREATE INDEX IF NOT EXISTS embedding_bank_global ON embedding_bank (global_person_id, camera_id, id);

-- Merges: from_global_id was found to be the same person as to_global_id and folded into it. Anyone holding an
-- old ID can follow this table to the current one.
CREATE TABLE IF NOT EXISTS global_alias (
    from_global_id INTEGER     PRIMARY KEY,
    to_global_id   INTEGER     NOT NULL,
    merged_ts      TIMESTAMPTZ NOT NULL,  -- tick of the merge
    similarity     REAL        NOT NULL   -- cosine of the two banks' mean embeddings
);

-- Every batch received; tick_ts is the key, so a batch sent twice is applied once.
CREATE TABLE IF NOT EXISTS batch_log (
    tick_ts     TIMESTAMPTZ PRIMARY KEY,
    received_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    records     INTEGER     NOT NULL,
    new_global  INTEGER     NOT NULL DEFAULT 0,
    merges      INTEGER     NOT NULL DEFAULT 0
);
