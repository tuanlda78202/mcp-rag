-- Enable the vector extension
create extension vector
with
  schema extensions;

-- Create the rag_embed table
-- drop table rag_embed;
create table rag_embed (
  id          text PRIMARY KEY NOT NULL,
  file_id     text NOT NULL,
  content     text,

  -- Search
  embedding   vector(768),

  -- Metadata
  updated_at  timestamp with time zone not null default (now() AT TIME ZONE 'utc'::text)
);

-- INDEXING
-- Create an index for the semantic vector search
-- We are using the vector_ip_ops (inner product) operator with this index
-- because we plan on using the inner product (<#>) operator later
create index on rag_embed using hnsw (embedding vector_ip_ops);

-- SECURITY
-- Create a policy to allow the authenticated role to read by Org ID
alter table "public"."rag_embed"
enable row level security;
