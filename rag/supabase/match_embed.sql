create or replace function match_embed (
  p_query_embedding    vector(768),
  p_match_threshold    float,
  p_match_count        int
)
returns table (
  id text,
  file_id text,
  content text,
  similarity float
)
language sql
as $$
  with top_embeddings as (
    select
      e.id,
      e.file_id,
      e.content,
      -- Calculate similarity (negative inner product)
      -(e.embedding <#> p_query_embedding) as similarity
    from rag_embed e
    where e.embedding <#> p_query_embedding < -p_match_threshold
    order by e.embedding <#> p_query_embedding
    limit least(p_match_count, 50)
  )
  select *
  from top_embeddings;
$$;


