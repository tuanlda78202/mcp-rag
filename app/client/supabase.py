from supabase import Client, create_client
from supabase.lib.client_options import ClientOptions

from app.config import settings

supabase: Client = create_client(
    settings.SUPABASE_URL,
    settings.SUPABASE_KEY,
    options=ClientOptions(postgrest_client_timeout=20),
)
