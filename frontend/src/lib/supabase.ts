import { createClient, type SupabaseClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || '';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || '';

// createClient throws if URL/key are empty, so guard against missing config
export const supabaseConfigured = !!(supabaseUrl && supabaseAnonKey);

// PKCE flow: password-reset and email-confirmation links arrive as
// ?code= query params rather than #access_token= hash fragments.
// Email scanners that pre-fetch the link only see the app HTML — they
// can't call exchangeCodeForSession — so the OTP is never consumed
// before the real user clicks.
export const supabase: SupabaseClient = supabaseConfigured
  ? createClient(supabaseUrl, supabaseAnonKey, { auth: { flowType: 'pkce' } })
  : (null as unknown as SupabaseClient);
