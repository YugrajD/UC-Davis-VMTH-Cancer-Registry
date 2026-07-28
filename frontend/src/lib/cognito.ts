import { CognitoUserPool } from 'amazon-cognito-identity-js';

const userPoolId = import.meta.env.VITE_COGNITO_USER_POOL_ID || '';
const clientId = import.meta.env.VITE_COGNITO_CLIENT_ID || '';
const endpoint = import.meta.env.VITE_COGNITO_ENDPOINT || undefined;

// cognito-local's seeded app client only allows USER_PASSWORD_AUTH (no SRP
// support), while the real Cognito pool should keep using the SDK's default
// USER_SRP_AUTH so the password never leaves the browser in plaintext.
export const authenticationFlowType: 'USER_PASSWORD_AUTH' | 'USER_SRP_AUTH' = endpoint
  ? 'USER_PASSWORD_AUTH'
  : 'USER_SRP_AUTH';

// Hosted UI domain for Google OAuth. Empty until a Hosted UI + Google OAuth
// client are configured on the real Cognito User Pool — cognito-local has
// no federation support, so this can't be exercised locally.
export const googleOAuthDomain: string = import.meta.env.VITE_COGNITO_GOOGLE_OAUTH_DOMAIN || '';
export const googleOAuthConfigured = !!googleOAuthDomain;

export const authConfigured = !!(userPoolId && clientId);

export const userPool: CognitoUserPool = authConfigured
  ? new CognitoUserPool({ UserPoolId: userPoolId, ClientId: clientId, endpoint })
  : (null as unknown as CognitoUserPool);

export function googleSignInUrl(redirectUri: string): string {
  const params = new URLSearchParams({
    identity_provider: 'Google',
    redirect_uri: redirectUri,
    response_type: 'code',
    client_id: clientId,
    scope: 'openid email profile',
  });
  return `${googleOAuthDomain.replace(/\/$/, '')}/oauth2/authorize?${params.toString()}`;
}
