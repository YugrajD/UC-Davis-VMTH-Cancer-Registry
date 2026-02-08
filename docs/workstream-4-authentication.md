# Workstream 4: Authentication & Access Control

[Back to Overview](../IMPLEMENTATION_PLAN.md)

---

**Gaps addressed:** #4 (Security requirement)

## 4.1 Database Migration: `database/migrations/009_users.sql`

```sql
-- 009_users.sql
-- User accounts for authentication

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'researcher'
        CHECK (role IN ('admin', 'researcher', 'viewer')),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Seed a default admin account (password: 'changeme' — MUST be changed on first login)
-- The hashed_password below is bcrypt hash of 'changeme'
-- In production, this should be set via environment variable or initial setup script
```

## 4.2 Backend — User Model

Add to `backend/app/models/models.py`:

```python
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(50), nullable=False, unique=True)
    email = Column(String(255), nullable=False, unique=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="researcher")
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
```

## 4.3 Auth Service: `backend/app/services/auth_service.py`

**Architecture:**

```python
"""
JWT-based authentication service.
Uses python-jose for JWT tokens and passlib for password hashing.
"""

from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import settings
from app.database import get_db
from app.models.models import User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def verify_password(plain: str, hashed: str) -> bool: ...
def hash_password(password: str) -> str: ...
def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str: ...

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """FastAPI dependency — extract and validate JWT, return User object.
    Raise 401 if token is invalid or user not found."""

async def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """FastAPI dependency — require admin role."""
```

**JWT token payload:**
```json
{
  "sub": "username",
  "role": "researcher",
  "exp": 1707400000
}
```

## 4.4 Auth Router: `backend/app/routers/auth.py`

**Endpoints:**

### `POST /api/v1/auth/login`

```
Request (form data, per OAuth2 spec):
  username: string
  password: string

Response 200:
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}

Response 401:
{
  "detail": "Incorrect username or password"
}
```

### `POST /api/v1/auth/register` (admin-only)

```
Request:
{
  "username": "jdoe",
  "email": "jdoe@ucdavis.edu",
  "password": "securepassword123",
  "role": "researcher"
}

Response 201:
{
  "id": 2,
  "username": "jdoe",
  "email": "jdoe@ucdavis.edu",
  "role": "researcher"
}

Response 403:
{
  "detail": "Only administrators can register new users"
}
```

### `GET /api/v1/auth/me`

```
Headers: Authorization: Bearer <token>

Response 200:
{
  "id": 1,
  "username": "admin",
  "email": "admin@ucdavis.edu",
  "role": "admin"
}
```

## 4.5 Protecting Existing Endpoints

Apply `Depends(get_current_user)` to these routes:

| Router | Endpoint | Protection |
|--------|----------|------------|
| `upload` | `POST /csv`, `POST /text`, `GET /history` | `get_current_user` |
| `review` | `GET /queue`, `PUT /{id}`, `GET /stats` | `get_current_user` |
| `auth` | `POST /register` | `require_admin` |
| All others | Dashboard, incidence, geo, trends, search | **Public** (read-only visualization) |

## 4.6 Configuration Additions

**`backend/app/config.py`** — add:

```python
SECRET_KEY: str = "CHANGE-ME-IN-PRODUCTION-use-openssl-rand-hex-32"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 480   # 8 hours
JWT_ALGORITHM: str = "HS256"
```

**`docker-compose.yml`** — add to backend environment:
```yaml
SECRET_KEY: "${SECRET_KEY:-dev-secret-key-change-in-production}"
```

## 4.7 Backend Requirements Additions

Add to `backend/requirements.txt`:

```
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
```

## 4.8 Frontend — Auth Context: `frontend/src/hooks/useAuth.ts`

```typescript
interface AuthContextType {
  user: UserInfo | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
}

// Store token in React state (memory), NOT localStorage
// This means token is lost on page refresh — acceptable for security
// Alternative: use httpOnly cookies (requires backend cookie support)
```

## 4.9 Frontend — Login Page: `frontend/src/components/LoginPage/LoginPage.tsx`

```
┌──────────────────────────────────────┐
│                                      │
│   UC Davis VMTH Cancer Registry      │
│                                      │
│   ┌──────────────────────────────┐   │
│   │  Username                    │   │
│   │  ┌────────────────────────┐  │   │
│   │  │                        │  │   │
│   │  └────────────────────────┘  │   │
│   │  Password                    │   │
│   │  ┌────────────────────────┐  │   │
│   │  │                        │  │   │
│   │  └────────────────────────┘  │   │
│   │                              │   │
│   │  [Sign In]                   │   │
│   │                              │   │
│   │  Invalid credentials         │   │
│   └──────────────────────────────┘   │
│                                      │
└──────────────────────────────────────┘
```

## 4.10 Frontend — API Client Auth Integration

**`frontend/src/api/client.ts`** — modify `fetchJson`:

```typescript
let authToken: string | null = null;

export function setAuthToken(token: string | null) {
  authToken = token;
}

export function getAuthHeaders(): Record<string, string> {
  return authToken ? { 'Authorization': `Bearer ${authToken}` } : {};
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url, {
    headers: getAuthHeaders(),
  });
  if (response.status === 401) {
    // Trigger logout via event or callback
    throw new Error('Unauthorized');
  }
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  return response.json();
}

export async function login(username: string, password: string): Promise<{ access_token: string }> {
  const formData = new URLSearchParams();
  formData.append('username', username);
  formData.append('password', password);

  const response = await fetch('/api/v1/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: formData,
  });
  if (!response.ok) throw new Error('Invalid credentials');
  return response.json();
}
```

## 4.11 Navigation — Conditional Tabs

**`frontend/src/components/Navigation/Navigation.tsx`:**

The Upload and Review tabs should only appear when the user is authenticated. Pass `isAuthenticated` as a prop and conditionally render those tabs.

```tsx
interface NavigationProps {
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
  isAuthenticated: boolean;
  onLoginClick: () => void;
}

// Filter TABS to exclude 'upload' and 'review' when not authenticated
// Add a "Sign In" / "Sign Out" button in the top-right corner of the banner
```

## 4.12 Files Summary

**Files to create:**
| File | Purpose |
|------|---------|
| `database/migrations/009_users.sql` | Users table |
| `backend/app/routers/auth.py` | Auth endpoints |
| `backend/app/services/auth_service.py` | JWT + password logic |
| `frontend/src/components/LoginPage/LoginPage.tsx` | Login form |
| `frontend/src/hooks/useAuth.ts` | Auth React context |

**Files to modify:**
| File | Change |
|------|--------|
| `backend/app/main.py` | Register auth router |
| `backend/app/models/models.py` | Add User model |
| `backend/app/schemas/schemas.py` | Add User/Token schemas |
| `backend/app/config.py` | Add SECRET_KEY, JWT settings |
| `backend/requirements.txt` | Add python-jose, passlib |
| `backend/app/routers/upload.py` | Add `Depends(get_current_user)` |
| `frontend/src/api/client.ts` | Add auth headers, login function |
| `frontend/src/App.tsx` | Wrap in AuthProvider, render LoginPage |
| `frontend/src/components/Navigation/Navigation.tsx` | Conditional tabs, sign in/out |
| `frontend/src/components/index.ts` | Export LoginPage |
| `docker-compose.yml` | Mount migration 009, add SECRET_KEY env |
