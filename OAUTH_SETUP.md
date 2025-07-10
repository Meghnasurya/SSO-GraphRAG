# OAuth Setup Guide

This guide explains how to set up Single Sign-On (SSO) integration with GitHub and Jira for the Agentic RAG Test Generator.

## Features

With SSO enabled, users can:
- 🔐 Sign in with GitHub and Jira accounts
- 📁 Select specific repositories and projects to analyze
- 🤖 Automatically create embeddings from code and documentation
- 🕸️ Build knowledge graphs connecting tests, code, and requirements
- 🧠 Use intelligent test coverage analysis

## GitHub OAuth Setup

### 1. Create GitHub OAuth App

1. Go to GitHub Settings → Developer settings → OAuth Apps
2. Click "New OAuth App"
3. Fill in the application details:
   - **Application name**: `Agentic RAG Test Generator`
   - **Homepage URL**: `http://localhost:8000` (or your domain)
   - **Authorization callback URL**: `http://localhost:8000/auth/github/callback`
4. Click "Register application"
5. Note down the **Client ID** and **Client Secret**

### 2. Configure Environment Variables

Add to your `.env` file:
```env
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret
```

## Jira OAuth Setup

### 1. Create Atlassian OAuth 2.0 App

1. Go to [Atlassian Developer Console](https://developer.atlassian.com/console)
2. Click "Create" → "OAuth 2.0 integration"
3. Enter app details:
   - **App name**: `Agentic RAG Test Generator`
   - **Company**: Your organization name
4. Configure permissions:
   - Add **Jira API** with scopes: `read:jira-work`, `read:jira-user`
5. Set authorization callback URL: `http://localhost:8000/auth/jira/callback`
6. Note down the **Client ID** and **Client Secret**

### 2. Configure Environment Variables

Add to your `.env` file:
```env
JIRA_OAUTH_CLIENT_ID=your_atlassian_oauth_client_id
JIRA_OAUTH_CLIENT_SECRET=your_atlassian_oauth_client_secret
```

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Copy `.env.example` to `.env` and fill in your OAuth credentials:

```bash
cp .env.example .env
# Edit .env with your actual credentials
```

### 3. Run the Application

```bash
python main.py
```

Or using uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Usage Flow

### 1. Access SSO Setup

1. Open http://localhost:8000
2. Click on the "SSO Setup" tab
3. Connect your GitHub and/or Jira accounts

### 2. Repository Selection (GitHub)

1. After GitHub authentication, you'll be redirected to repository selection
2. Choose repositories you want to analyze
3. Click "Process Selected Repositories"
4. The system will:
   - Extract code, documentation, and test files
   - Create embeddings for semantic search
   - Build knowledge graph relationships

### 3. Project Selection (Jira)

1. After Jira authentication, you'll be redirected to project selection
2. Choose Jira projects containing test cases
3. Click "Process Selected Projects"
4. The system will:
   - Fetch test cases from selected projects
   - Create embeddings for test case content
   - Build knowledge graph relationships

### 4. Use Enhanced Features

Once data is processed, you can:
- Use "Smart Test Check" for intelligent coverage analysis
- Generate tests with repository context
- Query across connected code and test data

## Security Considerations

### For Production Deployment:

1. **Use HTTPS**: Ensure all OAuth redirects use HTTPS URLs
2. **Secure Session Management**: Replace in-memory sessions with Redis or database-backed sessions
3. **Environment Variables**: Store OAuth secrets securely (e.g., AWS Secrets Manager, Azure Key Vault)
4. **Token Refresh**: Implement token refresh logic for long-lived sessions
5. **Rate Limiting**: Add rate limiting to prevent API abuse

### Session Security:

```python
# Add to main.py for production
from fastapi.middleware.sessions import SessionMiddleware
import redis

# Use Redis for session storage
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
```

## Troubleshooting

### Common Issues:

1. **OAuth Redirect Mismatch**: Ensure callback URLs match exactly in OAuth app settings
2. **CORS Issues**: Configure CORS middleware if frontend is on different domain
3. **Token Expiry**: Implement token refresh or re-authentication flow
4. **Rate Limiting**: GitHub and Jira have API rate limits - implement backoff strategies

### Debug Mode:

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## API Endpoints

New OAuth endpoints:

- `GET /auth/github` - Initiate GitHub OAuth
- `GET /auth/github/callback` - GitHub OAuth callback
- `GET /auth/jira` - Initiate Jira OAuth  
- `GET /auth/jira/callback` - Jira OAuth callback
- `GET /auth/status` - Check authentication status
- `GET /repositories` - Repository selection page
- `GET /projects` - Project selection page
- `POST /process-repositories` - Process selected repositories
- `POST /process-projects` - Process selected projects
- `GET /logout` - Clear session and logout

## Data Processing

### Repository Processing:
- Scans code files (`.py`, `.js`, `.ts`, `.java`, etc.)
- Extracts documentation (`.md`, `.rst`, `.txt`)
- Identifies test files (files with `test` in path or name)
- Creates embeddings for semantic search
- Builds knowledge graph with code relationships

### Project Processing:
- Fetches test cases from Jira projects
- Extracts test summaries and descriptions
- Creates embeddings for test content
- Builds knowledge graph with test relationships

The processed data enhances the RAG system's ability to provide intelligent test recommendations and coverage analysis.
