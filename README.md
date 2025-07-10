# RAG Test Generator with SSO

A FastAPI-based application for generating intelligent test cases using RAG (Retrieval Augmented Generation) with GitHub and Jira SSO integration.

## 🚀 Quick Start

### Option 1: Using the Batch File (Windows)
```bash
# Double-click or run from command prompt
start_servers.bat
```

### Option 2: Manual Start

#### 1. Start Backend Server (Port 8000)
```bash
python main.py
```

#### 2. Start Frontend Server (Port 3000) 
```bash
cd frontend
python serve.py
```

## 🔗 Application URLs

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🔧 Configuration

### Environment Variables

Make sure your `.env` file contains:

```env
# OAuth Configuration
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret
JIRA_CLIENT_ID=your_jira_client_id
JIRA_CLIENT_SECRET=your_jira_client_secret
SECRET_KEY=your_secret_key

# Database Configuration
NEO4J_URI=your_neo4j_uri
NEO4J_USER=your_neo4j_user
NEO4J_PASSWORD=your_neo4j_password

# Vector Database
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key

# AI Configuration
GEMINI_API_KEY=your_gemini_api_key
```

### OAuth App Configuration

#### GitHub OAuth App
- **Authorization callback URL**: `http://localhost:8000/auth/github/callback`
- **Homepage URL**: `http://localhost:3000`

#### Jira OAuth App
- **Callback URL**: `http://localhost:8000/auth/jira/callback`
- **Scopes**: `read:jira-user`, `read:jira-work`

## 📋 Usage Flow

1. **Authentication**: Sign in with GitHub or Jira
2. **Select Resources**: Choose repositories (GitHub) or projects (Jira)
3. **Create Embeddings**: Generate embeddings from selected resources
4. **Generate Tests**: Create intelligent test cases using RAG

## 🛠️ Development

### Project Structure
```
├── main.py                 # FastAPI backend server
├── frontend/
│   ├── index.html         # Frontend application
│   └── serve.py           # Frontend HTTP server
├── templates/
│   └── index.html         # Original integrated template
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
└── start_servers.bat     # Quick start script
```

### Key Features

- **OAuth Integration**: GitHub and Jira SSO
- **Session Management**: Secure session handling
- **CORS Support**: Frontend-backend communication
- **Vector Embeddings**: Qdrant integration
- **Graph Database**: Neo4j for relationship mapping
- **AI Generation**: Gemini for test case creation

## 🔍 Troubleshooting

### Common Issues

1. **OAuth "Invalid state parameter" error**
   - Ensure backend runs on port 8000 (not 5005)
   - Check OAuth app callback URLs match exactly

2. **CORS errors**
   - Verify frontend runs on port 3000
   - Check CORS middleware is properly configured

3. **Session issues**
   - Clear browser cookies and try again
   - Check session debug endpoint: `/debug/session`

4. **Port conflicts**
   - Ensure ports 3000 and 8000 are available
   - Use `netstat -an | findstr ":3000"` to check port usage

### Debug Endpoints

- `/auth/status` - Check authentication status
- `/debug/session` - Inspect session data
- `/data-status` - Check embeddings and graph data

## 📚 API Documentation

Visit http://localhost:8000/docs for interactive API documentation.

## 🔐 Security Notes

- Sessions are stored in memory (restart clears sessions)
- OAuth tokens are stored in session cookies
- Use HTTPS in production
- Rotate SECRET_KEY regularly

## 📞 Support

For issues and questions, check:
1. Environment variable configuration
2. OAuth app settings
3. Port availability
4. Debug endpoints for session state
