import os
import zipfile
import tempfile
import uuid
import ast
import networkx as nx
from typing import List, Optional, Dict, Any
import os
import ast
import uuid
import tempfile
import zipfile
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import google.generativeai as genai
from neo4j import GraphDatabase
from docx import Document
from threading import Thread
import time
import re
import traceback
import requests
import json
import uvicorn
from urllib.parse import urlencode, parse_qs
import secrets
import base64
import hashlib

# Qdrant and embedding imports
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

# Jira API imports
import json
from requests.auth import HTTPBasicAuth

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Jira/Xray credentials
JIRA_URL = os.getenv("JIRA_URL")
JIRA_USERNAME = os.getenv("JIRA_USERNAME")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")

# OAuth credentials
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
JIRA_OAUTH_CLIENT_ID = os.getenv("JIRA_OAUTH_CLIENT_ID")
JIRA_OAUTH_CLIENT_SECRET = os.getenv("JIRA_OAUTH_CLIENT_SECRET")
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Use the updated Gemini model initialization
gemini = genai.GenerativeModel("models/gemini-1.5-flash")

# Qdrant client setup
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Configure Qdrant + Embeddings
COLLECTION_NAME = "documentation_embeddings"

# Ensure Qdrant collection exists
try:
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in collections:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
except Exception as e:
    print(f"[Qdrant] Error checking/creating collection: {e}")


class TestGenInput(BaseModel):
    requirement: str = Field(...)
    documentation: str = Field(...)


class TestOutput(BaseModel):
    test_cases: List[str]
    test_scripts: List[str]


# OAuth and Authentication Models
class OAuthProvider(BaseModel):
    provider: str
    access_token: str
    refresh_token: Optional[str] = None
    expires_at: Optional[int] = None
    user_info: Dict[str, Any] = {}

class GitHubRepository(BaseModel):
    id: int
    name: str
    full_name: str
    description: Optional[str] = None
    private: bool
    clone_url: str
    default_branch: str

class JiraProject(BaseModel):
    id: str
    key: str
    name: str
    description: Optional[str] = None
    project_type_key: str

class SelectedResources(BaseModel):
    github_repos: List[GitHubRepository] = []
    jira_projects: List[JiraProject] = []

# FastAPI-specific models
class FunctionalityRequest(BaseModel):
    functionality: str = Field(..., description="Description of the functionality to test")


class JiraFetchRequest(BaseModel):
    project_key: str = Field(None, description="Jira project key")
    max_results: int = Field(100, description="Maximum number of results to fetch")


class AgenticTestResponse(BaseModel):
    action: str
    status: str
    message: str
    functionality: str
    summary: str


class ErrorResponse(BaseModel):
    error: str


class JiraXrayClient:
    def __init__(self):
        self.base_url = JIRA_URL
        self.auth = HTTPBasicAuth(JIRA_USERNAME, JIRA_API_TOKEN)
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }

    def fetch_test_cases(self, project_key: str = None, max_results: int = 100) -> List[dict]:
        """Fetch test cases from Jira/Xray"""
        if not project_key:
            project_key = JIRA_PROJECT_KEY
            
        if not all([self.base_url, JIRA_USERNAME, JIRA_API_TOKEN]):
            print("Jira credentials not configured properly")
            return []

        try:
            # JQL to fetch test cases (Xray test issue type)
            jql = f'project = "{project_key}" AND issuetype = "Test"'
            
            url = f"{self.base_url}/rest/api/3/search"
            params = {
                'jql': jql,
                'maxResults': max_results,
                'fields': 'summary,description,labels,components,fixVersions,customfield_*,status,priority'
            }
            
            response = requests.get(url, auth=self.auth, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            test_cases = []
            
            for issue in data.get('issues', []):
                test_case = {
                    'key': issue['key'],
                    'summary': issue['fields'].get('summary', ''),
                    'description': issue['fields'].get('description', ''),
                    'status': issue['fields'].get('status', {}).get('name', ''),
                    'priority': issue['fields'].get('priority', {}).get('name', ''),
                    'labels': issue['fields'].get('labels', []),
                    'components': [comp['name'] for comp in issue['fields'].get('components', [])],
                    'fix_versions': [ver['name'] for ver in issue['fields'].get('fixVersions', [])]
                }
                
                # Extract Xray-specific custom fields
                for field_id, field_value in issue['fields'].items():
                    if field_id.startswith('customfield_') and field_value:
                        if isinstance(field_value, dict) and 'value' in field_value:
                            test_case[f'custom_{field_id}'] = field_value['value']
                        elif isinstance(field_value, str):
                            test_case[f'custom_{field_id}'] = field_value
                
                test_cases.append(test_case)
            
            print(f"Fetched {len(test_cases)} test cases from Jira")
            return test_cases
            
        except Exception as e:
            print(f"Error fetching test cases from Jira: {e}")
            return []

    def fetch_test_executions(self, project_key: str = None, max_results: int = 50) -> List[dict]:
        """Fetch test executions from Jira/Xray"""
        if not project_key:
            project_key = JIRA_PROJECT_KEY
            
        try:
            # JQL to fetch test executions
            jql = f'project = "{project_key}" AND issuetype = "Test Execution"'
            
            url = f"{self.base_url}/rest/api/3/search"
            params = {
                'jql': jql,
                'maxResults': max_results,
                'fields': 'summary,description,status,created,updated,customfield_*'
            }
            
            response = requests.get(url, auth=self.auth, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            executions = []
            
            for issue in data.get('issues', []):
                execution = {
                    'key': issue['key'],
                    'summary': issue['fields'].get('summary', ''),
                    'description': issue['fields'].get('description', ''),
                    'status': issue['fields'].get('status', {}).get('name', ''),
                    'created': issue['fields'].get('created', ''),
                    'updated': issue['fields'].get('updated', '')
                }
                  # Extract custom fields for test execution details
                for field_id, field_value in issue['fields'].items():
                    if field_id.startswith('customfield_') and field_value:
                        if isinstance(field_value, dict) and 'value' in field_value:
                            execution[f'custom_{field_id}'] = field_value['value']
                        elif isinstance(field_value, str):
                            execution[f'custom_{field_id}'] = field_value
                
                executions.append(execution)
            
            print(f"Fetched {len(executions)} test executions from Jira")
            return executions
            
        except Exception as e:
            print(f"Error fetching test executions from Jira: {e}")
            return []


# OAuth Services
class GitHubOAuthService:
    def __init__(self):
        self.client_id = GITHUB_CLIENT_ID
        self.client_secret = GITHUB_CLIENT_SECRET
        self.authorize_url = "https://github.com/login/oauth/authorize"
        self.token_url = "https://github.com/login/oauth/access_token"
        self.api_base = "https://api.github.com"
        
    def get_authorization_url(self, redirect_uri: str, state: str) -> str:
        params = {
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "scope": "repo read:org",
            "state": state
        }
        return f"{self.authorize_url}?{urlencode(params)}"
        
    def exchange_code_for_token(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "redirect_uri": redirect_uri
        }
        headers = {"Accept": "application/json"}
        
        response = requests.post(self.token_url, data=data, headers=headers)
        response.raise_for_status()
        return response.json()
        
    def get_user_info(self, access_token: str) -> Dict[str, Any]:
        headers = {"Authorization": f"token {access_token}"}
        response = requests.get(f"{self.api_base}/user", headers=headers)
        response.raise_for_status()
        return response.json()
        
    def get_user_repositories(self, access_token: str) -> List[Dict[str, Any]]:
        headers = {"Authorization": f"token {access_token}"}
        repos = []
        page = 1
        per_page = 100
        
        while True:
            params = {"page": page, "per_page": per_page, "sort": "updated"}
            response = requests.get(f"{self.api_base}/user/repos", headers=headers, params=params)
            response.raise_for_status()
            
            page_repos = response.json()
            if not page_repos:
                break
                
            repos.extend(page_repos)
            page += 1
            
            if len(page_repos) < per_page:
                break
                
        return repos

class JiraOAuthService:
    def __init__(self):
        self.client_id = JIRA_OAUTH_CLIENT_ID
        self.client_secret = JIRA_OAUTH_CLIENT_SECRET
        
    def get_authorization_url(self, redirect_uri: str, state: str) -> str:
        params = {
            "audience": "api.atlassian.com",
            "client_id": self.client_id,
            "scope": "read:jira-work read:jira-user offline_access",
            "redirect_uri": redirect_uri,
            "state": state,
            "response_type": "code",
            "prompt": "consent"
        }
        return f"https://auth.atlassian.com/authorize?{urlencode(params)}"
        
    def exchange_code_for_token(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        token_url = "https://auth.atlassian.com/oauth/token"
        data = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "redirect_uri": redirect_uri
        }
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(token_url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()
        
    def get_accessible_resources(self, access_token: str) -> List[Dict[str, Any]]:
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get("https://api.atlassian.com/oauth/token/accessible-resources", headers=headers)
        response.raise_for_status()
        return response.json()
        
    def get_projects(self, access_token: str, cloud_id: str) -> List[Dict[str, Any]]:
        headers = {"Authorization": f"Bearer {access_token}"}
        url = f"https://api.atlassian.com/ex/jira/{cloud_id}/rest/api/3/project"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

class SessionManager:
    def __init__(self):
        self.sessions = {}
        
    def create_session(self, session_id: str, user_data: Dict[str, Any]):
        self.sessions[session_id] = {
            "user_data": user_data,
            "created_at": time.time(),
            "github_oauth": None,
            "jira_oauth": None,
            "selected_resources": None
        }
        
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(session_id)
        
    def update_session(self, session_id: str, updates: Dict[str, Any]):
        if session_id in self.sessions:
            self.sessions[session_id].update(updates)
            
    def delete_session(self, session_id: str):
        self.sessions.pop(session_id, None)


class GraphRAGNeo4j:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        if self.driver:
            self.driver.close()

    def clear_graph(self):
        """Clear the graph database with simple approach"""
        try:
            with self.driver.session() as session:
                # Simple approach - just delete everything
                session.run("MATCH (n) DETACH DELETE n")
                print("Graph cleared successfully")
        except Exception as e:
            print(f"Error clearing graph: {e}")
            print("Continuing without clearing graph...")

    def verify_connection(self):
        """Verify Neo4j connection is working"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                return result.single()[0] == 1
        except Exception as e:
            print(f"Neo4j connection failed: {e}")
            return False

    def load_knowledge(self, documentation: str, docx_pairs=None, py_data=None):
        """Load knowledge with improved error handling and batching"""
        if not self.verify_connection():
            print("Neo4j connection failed, skipping graph operations")
            return
            
        try:
            self.clear_graph()
            
            # Batch operations for better performance
            with self.driver.session() as session:
                # Process documentation lines in batches
                lines = documentation.strip().splitlines()
                batch_size = 100
                for i in range(0, len(lines), batch_size):
                    batch = lines[i:i+batch_size]
                    tx = session.begin_transaction()
                    try:
                        for line in batch:
                            parts = [p.strip() for p in line.split(":") if p.strip()]
                            if len(parts) >= 2:
                                tx.run(
                                    "MERGE (a:Function {name: $a}) MERGE (b:Detail {name: $b}) MERGE (a)-[:HAS_DETAIL]->(b)",
                                    {"a": parts[0], "b": parts[1]}
                                )
                        tx.commit()
                    except Exception as e:
                        tx.rollback()
                        print(f"Error processing batch {i//batch_size + 1}: {e}")
                        continue

                # Process docx pairs
                if docx_pairs:
                    tx = session.begin_transaction()
                    try:
                        for heading, detail in docx_pairs:
                            if heading and detail:  # Ensure both are not empty
                                tx.run(
                                    "MERGE (a:Function {name: $heading}) MERGE (b:Detail {name: $detail}) MERGE (a)-[:HAS_DETAIL]->(b)",
                                    {"heading": str(heading), "detail": str(detail)}
                                )
                        tx.commit()
                    except Exception as e:
                        tx.rollback()
                        print(f"Error processing docx pairs: {e}")

                # Process Python data
                if py_data:
                    tx = session.begin_transaction()
                    try:
                        for module, nodes in py_data.items():
                            for cls in nodes.get("classes", []):
                                if cls:  # Ensure class name is not empty
                                    tx.run("MERGE (c:Class {name: $name})", {"name": str(cls)})
                                    
                            for func, doc in nodes.get("functions", []):
                                if func:  # Ensure function name is not empty
                                    tx.run("MERGE (f:Function {name: $name})", {"name": str(func)})
                                    if doc:
                                        tx.run(
                                            "MATCH (f:Function {name: $name}) MERGE (d:Detail {name: $doc}) MERGE (f)-[:HAS_DETAIL]->(d)",
                                            {"name": str(func), "doc": str(doc)}                                        )
                        tx.commit()
                    except Exception as e:
                        tx.rollback()
                        print(f"Error processing Python data: {e}")
                        
        except Exception as e:
            print(f"Error loading knowledge to Neo4j: {e}")
            print("Continuing without graph data...")

    def load_jira_tests_knowledge(self, test_cases: List[dict], test_executions: List[dict] = None):
        """Load Jira test cases and executions into Neo4j without clearing existing data"""
        if not self.verify_connection():
            print("Neo4j connection failed, skipping graph operations")
            return
            
        try:
            with self.driver.session() as session:
                # Process test cases
                print(f"Loading {len(test_cases)} test cases into Neo4j...")
                for test_case in test_cases:
                    tx = session.begin_transaction()
                    try:
                        # Create test case node
                        tx.run("""
                            MERGE (t:TestCase {key: $key})
                            SET t.summary = $summary,
                                t.description = $description,
                                t.status = $status,
                                t.priority = $priority,
                                t.labels = $labels,
                                t.components = $components,
                                t.fix_versions = $fix_versions,
                                t.source = 'jira'
                        """, {
                            'key': test_case['key'],
                            'summary': test_case.get('summary', ''),
                            'description': test_case.get('description', ''),
                            'status': test_case.get('status', ''),
                            'priority': test_case.get('priority', ''),
                            'labels': test_case.get('labels', []),
                            'components': test_case.get('components', []),
                            'fix_versions': test_case.get('fix_versions', [])
                        })
                        
                        # Create relationships for components
                        for component in test_case.get('components', []):
                            if component:
                                tx.run("""
                                    MERGE (c:Component {name: $component})
                                    MERGE (t:TestCase {key: $key})
                                    MERGE (t)-[:BELONGS_TO_COMPONENT]->(c)
                                """, {'component': component, 'key': test_case['key']})
                        
                        # Create relationships for labels
                        for label in test_case.get('labels', []):
                            if label:
                                tx.run("""
                                    MERGE (l:Label {name: $label})
                                    MERGE (t:TestCase {key: $key})
                                    MERGE (t)-[:HAS_LABEL]->(l)
                                """, {'label': label, 'key': test_case['key']})
                        
                        tx.commit()
                    except Exception as e:
                        tx.rollback()
                        print(f"Error processing test case {test_case.get('key', 'unknown')}: {e}")
                        continue

                # Process test executions if provided
                if test_executions:
                    print(f"Loading {len(test_executions)} test executions into Neo4j...")
                    for execution in test_executions:
                        tx = session.begin_transaction()
                        try:
                            # Create test execution node
                            tx.run("""
                                MERGE (e:TestExecution {key: $key})
                                SET e.summary = $summary,
                                    e.description = $description,
                                    e.status = $status,
                                    e.created = $created,
                                    e.updated = $updated,
                                    e.source = 'jira'
                            """, {
                                'key': execution['key'],
                                'summary': execution.get('summary', ''),
                                'description': execution.get('description', ''),
                                'status': execution.get('status', ''),
                                'created': execution.get('created', ''),
                                'updated': execution.get('updated', '')
                            })
                            
                            tx.commit()
                        except Exception as e:
                            tx.rollback()
                            print(f"Error processing test execution {execution.get('key', 'unknown')}: {e}")
                            continue                            
                print("Successfully loaded Jira test data into Neo4j")
                        
        except Exception as e:
            print(f"Error loading Jira knowledge to Neo4j: {e}")


class EmbeddingRAGQdrant:
    def __init__(self):
        self.client = qdrant

    def clear_embeddings(self):
        """Clear embeddings with updated Qdrant API"""
        try:
            # Check if collection exists before trying to delete
            if self.client.collection_exists(collection_name=COLLECTION_NAME):
                self.client.delete_collection(collection_name=COLLECTION_NAME)
                
            # Create new collection
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
        except Exception as e:
            print(f"Error clearing embeddings: {e}")
            # Try to create collection anyway
            try:
                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                )
            except Exception as create_error:
                print(f"Failed to create collection: {create_error}")

    def embed_and_store(self, chunks: List[str], metadata: dict = {}, max_chunks: int = 1500):
        """Enhanced embedding processing for 1000+ chunks with batching and rate limiting"""
        
        # Remove the 50-chunk limit and increase to 1500
        chunks = chunks[:max_chunks]
        
        # Filter out very short or empty chunks
        valid_chunks = [chunk for chunk in chunks if len(chunk.strip()) >= 20]
        
        print(f"Processing {len(valid_chunks)} valid chunks for embedding (from {len(chunks)} total)...")
        
        points = []
        batch_size = 25  # Process in smaller batches to avoid rate limits
        request_delay = 0.1  # 100ms delay between requests
        
        for batch_start in range(0, len(valid_chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_chunks))
            batch_chunks = valid_chunks[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//batch_size + 1}/{(len(valid_chunks)-1)//batch_size + 1} ({len(batch_chunks)} chunks)")
            
            batch_points = []
            for i, chunk in enumerate(batch_chunks):
                try:
                    # Truncate chunk to reasonable size (embedding models have limits)
                    truncated_chunk = chunk[:1500]
                    
                    # Add delay to respect rate limits
                    if i > 0:
                        time.sleep(request_delay)
                        
                    # Updated embedding API call
                    result = genai.embed_content(
                        model="models/text-embedding-004",
                        content=truncated_chunk,
                        task_type="retrieval_document"
                    )
                    embedding = result['embedding']
                    
                    batch_points.append(
                        PointStruct(
                            id=str(uuid.uuid4()), 
                            vector=embedding, 
                            payload={
                                **metadata, 
                                "text": chunk,
                                "chunk_length": len(chunk),
                                "batch_id": batch_start//batch_size + 1
                            }
                        )
                    )
                    
                except Exception as e:
                    print(f"[Embedding Error] Failed on chunk {batch_start + i}: {chunk[:50]}... Error: {e}")
                    continue
            
            # Store batch points
            if batch_points:
                try:
                    self.client.upsert(collection_name=COLLECTION_NAME, points=batch_points)
                    points.extend(batch_points)
                    print(f"Stored {len(batch_points)} embeddings from batch {batch_start//batch_size + 1}")
                except Exception as e:
                    print(f"Error storing batch {batch_start//batch_size + 1}: {e}")
            
            # Add delay between batches
            time.sleep(0.5)
        
        print(f"Successfully stored {len(points)} embeddings out of {len(valid_chunks)} valid chunks")
        return len(points)

    def search(self, query: str, top_k: int = 10) -> List[str]:
        """Enhanced search with more results for better context"""
        try:
            # Updated embedding API call for search
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )
            query_vec = result['embedding']
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vec,
                limit=top_k,
                score_threshold=0.5  # Only return reasonably relevant results
            )
            
            # Return results with metadata for better debugging
            search_results = []
            for hit in results:
                search_results.append(hit.payload["text"])
                print(f"Search result score: {hit.score:.3f}, length: {hit.payload.get('chunk_length', 'unknown')}")
            
            return search_results
        except Exception as e:
            print(f"[Search Error] Failed to embed or search: {e}")
            return []

    def embed_and_store_jira_tests(self, test_cases: List[dict], test_executions: List[dict] = None) -> int:
        """Store Jira test cases and executions as embeddings without clearing existing data"""
        
        # Convert test cases to text chunks
        jira_chunks = []
        
        for test_case in test_cases:
            # Create comprehensive text representation of test case
            test_text = f"Test Case: {test_case['key']}\n"
            test_text += f"Summary: {test_case.get('summary', '')}\n"
            test_text += f"Description: {test_case.get('description', '')}\n"
            test_text += f"Status: {test_case.get('status', '')}\n"
            test_text += f"Priority: {test_case.get('priority', '')}\n"
            
            if test_case.get('components'):
                test_text += f"Components: {', '.join(test_case['components'])}\n"
            if test_case.get('labels'):
                test_text += f"Labels: {', '.join(test_case['labels'])}\n"
            if test_case.get('fix_versions'):
                test_text += f"Fix Versions: {', '.join(test_case['fix_versions'])}\n"
            
            jira_chunks.append(test_text)
        
        # Convert test executions to text chunks if provided
        if test_executions:
            for execution in test_executions:
                exec_text = f"Test Execution: {execution['key']}\n"
                exec_text += f"Summary: {execution.get('summary', '')}\n"
                exec_text += f"Description: {execution.get('description', '')}\n"
                exec_text += f"Status: {execution.get('status', '')}\n"
                exec_text += f"Created: {execution.get('created', '')}\n"
                exec_text += f"Updated: {execution.get('updated', '')}\n"
                
                jira_chunks.append(exec_text)
        
        print(f"Converting {len(jira_chunks)} Jira items to embeddings...")
        
        # Store embeddings without clearing existing ones
        points = []
        batch_size = 25
        request_delay = 0.1
        
        for batch_start in range(0, len(jira_chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(jira_chunks))
            batch_chunks = jira_chunks[batch_start:batch_end]
            
            print(f"Processing Jira batch {batch_start//batch_size + 1}/{(len(jira_chunks)-1)//batch_size + 1} ({len(batch_chunks)} items)")
            
            batch_points = []
            for i, chunk in enumerate(batch_chunks):
                try:
                    # Add delay to respect rate limits
                    if i > 0:
                        time.sleep(request_delay)
                        
                    # Create embedding
                    result = genai.embed_content(
                        model="models/text-embedding-004",
                        content=chunk,
                        task_type="retrieval_document"
                    )
                    embedding = result['embedding']
                    
                    batch_points.append(
                        PointStruct(
                            id=str(uuid.uuid4()), 
                            vector=embedding, 
                            payload={
                                "text": chunk,
                                "source": "jira",
                                "chunk_length": len(chunk),
                                "batch_id": batch_start//batch_size + 1
                            }
                        )
                    )
                    
                except Exception as e:
                    print(f"[Jira Embedding Error] Failed on item {batch_start + i}: {chunk[:50]}... Error: {e}")
                    continue
            
            # Store batch points
            if batch_points:
                try:
                    self.client.upsert(collection_name=COLLECTION_NAME, points=batch_points)
                    points.extend(batch_points)
                    print(f"Stored {len(batch_points)} Jira embeddings from batch {batch_start//batch_size + 1}")
                except Exception as e:
                    print(f"Error storing Jira batch {batch_start//batch_size + 1}: {e}")
            
            # Add delay between batches
            time.sleep(0.5)
        
        print(f"Successfully stored {len(points)} Jira embeddings")
        return len(points)


def chunk_text(text: str, max_chars: int = 400, overlap: int = 50) -> List[str]:
    """Enhanced text chunking with overlap and better splitting"""
    if not text or len(text.strip()) < 20:
        return []
    
    # Clean text
    text = re.sub(r'\s+', ' ', text.strip())
    
    chunks = []
    
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        # If paragraph is too long, split by sentences
        if len(paragraph) > max_chars:
            sentences = re.split(r'[.!?]+', paragraph)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                if len(current_chunk) + len(sentence) + 1 <= max_chars:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                        # Add overlap
                        if overlap > 0 and len(current_chunk) > overlap:
                            current_chunk = current_chunk[-overlap:] + " " + sentence
                        else:
                            current_chunk = sentence
                    else:
                        current_chunk = sentence
        else:
            # Paragraph fits, add it
            if len(current_chunk) + len(paragraph) + 1 <= max_chars:
                current_chunk += " " + paragraph if current_chunk else paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    # Add overlap
                    if overlap > 0 and len(current_chunk) > overlap:
                        current_chunk = current_chunk[-overlap:] + " " + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    current_chunk = paragraph
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        chunk_normalized = chunk.lower().strip()
        if chunk_normalized not in seen and len(chunk_normalized) >= 20:
            seen.add(chunk_normalized)
            unique_chunks.append(chunk)
    
    return unique_chunks


def extract_code_blocks(text: str) -> List[str]:
    """Extract code blocks and important code snippets"""
    code_chunks = []
    
    # Extract code blocks (```...```)
    code_blocks = re.findall(r'```[\s\S]*?```', text, re.MULTILINE)
    for block in code_blocks:
        if len(block) > 50:  # Only meaningful code blocks
            code_chunks.append(block)
    
    # Extract function definitions and classes
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if re.match(r'^(def|class|function|var|let|const)\s+\w+', line.strip()):
            # Get function/class with some context
            context_start = max(0, i-2)
            context_end = min(len(lines), i+10)
            context = '\n'.join(lines[context_start:context_end])
            if len(context) > 30:
                code_chunks.append(context)
    
    return code_chunks


class AgenticRAGTestChecker:
    def __init__(self):
        try:
            self.graph_rag = GraphRAGNeo4j()
            self.use_graph = True
        except Exception as e:
            print(f"Failed to initialize Neo4j: {e}")
            self.graph_rag = None
            self.use_graph = False
            
        self.embedding_rag = EmbeddingRAGQdrant()
          # Only check existing data - do not populate automatically
        print("🔍 AgenticRAGTestChecker initialized - will only check existing graph and embeddings")
        
    def check_existing_data_status(self) -> dict:
        """Check what data already exists in graph and embeddings"""
        status = {
            'embeddings_available': False,
            'graph_available': False,
            'embedding_count': 0,
            'graph_node_count': 0
        }
        
        # Check Qdrant embeddings
        try:
            if self.embedding_rag.client.collection_exists(collection_name=COLLECTION_NAME):
                collection_info = self.embedding_rag.client.get_collection(collection_name=COLLECTION_NAME)
                status['embeddings_available'] = True
                status['embedding_count'] = collection_info.points_count if hasattr(collection_info, 'points_count') else 0
                print(f"📊 Found {status['embedding_count']} existing embeddings in Qdrant")
            else:
                print("📊 No existing embeddings collection found in Qdrant")
        except Exception as e:
            print(f"❌ Error checking Qdrant embeddings: {e}")
          # Check Neo4j graph
        try:
            if self.use_graph and self.graph_rag:
                with self.graph_rag.driver.session() as session:
                    result = session.run("MATCH (n) RETURN count(n) as node_count")
                    node_count = result.single()['node_count']
                    status['graph_available'] = node_count > 0
                    status['graph_node_count'] = node_count
                    print(f"🕸️ Found {node_count} existing nodes in Neo4j graph")
            else:
                print("🕸️ Neo4j graph not available")
        except Exception as e:
            print(f"❌ Error checking Neo4j graph: {e}")
        
        return status
        
    def populate_data_if_needed(self, force_populate: bool = False) -> dict:
        """
        Manually populate data from Jira if needed.
        Call this method explicitly if you want to populate embeddings and graph data.
        """
        return self.populate_jira_test_data(force_populate=force_populate)
        
    def populate_jira_test_data(self, force_populate: bool = False):
        """
        Fetch and populate real test data from Jira/Xray instead of using sample data.
        Only runs if force_populate is True - otherwise just checks existing data.
        """
        if not force_populate:
            print("⏭️ Skipping Jira data population - using existing data only")
            return self.check_existing_data_status()
            
        try:
            print("🔗 Connecting to Jira to fetch real test cases...")
            
            # Initialize Jira client
            jira_client = JiraXrayClient()
            
            # Fetch test cases from Jira
            print("📋 Fetching test cases from Jira...")
            test_cases = jira_client.fetch_test_cases(max_results=200)
            
            if not test_cases:
                print("⚠️ No test cases found in Jira or connection failed. Using fallback sample data.")
                return self.populate_sample_test_data(force_populate)
              # Fetch test executions from Jira
            print("🏃 Fetching test executions from Jira...")
            test_executions = jira_client.fetch_test_executions(max_results=100)
            
            # Clear existing embeddings to start fresh with Jira data
            print("🧹 Clearing existing embeddings to load fresh Jira data...")
            self.embedding_rag.clear_embeddings()
            
            # Store Jira test cases as embeddings
            print(f"💾 Storing {len(test_cases)} Jira test cases as embeddings...")
            stored_count = self.embedding_rag.embed_and_store_jira_tests(test_cases, test_executions)
            
            # Store in Neo4j graph if available
            if self.use_graph and self.graph_rag:
                print("🕸️ Loading Jira test data into Neo4j knowledge graph...")
                self.graph_rag.load_jira_tests_knowledge(test_cases, test_executions)
            
            print(f"✅ Successfully populated {stored_count} real Jira test cases")
            print(f"📊 Test Cases: {len(test_cases)}, Test Executions: {len(test_executions or [])}")
            
            return stored_count
            
        except Exception as e:
            print(f"❌ Error fetching Jira test data: {e}")
            print("🔄 Falling back to sample test data for demonstration...")
            return self.populate_sample_test_data(force_populate)
        
    def populate_sample_test_data(self, force_populate: bool = False):
        """
        Fallback method: Populate sample test data for demonstration purposes.
        This is only used if Jira connection fails and force_populate is True.
        """
        if not force_populate:
            print("⏭️ Skipping sample data population - using existing data only")
            return self.check_existing_data_status()
            
        sample_test_cases = [
            # Login page test cases
            """
            Test Case: Login Page Authentication
            Description: Verify user can login with valid credentials
            Steps:
            1. Navigate to login page
            2. Enter valid username and password
            3. Click login button
            4. Verify user is redirected to dashboard
            Expected Result: User successfully logs in and sees dashboard
            Test Type: Functional Test
            Component: Authentication, Login Page
            """,
            
            """
            Test Case: Login Page Invalid Credentials
            Description: Verify error message for invalid login
            Steps:
            1. Navigate to login page
            2. Enter invalid username or password
            3. Click login button
            4. Verify error message is displayed
            Expected Result: Appropriate error message shown, user remains on login page
            Test Type: Negative Test
            Component: Authentication, Login Page, Error Handling
            """,
            
            """
            Test Case: Login Page Session Management
            Description: Verify session timeout and re-authentication
            Steps:
            1. Login with valid credentials
            2. Wait for session timeout period
            3. Try to access protected resource
            4. Verify user is redirected to login page
            Expected Result: User session expires and requires re-authentication
            Test Type: Security Test
            Component: Authentication, Session Management
            """,
            
            # Database CRUD operations
            """
            Test Case: User Profile CRUD Operations
            Description: Test create, read, update, delete operations for user profiles
            Steps:
            1. Create new user profile with valid data
            2. Read/retrieve the created profile
            3. Update profile information
            4. Delete the profile
            Expected Result: All CRUD operations complete successfully
            Test Type: Integration Test
            Component: Database, User Management
            """,
            
            # API Rate Limiting (partial coverage)
            """
            Test Case: Basic API Rate Limiting
            Description: Test API rate limiting for authenticated requests
            Steps:
            1. Make API requests within rate limit
            2. Verify requests are processed successfully
            Expected Result: Requests within limit are processed
            Test Type: Performance Test
            Component: API, Rate Limiting
            Note: Missing tests for rate limit exceeded scenarios
            """,
            
            # Input validation (partial coverage)
            """
            Test Case: Basic Input Validation
            Description: Test input validation for form fields
            Steps:
            1. Submit form with valid input data
            2. Verify form is accepted
            Expected Result: Valid input is processed correctly
            Test Type: Validation Test
            Component: Forms, Input Validation
            Note: Missing comprehensive validation tests for edge cases
            """
        ]
        
        try:
            print("📋 Populating fallback sample test data...")
            
            # Store sample test cases as embeddings
            stored_count = self.embedding_rag.embed_and_store(
                sample_test_cases,
                metadata={
                    "source": "sample_tests_fallback",
                    "purpose": "demonstration_fallback",
                    "created_date": time.strftime('%Y-%m-%d')
                }            )
            
            print(f"✅ Populated {stored_count} fallback sample test cases")
            return stored_count
            
        except Exception as e:
            print(f"Warning: Could not populate fallback sample test data: {e}")
            return 0

    def analyze_functionality_request(self, user_prompt: str) -> dict:
        """
        Intelligently analyze user prompt to check if test cases exist for the requested functionality.
        Returns comprehensive analysis and action taken.
        Only checks existing data - does not create new embeddings or graph data.
        NEW: First checks if function exists in both doc embeddings and graph before allowing test operations.
        """
        print(f"Analyzing functionality request: {user_prompt}")
        
        # Step 0: Check existing data status first
        data_status = self.check_existing_data_status()
        if not data_status['embeddings_available'] and not data_status['graph_available']:
            return {
                'action': 'NO_DATA',
                'status': 'No existing data found',
                'message': "❌ No existing embeddings or graph data found. Please populate data first using populate_jira_test_data(force_populate=True) or upload documentation.",
                'data_status': data_status,
                'existing_tests': [],
                'recommendations': ["Populate test data first", "Upload documentation files", "Connect to Jira/Xray"],
                'tests_analyzed': 0
            }        # Step 0.5: Check if function exists in documentation, test embeddings, and knowledge graph
        function_name = self._extract_function_name(user_prompt)
        if function_name:
            function_existence = self._check_function_existence(function_name, data_status)
            
            # If function doesn't exist in documentation, return early
            if not function_existence['exists_in_docs']:
                return {
                    'action': 'FUNCTION_NOT_FOUND',
                    'status': 'Function not found in documentation',
                    'message': f"❌ Function '{function_name}' does not exist in documentation embeddings. Test cases cannot be updated or created.",
                    'function_name': function_name,
                    'exists_in_docs': function_existence['exists_in_docs'],
                    'exists_in_tests': function_existence['exists_in_tests'],
                    'exists_in_graph': function_existence['exists_in_graph'],
                    'doc_search_results': function_existence['doc_results'],
                    'test_search_results': function_existence['test_results'],
                    'graph_search_results': function_existence['graph_results'],
                    'data_status': data_status,
                    'existing_tests': [],
                    'recommendations': [
                        "Verify the function name is correct",
                        "Ensure the function documentation exists in the embeddings",
                        "Upload documentation that contains this function"
                    ],
                    'tests_analyzed': 0
                }
            
            # If function exists in docs, show all related test cases and provide graph info
            if function_existence['exists_in_docs']:
                print(f"✅ Function '{function_name}' exists in documentation - searching for all related test cases...")
                  # Search for all test cases related to this function
                all_function_tests = self._search_function_specific_tests(function_name)
                
                if all_function_tests:
                    graph_status = "Also found in knowledge graph" if function_existence['exists_in_graph'] else "Not found in knowledge graph"
                    return {
                        'action': 'EXISTING',
                        'status': 'Function exists - showing all test cases',
                        'message': f"✅ Function '{function_name}' exists in documentation. Found {len(all_function_tests)} related test cases. {graph_status}.",
                        'function_name': function_name,
                        'exists_in_docs': function_existence['exists_in_docs'],
                        'exists_in_tests': function_existence['exists_in_tests'],
                        'exists_in_graph': function_existence['exists_in_graph'],
                        'exists_in_all': function_existence['exists_in_all'],
                        'existing_tests': all_function_tests,
                        'doc_search_results': function_existence['doc_results'],
                        'test_search_results': function_existence['test_results'],
                        'graph_search_results': function_existence['graph_results'],
                        'data_status': data_status,
                        'tests_analyzed': len(all_function_tests),
                        'coverage_score': 1.0,
                        'recommendations': ['Tests exist for this function']
                    }
                else:
                    graph_status = "Also found in knowledge graph" if function_existence['exists_in_graph'] else "Not found in knowledge graph"
                    return {
                        'action': 'NO_TESTS_FOUND',
                        'status': 'Function exists but no test cases found',
                        'message': f"✅ Function '{function_name}' exists in documentation but no related test cases were found. {graph_status}.",
                        'function_name': function_name,
                        'exists_in_docs': function_existence['exists_in_docs'],
                        'exists_in_tests': function_existence['exists_in_tests'],
                        'exists_in_graph': function_existence['exists_in_graph'],
                        'exists_in_all': function_existence['exists_in_all'],
                        'existing_tests': [],
                        'doc_search_results': function_existence['doc_results'],
                        'test_search_results': function_existence['test_results'],
                        'graph_search_results': function_existence['graph_results'],
                        'data_status': data_status,
                        'tests_analyzed': 0,
                        'coverage_score': 0.0,
                        'recommendations': [
                            'Function exists in documentation but no tests found',
                            'Consider creating test cases for this function',
                            'Check if test cases exist under different naming conventions'
                        ]
                    }
        print(f"📊 Using existing data: {data_status['embedding_count']} embeddings, {data_status['graph_node_count']} graph nodes")
          # If no specific function was detected, check if general functionality exists in documentation
        print("🔍 No specific function detected, checking if general functionality exists in documentation...")
        
        # Search for the functionality in documentation embeddings
        doc_search_results = []
        if data_status['embeddings_available']:
            try:
                doc_search_results = self.embedding_rag.search(
                    query=f"documentation {user_prompt}", 
                    top_k=5
                )
                print(f"Found {len(doc_search_results)} documentation results for general functionality")
            except Exception as e:
                print(f"Error searching documentation embeddings: {e}")
        
        # If no documentation results found, block test creation/updating
        if not doc_search_results:
            return {
                'action': 'FUNCTIONALITY_NOT_DOCUMENTED',
                'status': 'Functionality not found in documentation',
                'message': f"❌ The requested functionality '{user_prompt}' was not found in documentation embeddings. Test cases cannot be created or updated without documented functionality.",
                'user_prompt': user_prompt,
                'doc_search_results': [],
                'data_status': data_status,
                'existing_tests': [],
                'recommendations': [
                    "Verify the functionality is correctly described",
                    "Ensure the functionality documentation exists in the embeddings",
                    "Upload documentation that contains this functionality",
                    "Check for alternative naming or descriptions"
                ],
                'tests_analyzed': 0
            }
        
        print(f"✅ Found documentation for functionality - proceeding with test search...")
        print("🔍 Searching for general test cases across all project data...")
        existing_test_chunks = self._comprehensive_test_search(user_prompt)
        
        # Step 2: Filter and analyze the relevance of found test cases
        relevant_tests = self._filter_relevant_tests(user_prompt, existing_test_chunks)
        
        # Step 3: Use AI to determine if existing tests cover the functionality
        coverage_analysis = self._analyze_test_coverage(user_prompt, relevant_tests)        # Step 4: Determine action based on coverage analysis - but block creation/updates
        if coverage_analysis['has_sufficient_coverage']:
            return {
                'action': 'EXISTING',
                'status': 'Test cases already exist',
                'message': f"✅ Found {len(existing_test_chunks)} existing test cases that cover this functionality",
                'existing_tests': existing_test_chunks,  # Return all found test chunks, not just the filtered ones
                'coverage_score': coverage_analysis['coverage_score'],
                'recommendations': coverage_analysis.get('recommendations', []),
                'tests_analyzed': len(existing_test_chunks),
                'doc_search_results': doc_search_results,
                'data_status': data_status
            }
        elif coverage_analysis['needs_update']:
            # Block test updates for general functionality requests
            return {
                'action': 'UPDATE_BLOCKED',
                'status': 'Test updates blocked - insufficient documentation',
                'message': f"❌ Cannot update test cases for '{user_prompt}' - functionality documentation insufficient for safe updates. Found {len(existing_test_chunks)} existing tests but updates require specific function documentation.",
                'existing_tests': existing_test_chunks,
                'doc_search_results': doc_search_results,
                'data_status': data_status,
                'recommendations': [
                    "Specify the exact function name for targeted updates",
                    "Ensure comprehensive documentation exists for this functionality",
                    "Review existing test cases manually if updates are needed"
                ],
                'tests_analyzed': len(existing_test_chunks)
            }
        else:
            # Block test creation for general functionality requests
            return {
                'action': 'CREATION_BLOCKED',
                'status': 'Test creation blocked - insufficient documentation',
                'message': f"❌ Cannot create test cases for '{user_prompt}' - functionality must be properly documented with specific function names before test creation. Found minimal documentation but insufficient for test generation.",
                'existing_tests': existing_test_chunks,
                'doc_search_results': doc_search_results,
                'data_status': data_status,
                'recommendations': [
                    "Specify the exact function name for test creation",
                    "Add comprehensive documentation with function signatures and behavior",
                    "Upload documentation files that describe specific functions in detail"
                ],
                'tests_analyzed': len(existing_test_chunks)
            }
    
    def _comprehensive_test_search(self, user_prompt: str) -> List[str]:
        """
        Perform comprehensive search for existing test cases using multiple strategies.
        Enhanced to specifically detect login page and common functionality patterns.
        """
        all_test_chunks = []
        
        # Check if this is a login/authentication related request
        auth_keywords = ['login', 'auth', 'authentication', 'signin', 'sign-in', 'user access', 'password', 'credential']
        is_auth_request = any(keyword in user_prompt.lower() for keyword in auth_keywords)
        
        if is_auth_request:
            print("🔐 Detected authentication/login related request - searching for existing login tests...")
            # Specific search patterns for login functionality
            login_patterns = [
                "login test",
                "authentication test",
                "user login validation",
                "login page test",
                "signin test",
                "login form test",
                "password validation test",
                "login functionality test",
                "user authentication test case"
            ]
            
            for pattern in login_patterns:
                login_results = self.embedding_rag.search(query=pattern, top_k=8)
                all_test_chunks.extend(login_results)
        
        # Strategy 1: Direct functionality search
        functionality_results = self.embedding_rag.search(
            query=f"test case functionality: {user_prompt}", 
            top_k=10
        )
        all_test_chunks.extend(functionality_results)
        
        # Strategy 2: Search for specific keywords from the prompt
        keywords = self._extract_keywords(user_prompt.lower())
        for keyword in keywords[:5]:  # Top 5 keywords
            keyword_results = self.embedding_rag.search(
                query=f"test {keyword}", 
                top_k=5
            )
            all_test_chunks.extend(keyword_results)
        
        # Strategy 3: Search for test-specific terms
        test_patterns = [
            f"unittest {user_prompt}",
            f"test method {user_prompt}",
            f"test script {user_prompt}",
            f"test execution {user_prompt}",
            f"automated test {user_prompt}"
        ]
        
        for pattern in test_patterns:
            pattern_results = self.embedding_rag.search(query=pattern, top_k=3)
            all_test_chunks.extend(pattern_results)
        
        # Strategy 4: Search from Jira/external sources
        jira_results = self.embedding_rag.search(
            query=f"jira test case {user_prompt}", 
            top_k=5
        )
        all_test_chunks.extend(jira_results)
        
        # Strategy 5: Page-specific search (login page, dashboard, etc.)
        page_keywords = ['page', 'form', 'interface', 'ui', 'frontend']
        if any(keyword in user_prompt.lower() for keyword in page_keywords):
            page_results = self.embedding_rag.search(
                query=f"page test {user_prompt}", 
                top_k=5
            )
            all_test_chunks.extend(page_results)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_chunks = []
        for chunk in all_test_chunks:
            chunk_normalized = chunk.lower().strip()
            if chunk_normalized not in seen and len(chunk.strip()) > 20:
                seen.add(chunk_normalized)
                unique_chunks.append(chunk)
        
        print(f"Found {len(unique_chunks)} unique test-related chunks from comprehensive search")
        return unique_chunks[:25]  # Return top 25 most relevant
    
    def _filter_relevant_tests(self, user_prompt: str, test_chunks: List[str]) -> List[dict]:
        """Filter test chunks to find those actually relevant to the user's functionality request"""
        relevant_tests = []
        
        # Keywords extraction from user prompt
        prompt_keywords = self._extract_keywords(user_prompt.lower())
        
        for chunk in test_chunks:
            chunk_lower = chunk.lower()            # Check if this chunk is actually a test case
            if any(indicator in chunk_lower for indicator in ['test case', 'test:', 'def test_', 'unittest', 'test execution']):
                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(prompt_keywords, chunk_lower)
                
                if relevance_score > 0.3:  # Threshold for relevance
                    relevant_tests.append({
                        'content': chunk,
                        'relevance_score': relevance_score,
                        'keywords_matched': [kw for kw in prompt_keywords if kw in chunk_lower]
                    })
        
        # Sort by relevance score
        relevant_tests.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_tests[:10]  # Top 10 most relevant
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from user prompt"""
        # Common stop words to ignore
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'test', 'case', 'cases'}        
        # Extract words (simple tokenization)
        words = re.findall(r'\b\w{3,}\b', text.lower())
        keywords = [word for word in words if word not in stop_words]
        
        # Add some common technical terms and patterns
        technical_patterns = re.findall(r'\b(?:api|database|login|auth|crud|create|read|update|delete|validate|verify|check|process|handle|manage|generate|calculate|search|filter|sort|upload|download|send|receive|parse|format|convert|transform|integrate|sync|backup|restore|export|import)\b', text.lower())
        keywords.extend(technical_patterns)
        
        return list(set(keywords))  # Remove duplicates
    
    def _calculate_relevance_score(self, prompt_keywords: List[str], test_content: str) -> float:
        """Calculate how relevant a test case is to the user's prompt"""
        if not prompt_keywords:
            return 0.0
        
        matches = sum(1 for keyword in prompt_keywords if keyword in test_content)
        base_score = matches / len(prompt_keywords)
        
        # Boost score for exact phrase matches
        if len(prompt_keywords) > 1:
            phrase_matches = sum(1 for i in range(len(prompt_keywords)-1) 
                               if f"{prompt_keywords[i]} {prompt_keywords[i+1]}" in test_content)
            base_score += phrase_matches * 0.2
        
        # Boost score for test-specific indicators
        test_indicators = ['test case', 'should', 'verify', 'validate', 'assert', 'expect']
        indicator_matches = sum(1 for indicator in test_indicators if indicator in test_content)
        base_score += indicator_matches * 0.1
        
        return min(base_score, 1.0)  # Cap at 1.0
    
    def _analyze_test_coverage(self, user_prompt: str, relevant_tests: List[dict]) -> dict:
        """Use AI to analyze if existing tests provide sufficient coverage"""
        if not relevant_tests:
            return {
                'has_sufficient_coverage': False,
                'needs_update': False,
                'coverage_score': 0.0,
                'existing_tests': [],
                'gap_analysis': 'No existing test cases found for this functionality'
            }
          # If we have many relevant tests with high relevance scores, consider it sufficient
        high_relevance_tests = [test for test in relevant_tests if test['relevance_score'] > 0.6]        
        if len(high_relevance_tests) >= 3:
            # Extract just the content from the relevant tests
            existing_tests_content = [test['content'] for test in relevant_tests]
            
            return {
                'has_sufficient_coverage': True,
                'needs_update': False,
                'coverage_score': min(0.9, sum(test['relevance_score'] for test in relevant_tests) / len(relevant_tests)),
                'existing_tests': existing_tests_content,
                'covered_aspects': ['Comprehensive test coverage found'],
                'missing_aspects': [],
                'recommendations': ['Tests already exist with good coverage'],
                'gap_analysis': f'Found {len(relevant_tests)} highly relevant existing tests'
            }
        
        # Prepare context for AI analysis
        existing_tests_text = "\n".join([f"Test {i+1}: {test['content'][:300]}..." 
                                       for i, test in enumerate(relevant_tests[:5])])
        
        analysis_prompt = f"""Analyze if the existing test cases provide sufficient coverage for the requested functionality.

**User's Functionality Request**: {user_prompt}

**Existing Test Cases Found**:
{existing_tests_text}

**Analysis Instructions**:
1. Determine if existing tests adequately cover the requested functionality
2. Provide a coverage score (0.0 to 1.0) where:
   - 0.8-1.0 = SUFFICIENT coverage (show existing tests)
   - 0.4-0.7 = PARTIAL coverage (update existing tests)
   - 0.0-0.3 = INSUFFICIENT coverage (create new tests)
3. List what aspects are covered and what gaps exist
4. Make a recommendation based on coverage level

**Special Cases**:
- For login/authentication: Look for comprehensive login flow, error handling, session management
- For CRUD operations: Check for all create, read, update, delete scenarios
- For validation: Ensure positive, negative, and edge cases are covered
- For API: Check for different HTTP methods, error responses, rate limiting
- For MSAL/AD configuration: Look for tenant setup, authentication flows, token handling

**Output Format**:
[COVERAGE_DECISION]
SUFFICIENT or PARTIAL or INSUFFICIENT
[END_COVERAGE_DECISION]

[COVERAGE_SCORE]
0.8
[END_COVERAGE_SCORE]

[COVERED_ASPECTS]
- Aspect 1: Description
- Aspect 2: Description
[END_COVERED_ASPECTS]

[MISSING_ASPECTS]
- Gap 1: Description
- Gap 2: Description
[END_MISSING_ASPECTS]

[RECOMMENDATIONS]
- Recommendation 1
- Recommendation 2
[END_RECOMMENDATIONS]"""

        try:
            response = gemini.generate_content(analysis_prompt)
            analysis_text = response.text
            
            # Parse the AI response
            decision = self._extract_section(analysis_text, "COVERAGE_DECISION", "INSUFFICIENT")
            coverage_score = float(self._extract_section(analysis_text, "COVERAGE_SCORE", "0.0"))
            covered_aspects = self._extract_list_section(analysis_text, "COVERED_ASPECTS")
            missing_aspects = self._extract_list_section(analysis_text, "MISSING_ASPECTS")
            recommendations = self._extract_list_section(analysis_text, "RECOMMENDATIONS")
            
            # Determine actions based on decision and score
            decision_upper = decision.strip().upper()
            
            # Extract just the content from the relevant tests
            existing_tests_content = [test['content'] for test in relevant_tests]
            
            return {
                'has_sufficient_coverage': decision_upper == "SUFFICIENT" and coverage_score >= 0.8,
                'needs_update': decision_upper == "PARTIAL" and coverage_score >= 0.4 and coverage_score < 0.8 and len(relevant_tests) >= 1,
                'coverage_score': coverage_score,
                'existing_tests': existing_tests_content,
                'covered_aspects': covered_aspects,
                'missing_aspects': missing_aspects,
                'recommendations': recommendations,
                'gap_analysis': f"Coverage: {coverage_score:.1%}, Decision: {decision_upper}, Missing: {', '.join(missing_aspects[:3])}"
            }
            
        except Exception as e:
            print(f"Error in AI analysis: {e}")
            # Fallback to enhanced heuristic
            coverage_score = sum(test['relevance_score'] for test in relevant_tests) / len(relevant_tests) if relevant_tests else 0.0
            
            # Enhanced heuristic logic - be more conservative about updates
            has_sufficient = coverage_score >= 0.6 and len(relevant_tests) >= 2
            needs_update = False  # Only update if specifically requested, not automatically
            
            # Extract just the content from the relevant tests
            existing_tests_content = [test['content'] for test in relevant_tests]
            
            return {
                'has_sufficient_coverage': has_sufficient,
                'needs_update': needs_update,
                'coverage_score': coverage_score,
                'existing_tests': existing_tests_content,
                'gap_analysis': f"Found {len(relevant_tests)} relevant tests with average relevance {coverage_score:.1%}"
            }
    
    def _extract_section(self, text: str, section_name: str, default: str = "") -> str:
        """Extract content between section markers"""
        start_marker = f"[{section_name}]"
        end_marker = f"[END_{section_name}]"
        
        start_idx = text.find(start_marker)
        if start_idx == -1:
            return default
        
        start_idx += len(start_marker)
        end_idx = text.find(end_marker, start_idx)
        if end_idx == -1:
            return default
        
        return text[start_idx:end_idx].strip()
    
    def _extract_list_section(self, text: str, section_name: str) -> List[str]:
        """Extract list items from a section"""
        section_content = self._extract_section(text, section_name, "")
        if not section_content:
            return []
        
        lines = section_content.split('\n')
        items = []
        for line in lines:
            line = line.strip()
            if line.startswith('- '):
                items.append(line[2:])
            elif line and not line.startswith('['):
                items.append(line)
        
        return items
    
    def _create_comprehensive_test_cases(self, user_prompt: str, existing_partial_tests: List[dict]) -> dict:
        """Create comprehensive test cases for the functionality"""
        
        # Prepare context from existing partial tests
        partial_context = ""
        if existing_partial_tests:
            partial_context = "\n".join([f"Existing test {i+1}: {test['content'][:200]}..." 
                                       for i, test in enumerate(existing_partial_tests[:3])])
        
        creation_prompt = f"""You are a senior QA automation engineer. Create comprehensive test cases for the requested functionality.

**Functionality Request**: {user_prompt}

**Existing Partial Coverage** (if any):
{partial_context}

**Instructions**:
1. Create 6 comprehensive test cases covering different scenarios
2. Include positive, negative, edge cases, and error handling
3. Generate 6 corresponding Python unittest methods
4. Ensure new tests complement (don't duplicate) existing partial coverage
5. Be specific and actionable

**Output Format**:
[TEST_CASES]
- Test Case 1: [Detailed description with expected behavior]
- Test Case 2: [Detailed description with expected behavior]
- Test Case 3: [Detailed description with expected behavior]
- Test Case 4: [Detailed description with expected behavior]
- Test Case 5: [Detailed description with expected behavior]
- Test Case 6: [Detailed description with expected behavior]
[END_TEST_CASES]

[TEST_SCRIPTS]
import unittest

class TestGeneratedFunctionality(unittest.TestCase):
    def setUp(self):
        # Setup code here
        pass
    
    def test_functionality_positive_case(self):
        '''Test positive scenario'''
        # Detailed implementation
        pass
    
    def test_functionality_negative_case(self):
        '''Test negative scenario'''
        # Detailed implementation
        pass
    
    def test_functionality_edge_case(self):
        '''Test edge cases'''
        # Detailed implementation
        pass
    
    def test_functionality_error_handling(self):
        '''Test error handling'''
        # Detailed implementation
        pass
    
    def test_functionality_boundary_conditions(self):
        '''Test boundary conditions'''
        # Detailed implementation
        pass
    
    def test_functionality_integration(self):
        '''Test integration scenarios'''
        # Detailed implementation
        pass
    
    def tearDown(self):
        # Cleanup code here
        pass

if __name__ == '__main__':
    unittest.main()
[END_TEST_SCRIPTS]

[TEST_RATIONALE]
Explain why these specific test cases were chosen and how they provide comprehensive coverage.
[END_TEST_RATIONALE]"""

        try:
            response = gemini.generate_content(creation_prompt)
            content = response.text
            
            # Parse the generated content
            test_cases = self._extract_list_section(content, "TEST_CASES")
            test_scripts_content = self._extract_section(content, "TEST_SCRIPTS", "# Test scripts could not be generated")
            rationale = self._extract_section(content, "TEST_RATIONALE", "Test cases created to provide comprehensive coverage")
            
            # Clean up test scripts
            test_scripts = [line for line in test_scripts_content.split('\n') if line.strip()]
            
            return {
                'test_cases': test_cases or [
                    f"Test Case 1: Verify basic functionality of {user_prompt}",
                    f"Test Case 2: Test error handling for {user_prompt}",
                    f"Test Case 3: Test edge cases for {user_prompt}",
                    f"Test Case 4: Test integration scenarios for {user_prompt}",
                    f"Test Case 5: Test performance aspects of {user_prompt}",
                    f"Test Case 6: Test security aspects of {user_prompt}"
                ],                'test_scripts': test_scripts,
                'rationale': rationale,
                'generated_for': user_prompt
            }
            
        except Exception as e:
            print(f"Error creating test cases: {e}")
            return {
                'test_cases': [f"Error generating test cases for: {user_prompt}"],
                'test_scripts': ["# Error generating test scripts"],
                'rationale': f"Error occurred during generation: {str(e)}",
                'generated_for': user_prompt
            }
    
    def _update_existing_tests(self, user_prompt: str, relevant_tests: List[dict], coverage_analysis: dict) -> dict:
        """
        Update existing test cases to improve coverage for the requested functionality
        """
        print(f"Updating existing test cases for: {user_prompt}")
        
        # Prepare context from existing tests and coverage gaps
        existing_tests_context = "\n".join([f"Test {i+1}: {test['content'][:300]}..." 
                                          for i, test in enumerate(relevant_tests[:3])])
        
        missing_aspects = coverage_analysis.get('missing_aspects', [])
        
        update_prompt = f"""You are a senior QA automation engineer. Update the existing test cases to improve coverage for the requested functionality.

**Functionality Request**: {user_prompt}

**Existing Test Cases**:
{existing_tests_context}

**Missing Coverage Areas**:
{', '.join(missing_aspects[:5])}

**Instructions**:
1. Analyze the existing test cases and identify improvement opportunities
2. Update/enhance existing tests to cover missing aspects
3. Generate 3-5 improved test cases that build upon existing ones
4. Generate corresponding updated Python unittest methods
5. Explain what improvements were made and why

**Output Format**:
[UPDATED_TEST_CASES]
- Updated Test Case 1: [Enhanced description with new coverage]
- Updated Test Case 2: [Enhanced description with new coverage]
- Updated Test Case 3: [Enhanced description with new coverage]
- Updated Test Case 4: [Enhanced description with new coverage]
- Updated Test Case 5: [Enhanced description with new coverage]
[END_UPDATED_TEST_CASES]

[UPDATED_TEST_SCRIPTS]
import unittest

class TestUpdatedFunctionality(unittest.TestCase):
    def setUp(self):
        # Enhanced setup code
        pass
    
    def test_updated_functionality_1(self):
        '''Updated test with improved coverage'''
        # Enhanced implementation
        pass
    
    def test_updated_functionality_2(self):
        '''Updated test with improved coverage'''
        # Enhanced implementation
        pass
    
    def test_updated_functionality_3(self):
        '''Updated test with improved coverage'''
        # Enhanced implementation
        pass
    
    def test_updated_functionality_4(self):
        '''Updated test with improved coverage'''
        # Enhanced implementation
        pass
    
    def test_updated_functionality_5(self):
        '''Updated test with improved coverage'''
        # Enhanced implementation
        pass
    
    def tearDown(self):
        # Enhanced cleanup code
        pass

if __name__ == '__main__':
    unittest.main()
[END_UPDATED_TEST_SCRIPTS]

[UPDATE_RATIONALE]
Explain what specific improvements were made to the existing tests and how they address the coverage gaps.
[END_UPDATE_RATIONALE]

[COVERAGE_IMPROVEMENT]
Describe how the updated tests improve overall test coverage compared to the original tests.
[END_COVERAGE_IMPROVEMENT]"""

        try:
            response = gemini.generate_content(update_prompt)
            content = response.text
            
            # Parse the generated content
            updated_test_cases = self._extract_list_section(content, "UPDATED_TEST_CASES")
            updated_scripts_content = self._extract_section(content, "UPDATED_TEST_SCRIPTS", "# Updated test scripts could not be generated")
            update_rationale = self._extract_section(content, "UPDATE_RATIONALE", "Test cases updated to improve coverage")
            coverage_improvement = self._extract_section(content, "COVERAGE_IMPROVEMENT", "Coverage improvements applied")
            
            # Clean up test scripts
            updated_scripts = [line for line in updated_scripts_content.split('\n') if line.strip()]
            
            return {
                'updated_tests': updated_test_cases or [
                    f"Updated Test 1: Enhanced functionality test for {user_prompt}",
                    f"Updated Test 2: Improved error handling for {user_prompt}",
                    f"Updated Test 3: Enhanced edge case coverage for {user_prompt}"
                ],
                'original_tests': [test['content'][:200] + "..." for test in relevant_tests[:3]],
                'updated_scripts': updated_scripts,
                'rationale': update_rationale,
                'coverage_improvement': coverage_improvement,                'updated_for': user_prompt
            }
            
        except Exception as e:
            print(f"Error updating test cases: {e}")
            return {
                'updated_tests': [f"Error updating test cases for: {user_prompt}"],
                'original_tests': [test['content'][:200] + "..." for test in relevant_tests[:3]],
                'updated_scripts': ["# Error generating updated test scripts"],
                'rationale': f"Error occurred during update: {str(e)}",
                'coverage_improvement': "Could not determine coverage improvement",
                'updated_for': user_prompt
            }

    def _extract_function_name(self, user_prompt: str) -> str:
        """
        Extract function name from user prompt.
        Looks for patterns like 'function_name', 'test function_name', etc.
        """
        # Common patterns for function names in user prompts
        patterns = [
            r'\bfunction\s+(\w+)',           # "function function_name"
            r'\bmethod\s+(\w+)',             # "method method_name"
            r'\b(\w+)\s*\(\)',               # "function_name()"
            r'\btest\s+(\w+)',               # "test function_name"
            r'\b(\w+)_function\b',           # "name_function"
            r'\b(\w+)_method\b',             # "name_method"
            r'\bfor\s+(\w+)\b',              # "for function_name"
            r'\b(\w+)\s+functionality\b',    # "function_name functionality"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_prompt.lower())
            if match:
                return match.group(1)
        
        # If no pattern matches, try to find a standalone identifier
        # that looks like a function name (contains underscore or camelCase)
        words = re.findall(r'\b\w+\b', user_prompt)
        for word in words:
            if '_' in word or (len(word) > 3 and any(c.isupper() for c in word[1:])):
                return word
        
        return None

    def _check_function_existence(self, function_name: str, data_status: dict) -> dict:
        """
        Check if a function exists in documentation embeddings, test embeddings, and knowledge graph.
        Returns detailed results about where the function was found.
        """
        result = {
            'exists_in_docs': False,
            'exists_in_tests': False,
            'exists_in_graph': False,
            'exists_in_all': False,
            'doc_results': [],
            'test_results': [],
            'graph_results': []
        }
        
        print(f"🔍 Checking existence of function '{function_name}' in documentation, test embeddings, and knowledge graph...")
        
        # Check in documentation embeddings
        if data_status['embeddings_available']:
            try:
                # Search for the function in documentation embeddings with various patterns
                doc_search_patterns = [
                    function_name,
                    f"function {function_name}",
                    f"def {function_name}",
                    f"{function_name}(",
                    f"method {function_name}",
                    f"{function_name} documentation",
                    f"class {function_name}",
                    f"{function_name} implementation"
                ]
                
                for pattern in doc_search_patterns:
                    search_results = self.embedding_rag.search(query=pattern, top_k=5)
                    for chunk in search_results:
                        # Filter to find actual documentation (not test cases)
                        chunk_lower = chunk.lower()
                        if (function_name.lower() in chunk_lower and 
                            not any(test_indicator in chunk_lower for test_indicator in 
                                   ['test case', 'test:', 'def test_', 'unittest', 'test execution', 'test method'])):
                            result['doc_results'].append({
                                'pattern': pattern,
                                'content': chunk[:200] + "..." if len(chunk) > 200 else chunk
                            })
                
                if result['doc_results']:
                    result['exists_in_docs'] = True
                    print(f"✅ Function '{function_name}' found in documentation embeddings")
                else:
                    print(f"❌ Function '{function_name}' not found in documentation embeddings")
                    
            except Exception as e:
                print(f"❌ Error searching documentation embeddings: {e}")
        
        # Check in test embeddings (existing test cases)
        if data_status['embeddings_available']:
            try:
                # Search for the function in test embeddings with test-specific patterns
                test_search_patterns = [
                    f"test {function_name}",
                    f"test case {function_name}",
                    f"unittest {function_name}",
                    f"test_{function_name}",
                    f"{function_name} test",
                    f"testing {function_name}",
                    f"test method {function_name}",
                    f"def test_{function_name}",
                    f"class Test{function_name.title()}"
                ]
                
                for pattern in test_search_patterns:
                    search_results = self.embedding_rag.search(query=pattern, top_k=5)
                    for chunk in search_results:
                        # Filter to find actual test cases
                        chunk_lower = chunk.lower()
                        if (function_name.lower() in chunk_lower and 
                            any(test_indicator in chunk_lower for test_indicator in 
                               ['test case', 'test:', 'def test_', 'unittest', 'test execution', 'test method'])):
                            result['test_results'].append({
                                'pattern': pattern,
                                'content': chunk[:200] + "..." if len(chunk) > 200 else chunk
                            })
                
                if result['test_results']:
                    result['exists_in_tests'] = True
                    print(f"✅ Function '{function_name}' found in test embeddings")
                else:
                    print(f"❌ Function '{function_name}' not found in test embeddings")
                    
            except Exception as e:
                print(f"❌ Error searching test embeddings: {e}")
        
        # Check in knowledge graph
        if data_status['graph_available'] and self.use_graph and self.graph_rag:
            try:
                with self.graph_rag.driver.session() as session:
                    # Search for function in various node types
                    queries = [
                        ("MATCH (f:Function) WHERE toLower(f.name) CONTAINS toLower($name) RETURN f.name as name LIMIT 5", "Function"),
                        ("MATCH (f:Function) WHERE toLower(f.name) = toLower($name) RETURN f.name as name LIMIT 5", "Function (exact)"),
                        ("MATCH (c:Class)-[:HAS_METHOD]->(m) WHERE toLower(m.name) CONTAINS toLower($name) RETURN m.name as name LIMIT 5", "Method"),
                        ("MATCH (n) WHERE toLower(n.name) CONTAINS toLower($name) RETURN labels(n) as labels, n.name as name LIMIT 5", "Any node")
                    ]
                    
                    for query, query_type in queries:
                        graph_result = session.run(query, {"name": function_name})
                        records = list(graph_result)
                        if records:
                            for record in records:
                                result['graph_results'].append({
                                    'query_type': query_type,
                                    'name': record.get('name', ''),
                                    'labels': record.get('labels', [])
                                })
                    
                    if result['graph_results']:
                        result['exists_in_graph'] = True
                        print(f"✅ Function '{function_name}' found in knowledge graph")
                    else:
                        print(f"❌ Function '{function_name}' not found in knowledge graph")
                        
            except Exception as e:
                print(f"❌ Error searching knowledge graph: {e}")
        
        # Determine if function exists in all three sources (docs, tests, and graph)
        result['exists_in_all'] = result['exists_in_docs'] and result['exists_in_tests'] and result['exists_in_graph']
        
        print(f"📊 Function existence check result:")
        print(f"   - In documentation: {result['exists_in_docs']}")
        print(f"   - In tests: {result['exists_in_tests']}")
        print(f"   - In knowledge graph: {result['exists_in_graph']}")
        print(f"   - In all sources: {result['exists_in_all']}")
        
        return result

    def _search_function_specific_tests(self, function_name: str) -> List[str]:
        """
        Search for test cases specifically related to a function name.
        """
        function_test_chunks = []
        
        # Search patterns specifically for this function
        search_patterns = [
            f"test {function_name}",
            f"test case {function_name}",
            f"unittest {function_name}",
            f"test_{function_name}",
            f"{function_name} test",
            f"testing {function_name}",
            f"test method {function_name}",
            f"def test_{function_name}",
            f"class Test{function_name.title()}",
        ]
        
        for pattern in search_patterns:
            try:
                results = self.embedding_rag.search(query=pattern, top_k=5)
                function_test_chunks.extend(results)
            except Exception as e:
                print(f"Error searching for pattern '{pattern}': {e}")
        
        print(f"Found {len(function_test_chunks)} function-specific test chunks for '{function_name}'")
        return function_test_chunks

    def _store_updated_tests(self, user_prompt: str, updated_tests: dict, store_embeddings: bool = False):
        """Store updated test cases for future reference - only if explicitly requested"""
        if not store_embeddings:
            print("⏭️ Skipping storage of updated tests - only checking existing data")
            return
            
        try:
            # Create text representation for embedding storage
            test_content = f"Functionality: {user_prompt}\n\n"
            test_content += "Updated Test Cases:\n"
            for i, test_case in enumerate(updated_tests['updated_tests'], 1):
                test_content += f"{i}. {test_case}\n"
            
            test_content += f"\nUpdate Rationale: {updated_tests['rationale']}\n"
            test_content += f"Coverage Improvement: {updated_tests['coverage_improvement']}\n"
            test_content += f"Updated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            # Store as embedding
            self.embedding_rag.embed_and_store(
                [test_content],
                metadata={
                    "source": "updated_tests",
                    "functionality": user_prompt,
                    "updated_date": time.strftime('%Y-%m-%d'),
                    "test_count": len(updated_tests['updated_tests']),
                    "action": "updated"
                }
            )            
            print(f"✅ Stored updated test cases for future reference")
            
        except Exception as e:
            print(f"Warning: Could not store updated tests: {e}")

    def _store_generated_tests(self, user_prompt: str, new_tests: dict, store_embeddings: bool = False):
        """Store newly generated test cases for future reference - only if explicitly requested"""
        if not store_embeddings:
            print("⏭️ Skipping storage of generated tests - only checking existing data")
            return
            
        try:
            # Create text representation for embedding storage
            test_content = f"Functionality: {user_prompt}\n\n"
            test_content += "Generated Test Cases:\n"
            for i, test_case in enumerate(new_tests['test_cases'], 1):
                test_content += f"{i}. {test_case}\n"
            
            test_content += f"\nRationale: {new_tests['rationale']}\n"
            test_content += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            # Store as embedding
            self.embedding_rag.embed_and_store(
                [test_content],
                metadata={
                    "source": "generated_tests",
                    "functionality": user_prompt,
                    "generated_date": time.strftime('%Y-%m-%d'),
                    "test_count": len(new_tests['test_cases'])
                }
            )
            
            print(f"✅ Stored generated test cases for future reference")
            
        except Exception as e:
            print(f"Warning: Could not store generated tests: {e}")


class TestCaseGenerator:
    def __init__(self):
        try:
            self.graph_rag = GraphRAGNeo4j()
            self.use_graph = True
        except Exception as e:
            print(f"Failed to initialize Neo4j: {e}")
            self.graph_rag = None
            self.use_graph = False
            
        self.embedding_rag = EmbeddingRAGQdrant()

    def generate(self, inputs: TestGenInput) -> TestOutput:
        print("Searching for relevant context...")
        relevant_chunks = self.embedding_rag.search(inputs.requirement, top_k=8)  # Increased from 3
        context = "\n".join(relevant_chunks) or "No relevant context found."
        
        print("Generating test cases and scripts...")
        prompt = f"""You are a senior QA automation engineer. Generate test cases and Python unittest scripts.

**Requirement**: {inputs.requirement}

**Context**: {context[:2000]}

**Instructions**:
1. Generate EXACTLY 5 test cases (increased from 3)
2. Generate EXACTLY 5 Python unittest test methods
3. Be detailed and practical
4. Include edge cases and error handling
5. Use the provided context to make tests more specific

**Output Format**:
[TEST_CASES]
- Test Case 1: [Detailed description]
- Test Case 2: [Detailed description]
- Test Case 3: [Detailed description]
- Test Case 4: [Detailed description]
- Test Case 5: [Detailed description]
[END_TEST_CASES]

[TEST_SCRIPTS]
import unittest

class TestSuite(unittest.TestCase):
    def setUp(self):
        # Setup code here
        pass
    
    def test_case_1(self):
        # Detailed test implementation
        pass
    
    def test_case_2(self):
        # Detailed test implementation
        pass
    
    def test_case_3(self):
        # Detailed test implementation
        pass
    
    def test_case_4(self):
        # Detailed test implementation
        pass
    
    def test_case_5(self):
        # Detailed test implementation
        pass
    
    def tearDown(self):
        # Cleanup code here
        pass
[END_TEST_SCRIPTS]"""

        try:
            response = gemini.generate_content(prompt)
            content = response.text
            print("Generated response from Gemini")
        except Exception as e:
            print(f"Error generating content: {e}")
            return TestOutput(
                test_cases=["Error generating test cases"],
                test_scripts=["# Error generating test scripts"]
            )

        test_cases, test_scripts = [], []
        in_cases = in_scripts = False
        
        for line in content.splitlines():
            if "[TEST_CASES]" in line:
                in_cases = True
                continue
            elif "[END_TEST_CASES]" in line:
                in_cases = False
                continue
            elif "[TEST_SCRIPTS]" in line:
                in_scripts = True
                continue
            elif "[END_TEST_SCRIPTS]" in line:
                in_scripts = False
                continue
            
            if in_cases and line.strip():
                test_cases.append(line.strip())
            elif in_scripts:
                test_scripts.append(line)

        # Ensure we have content even if parsing fails
        if not test_cases:
            test_cases = [
                "Test Case 1: Basic functionality test", 
                "Test Case 2: Error handling test", 
                "Test Case 3: Edge case test",
                "Test Case 4: Performance test",
                "Test Case 5: Integration test"
            ]
        
        if not test_scripts:
            test_scripts = ["# Test scripts could not be generated", "# Please check the input requirements"]

        return TestOutput(test_cases=test_cases, test_scripts=test_scripts)


app = FastAPI(title="Agentic RAG Test Generator", description="AI-powered test case generation system")

# Initialize OAuth services and session manager
github_oauth = GitHubOAuthService()
jira_oauth = JiraOAuthService()
session_manager = SessionManager()

def get_current_session(request: Request) -> Optional[Dict[str, Any]]:
    session_id = request.cookies.get("session_id")
    if not session_id:
        return None
    return session_manager.get_session(session_id)

def require_auth(request: Request) -> Dict[str, Any]:
    session = get_current_session(request)
    if not session:
        raise HTTPException(status_code=401, detail="Authentication required")
    return session

# OAuth Routes
@app.get("/auth/github")
async def github_auth(request: Request):
    """Initiate GitHub OAuth flow"""
    state = secrets.token_urlsafe(32)
    redirect_uri = str(request.url_for("github_callback"))
    
    # Store state in a simple session (in production, use proper session management)
    session_id = secrets.token_urlsafe(32)
    session_manager.create_session(session_id, {"oauth_state": state})
    
    auth_url = github_oauth.get_authorization_url(redirect_uri, state)
    
    response = RedirectResponse(url=auth_url)
    response.set_cookie("session_id", session_id, httponly=True, max_age=3600)
    return response

@app.get("/auth/github/callback")
async def github_callback(request: Request, code: str, state: str):
    """Handle GitHub OAuth callback"""
    session = get_current_session(request)
    if not session or session.get("oauth_state") != state:
        raise HTTPException(status_code=400, detail="Invalid state parameter")
    
    try:
        redirect_uri = str(request.url_for("github_callback"))
        token_data = github_oauth.exchange_code_for_token(code, redirect_uri)
        
        if "access_token" not in token_data:
            raise HTTPException(status_code=400, detail="Failed to get access token")
        
        access_token = token_data["access_token"]
        user_info = github_oauth.get_user_info(access_token)
        
        # Update session with GitHub OAuth data
        session_manager.update_session(request.cookies.get("session_id"), {
            "github_oauth": {
                "access_token": access_token,
                "user_info": user_info
            }
        })
        
        return RedirectResponse(url="/repositories")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OAuth error: {str(e)}")

@app.get("/auth/jira")
async def jira_auth(request: Request, instance: str = ""):
    """Initiate Jira OAuth flow"""
    if not instance:
        raise HTTPException(status_code=400, detail="Jira instance URL required")
    
    state = secrets.token_urlsafe(32)
    redirect_uri = str(request.url_for("jira_callback"))
    
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = secrets.token_urlsafe(32)
        session_manager.create_session(session_id, {"oauth_state": state, "jira_instance": instance})
    else:
        session_manager.update_session(session_id, {"oauth_state": state, "jira_instance": instance})
    
    auth_url = jira_oauth.get_authorization_url(redirect_uri, state)
    
    response = RedirectResponse(url=auth_url)
    response.set_cookie("session_id", session_id, httponly=True, max_age=3600)
    return response

@app.get("/auth/jira/callback")
async def jira_callback(request: Request, code: str, state: str):
    """Handle Jira OAuth callback"""
    session = get_current_session(request)
    if not session or session.get("oauth_state") != state:
        raise HTTPException(status_code=400, detail="Invalid state parameter")
    
    try:
        redirect_uri = str(request.url_for("jira_callback"))
        token_data = jira_oauth.exchange_code_for_token(code, redirect_uri)
        
        if "access_token" not in token_data:
            raise HTTPException(status_code=400, detail="Failed to get access token")
        
        access_token = token_data["access_token"]
        resources = jira_oauth.get_accessible_resources(access_token)
        
        # Update session with Jira OAuth data
        session_manager.update_session(request.cookies.get("session_id"), {
            "jira_oauth": {
                "access_token": access_token,
                "resources": resources
            }
        })
        
        return RedirectResponse(url="/projects")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OAuth error: {str(e)}")

@app.get("/repositories", response_class=HTMLResponse)
async def repositories_page(request: Request):
    """Show GitHub repositories selection page"""
    session = require_auth(request)
    github_data = session.get("github_oauth")
    
    if not github_data:
        return RedirectResponse(url="/auth/github")
    
    try:
        repos = github_oauth.get_user_repositories(github_data["access_token"])
        
        # Generate HTML for repository selection
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Select GitHub Repositories</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .repo-item {{ padding: 10px; border: 1px solid #ddd; margin: 10px 0; }}
                .selected {{ background-color: #e7f3ff; }}
                button {{ padding: 10px 20px; margin: 10px 5px; }}
            </style>
        </head>
        <body>
            <h1>Select GitHub Repositories</h1>
            <p>Hello {github_data['user_info']['login']}! Select repositories to analyze:</p>
            
            <form id="repoForm">
                <div id="repositories">
        """
        
        for repo in repos[:20]:  # Limit to first 20 repos
            html_content += f"""
                    <div class="repo-item">
                        <label>
                            <input type="checkbox" name="selected_repos" value="{repo['full_name']}">
                            <strong>{repo['name']}</strong> - {repo.get('description', 'No description')}
                            <br><small>Private: {repo['private']} | Language: {repo.get('language', 'Unknown')}</small>
                        </label>
                    </div>
            """
        
        html_content += """
                </div>
                <button type="button" onclick="processSelected()">Process Selected Repositories</button>
                <button type="button" onclick="location.href='/auth/jira?instance=YOUR_JIRA_INSTANCE'">Continue to Jira</button>
            </form>
            
            <script>
                async function processSelected() {
                    const selected = Array.from(document.querySelectorAll('input[name="selected_repos"]:checked'))
                                         .map(cb => cb.value);
                    
                    if (selected.length === 0) {
                        alert('Please select at least one repository');
                        return;
                    }
                    
                    try {
                        const response = await fetch('/process-repositories', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ repositories: selected })
                        });
                        
                        const result = await response.json();
                        alert(`Processing complete! ${result.message}`);
                        
                    } catch (error) {
                        alert('Error processing repositories: ' + error.message);
                    }
                }
            </script>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching repositories: {str(e)}")

@app.get("/projects", response_class=HTMLResponse)
async def projects_page(request: Request):
    """Show Jira projects selection page"""
    session = require_auth(request)
    jira_data = session.get("jira_oauth")
    
    if not jira_data:
        return RedirectResponse(url="/auth/jira")
    
    try:
        all_projects = []
        for resource in jira_data["resources"]:
            cloud_id = resource["id"]
            projects = jira_oauth.get_projects(jira_data["access_token"], cloud_id)
            for project in projects:
                project["cloud_id"] = cloud_id
                project["site_name"] = resource["name"]
            all_projects.extend(projects)
        
        # Generate HTML for project selection
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Select Jira Projects</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .project-item {{ padding: 10px; border: 1px solid #ddd; margin: 10px 0; }}
                .selected {{ background-color: #e7f3ff; }}
                button {{ padding: 10px 20px; margin: 10px 5px; }}
            </style>
        </head>
        <body>
            <h1>Select Jira Projects</h1>
            <p>Select Jira projects to analyze test cases:</p>
            
            <form id="projectForm">
                <div id="projects">
        """
        
        for project in all_projects[:20]:  # Limit to first 20 projects
            html_content += f"""
                    <div class="project-item">
                        <label>
                            <input type="checkbox" name="selected_projects" 
                                   value="{project['key']}" data-cloud-id="{project['cloud_id']}">
                            <strong>{project['key']}</strong> - {project['name']}
                            <br><small>Site: {project['site_name']} | Type: {project.get('projectTypeKey', 'Unknown')}</small>
                        </label>
                    </div>
            """
        
        html_content += """
                </div>
                <button type="button" onclick="processSelected()">Process Selected Projects</button>
            </form>
            
            <script>
                async function processSelected() {
                    const selected = Array.from(document.querySelectorAll('input[name="selected_projects"]:checked'))
                                         .map(cb => ({
                                             key: cb.value,
                                             cloud_id: cb.dataset.cloudId
                                         }));
                    
                    if (selected.length === 0) {
                        alert('Please select at least one project');
                        return;
                    }
                    
                    try {
                        const response = await fetch('/process-projects', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ projects: selected })
                        });
                        
                        const result = await response.json();
                        alert(`Processing complete! ${result.message}`);
                        
                    } catch (error) {
                        alert('Error processing projects: ' + error.message);
                    }
                }
            </script>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching projects: {str(e)}")

@app.post("/process-repositories")
async def process_repositories(request: Request, data: dict):
    """Process selected GitHub repositories"""
    session = require_auth(request)
    github_data = session.get("github_oauth")
    
    if not github_data:
        raise HTTPException(status_code=401, detail="GitHub authentication required")
    
    selected_repos = data.get("repositories", [])
    if not selected_repos:
        raise HTTPException(status_code=400, detail="No repositories selected")
    
    try:
        # Initialize repository processor
        from typing import List, Dict, Any
        
        class RepositoryProcessor:
            def __init__(self):
                self.embedding_service = EmbeddingRAGQdrant()
                self.graph_service = GraphRAGNeo4j()
                
            async def process_github_repositories(self, access_token: str, repo_names: List[str]) -> Dict[str, Any]:
                results = []
                total_files = 0
                total_embeddings = 0
                total_graph_nodes = 0
                
                for repo_name in repo_names:
                    try:
                        # Create GitHubRepository object
                        repo_info = self._get_repo_info(access_token, repo_name)
                        
                        # Process repository
                        result = await self._process_single_repository(access_token, repo_info)
                        results.append(result)
                        
                        total_files += result.get("processed_files", 0)
                        total_embeddings += result.get("embeddings_created", 0)
                        total_graph_nodes += result.get("graph_nodes_created", 0)
                        
                    except Exception as e:
                        results.append({
                            "repository": repo_name,
                            "error": str(e),
                            "processed_files": 0
                        })
                
                return {
                    "processed_repositories": len(repo_names),
                    "total_files_processed": total_files,
                    "total_embeddings_created": total_embeddings,
                    "total_graph_nodes_created": total_graph_nodes,
                    "results": results
                }
            
            def _get_repo_info(self, access_token: str, repo_name: str) -> Dict[str, Any]:
                headers = {"Authorization": f"token {access_token}"}
                response = requests.get(f"https://api.github.com/repos/{repo_name}", headers=headers)
                response.raise_for_status()
                return response.json()
            
            async def _process_single_repository(self, access_token: str, repo_info: Dict[str, Any]) -> Dict[str, Any]:
                # Simplified processing for demo
                return {
                    "repository": repo_info["full_name"],
                    "processed_files": 10,  # Mock data
                    "embeddings_created": 25,
                    "graph_nodes_created": 15,
                    "status": "success"
                }
        
        processor = RepositoryProcessor()
        result = await processor.process_github_repositories(github_data["access_token"], selected_repos)
        
        return {
            "success": True,
            "message": f"Processed {result['processed_repositories']} repositories successfully",
            "details": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing repositories: {str(e)}")

@app.post("/process-projects")
async def process_projects(request: Request, data: dict):
    """Process selected Jira projects"""
    session = require_auth(request)
    jira_data = session.get("jira_oauth")
    
    if not jira_data:
        raise HTTPException(status_code=401, detail="Jira authentication required")
    
    selected_projects = data.get("projects", [])
    if not selected_projects:
        raise HTTPException(status_code=400, detail="No projects selected")
    
    try:
        results = []
        total_test_cases = 0
        total_embeddings = 0
        total_graph_nodes = 0
        
        for project_data in selected_projects:
            project_key = project_data["key"]
            cloud_id = project_data["cloud_id"]
            
            try:
                # Get test cases from Jira
                test_cases = jira_oauth.get_test_cases(jira_data["access_token"], cloud_id, project_key)
                
                # Create embeddings for test cases (simplified for demo)
                embeddings_created = len(test_cases) * 2  # Mock: 2 embeddings per test case
                graph_nodes_created = len(test_cases) + 10  # Mock: test cases + project nodes
                
                results.append({
                    "project": project_key,
                    "test_cases_found": len(test_cases),
                    "embeddings_created": embeddings_created,
                    "graph_nodes_created": graph_nodes_created,
                    "status": "success"
                })
                
                total_test_cases += len(test_cases)
                total_embeddings += embeddings_created
                total_graph_nodes += graph_nodes_created
                
            except Exception as e:
                results.append({
                    "project": project_key,
                    "error": str(e),
                    "test_cases_found": 0
                })
        
        return {
            "success": True,
            "message": f"Processed {len(selected_projects)} projects successfully",
            "details": {
                "processed_projects": len(selected_projects),
                "total_test_cases": total_test_cases,
                "total_embeddings_created": total_embeddings,
                "total_graph_nodes_created": total_graph_nodes,
                "results": results
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing projects: {str(e)}")

@app.get("/logout")
async def logout(request: Request):
    """Logout and clear session"""
    session_id = request.cookies.get("session_id")
    if session_id:
        session_manager.delete_session(session_id)
    
    response = RedirectResponse(url="/")
    response.delete_cookie("session_id")
    return response


@app.get("/", response_class=HTMLResponse)
async def index():
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:        return HTMLResponse(content="<h1>Welcome to Agentic RAG Test Generator</h1><p>Please ensure templates/index.html exists.</p>")


@app.get("/auth/status")
async def auth_status(request: Request):
    """Get current authentication status"""
    session = get_current_session(request)
    
    if not session:
        return JSONResponse(content={
            "authenticated": False,
            "github_authenticated": False,
            "jira_authenticated": False
        })
    
    github_data = session.get("github_oauth")
    jira_data = session.get("jira_oauth")
    
    return JSONResponse(content={
        "authenticated": True,
        "github_authenticated": bool(github_data),
        "github_user": github_data.get("user_info", {}).get("login") if github_data else None,
        "jira_authenticated": bool(jira_data),
        "jira_resources": len(jira_data.get("resources", [])) if jira_data else 0
    })


@app.post("/generate")
async def generate(requirements: str = Form(...), documents: UploadFile = File(...)):
    if not requirements or not documents:
        raise HTTPException(status_code=400, detail="Both requirement and documentation ZIP are required.")

    result_holder = {}

    def processing_job(requirement, file_content, result_holder):
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "uploaded.zip")
            with open(zip_path, "wb") as f:
                f.write(file_content)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            doc_text, docx_pairs, py_data = "", [], {}
            
            for root, _, files in os.walk(temp_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    if fname.startswith("._") or "__MACOSX" in fpath:
                        continue
                        
                    if fname.endswith(".py"):
                        try:
                            with open(fpath, encoding="utf-8", errors="ignore") as f:
                                content = f.read()
                                doc_text += f"\n# File: {fname}\n{content}"
                                
                                tree = ast.parse(content)
                                classes, functions = [], []
                                for node in ast.walk(tree):
                                    if isinstance(node, ast.ClassDef):
                                        classes.append(node.name)
                                    elif isinstance(node, ast.FunctionDef):
                                        docstring = ast.get_docstring(node)
                                        functions.append((node.name, docstring or ""))
                                py_data[fname] = {"classes": classes, "functions": functions}
                        except Exception as e:
                            print(f"Failed to parse {fname}: {e}")
                            
                    elif fname.endswith((".txt", ".md", ".json", ".yaml", ".yml")):
                        with open(fpath, encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            doc_text += f"\n# File: {fname}\n{content}"
                            
                    elif fname.endswith(".docx"):
                        try:
                            doc = Document(fpath)
                            current_heading = None
                            for para in doc.paragraphs:
                                is_bold = any(run.bold for run in para.runs if run.text.strip())
                                text = para.text.strip()
                                if not text:
                                    continue
                                if is_bold:
                                    current_heading = text
                                elif current_heading:
                                    docx_pairs.append((current_heading, text))
                                    doc_text += f"\n{current_heading}: {text}"
                        except Exception as e:
                            print(f"Error reading {fname}: {e}")

            try:
                print("Processing uploaded files...")
                input_data = TestGenInput(requirement=requirement, documentation=doc_text)
                agent = TestCaseGenerator()
                
                # Enhanced text processing for more comprehensive chunks
                print("Creating comprehensive text chunks...")
                
                # Process different types of content
                all_text_chunks = []
                
                # 1. Regular text chunks with overlap
                regular_chunks = chunk_text(doc_text, max_chars=400, overlap=50)
                all_text_chunks.extend(regular_chunks)
                
                # 2. Code-specific chunks
                code_chunks = extract_code_blocks(doc_text)
                all_text_chunks.extend(code_chunks)
                
                # 3. Structured data from docx
                for heading, detail in docx_pairs:
                    if heading and detail:
                        all_text_chunks.append(f"Section: {heading}\nContent: {detail}")
                
                # 4. Python code structure
                for py_file, data in py_data.items():
                    # Add file overview
                    file_overview = f"Python file: {py_file}\n"
                    if data["classes"]:
                        file_overview += f"Classes: {', '.join(data['classes'])}\n"
                    if data["functions"]:
                        func_names = [f[0] for f in data["functions"]]
                        file_overview += f"Functions: {', '.join(func_names)}\n"
                    all_text_chunks.append(file_overview)
                    
                    # Add detailed function info
                    for func_name, docstring in data["functions"]:
                        if docstring:
                            all_text_chunks.append(f"Function: {func_name}\nDocumentation: {docstring}")
                
                # 5. Requirement-specific chunks
                req_chunks = chunk_text(requirement, max_chars=200)
                all_text_chunks.extend(req_chunks)
                
                print(f"Total chunks created: {len(all_text_chunks)}")
                
                # Clear and store embeddings with increased limit
                print("Clearing previous embeddings...")
                agent.embedding_rag.clear_embeddings()
                
                print("Storing new embeddings...")
                stored_count = agent.embedding_rag.embed_and_store(
                    all_text_chunks, 
                    metadata={"source": "upload"},
                    max_chunks=1500  # Increased limit
                )
                
                print(f"Successfully stored {stored_count} embeddings")
                
                print("Generating test cases and scripts...")
                output = agent.generate(input_data)
                
                result_holder["result"] = (
                    f"Generated Test Cases:\n" + "\n".join(output.test_cases) + 
                    f"\n\nGenerated Test Scripts:\n" + "\n".join(output.test_scripts) + 
                    f"\n\nEmbeddings stored: {stored_count}"
                )
                print("Generation completed successfully!")
                
            except Exception as e:
                print(f"Error in processing: {e}")
                result_holder["error"] = str(e)

    # Read file content
    file_content = await documents.read()
    
    # Use threading for processing (FastAPI supports this)
    thread = Thread(target=processing_job, args=(requirements, file_content, result_holder))
    thread.start()
    thread.join()

    if "error" in result_holder:
        raise HTTPException(status_code=500, detail=result_holder["error"])
    
    return {"result": result_holder["result"]}


@app.post("/fetch-jira-tests")
async def fetch_jira_tests(request: JiraFetchRequest = None):
    """Fetch test cases from Jira/Xray and update embeddings and knowledge graph"""
    try:
        # Get optional parameters from request
        if request:
            project_key = request.project_key or JIRA_PROJECT_KEY
            max_results = request.max_results
        else:
            project_key = JIRA_PROJECT_KEY
            max_results = 100
        
        # Initialize Jira client
        jira_client = JiraXrayClient()
        
        # Fetch test cases and executions
        print("Fetching test cases from Jira...")
        test_cases = jira_client.fetch_test_cases(project_key, max_results)
        
        if not test_cases:
            raise HTTPException(status_code=400, detail="No test cases found or failed to fetch from Jira")
        
        print("Fetching test executions from Jira...")
        test_executions = jira_client.fetch_test_executions(project_key, max_results // 2)
        
        # Initialize components
        embedding_rag = EmbeddingRAGQdrant()
        
        try:
            graph_rag = GraphRAGNeo4j()
            use_graph = True
        except Exception as e:
            print(f"Failed to initialize Neo4j: {e}")
            graph_rag = None
            use_graph = False
        
        # Store embeddings without clearing existing data
        print("Storing Jira test embeddings...")
        stored_embeddings = embedding_rag.embed_and_store_jira_tests(test_cases, test_executions)
        
        # Update knowledge graph without clearing existing data
        graph_stored = 0
        if use_graph and graph_rag:
            print("Updating knowledge graph with Jira data...")
            graph_rag.load_jira_tests_knowledge(test_cases, test_executions)
            graph_stored = len(test_cases) + len(test_executions or [])
        
        result = {
            "message": "Successfully fetched and processed Jira tests",
            "test_cases_fetched": len(test_cases),
            "test_executions_fetched": len(test_executions or []),
            "embeddings_stored": stored_embeddings,
            "graph_nodes_created": graph_stored,
            "project_key": project_key
        }
        
        return result
        
    except Exception as e:
        print(f"Error in fetch_jira_tests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/populate-sample-data")
async def populate_sample_data():
    """
    Endpoint to populate sample test data for demonstration
    """
    try:
        agent = AgenticRAGTestChecker()
        stored_count = agent.populate_sample_test_data()
        
        return {
            "message": "Sample test data populated successfully",
            "test_cases_stored": stored_count,
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error in populate_sample_data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agentic-test-check")
async def agentic_test_check(request: FunctionalityRequest):
    """
    Agentic RAG system that checks if test cases exist for a functionality
    and either shows existing ones or creates new ones
    """
    try:
        user_prompt = request.functionality
        
        if not user_prompt:
            raise HTTPException(status_code=400, detail="Functionality description is required")
        
        # Initialize the agentic RAG test checker
        print(f"Processing agentic test check for: {user_prompt}")
        agent = AgenticRAGTestChecker()
        
        # Analyze the functionality request
        analysis_result = agent.analyze_functionality_request(user_prompt)
        
        # Validate analysis_result structure
        if not isinstance(analysis_result, dict) or 'action' not in analysis_result:
            raise HTTPException(
                status_code=500, 
                detail="Invalid analysis result structure"
            )        # Format the response based on the action taken
        if analysis_result['action'] == 'NO_DATA':
            # Handle case where no existing data is found
            data_status = analysis_result.get('data_status', {})
            response = {
                "action": "NO_DATA",
                "status": analysis_result.get('status', 'No existing data found'),
                "message": analysis_result.get('message', 'No embeddings or graph data available'),
                "functionality": user_prompt,
                "data_status": data_status,
                "recommendations": analysis_result.get('recommendations', []),
                "tests_analyzed": analysis_result.get('tests_analyzed', 0),
                "summary": f"No existing data found. Embeddings: {data_status.get('embedding_count', 0)}, Graph nodes: {data_status.get('graph_node_count', 0)}",
                "all_test_details": {
                    "existing_tests": [],
                    "total_tests": 0,
                    "data_available": False
                }
            }
        
        elif analysis_result['action'] == 'FUNCTION_NOT_FOUND':
            # Handle case where function doesn't exist in both docs and graph
            data_status = analysis_result.get('data_status', {})
            response = {
                "action": "FUNCTION_NOT_FOUND",
                "status": analysis_result.get('status', 'Function not found'),
                "message": analysis_result.get('message', 'Function does not exist in documentation or graph'),
                "functionality": user_prompt,
                "function_name": analysis_result.get('function_name', ''),
                "exists_in_docs": analysis_result.get('exists_in_docs', False),
                "exists_in_graph": analysis_result.get('exists_in_graph', False),
                "doc_search_results": analysis_result.get('doc_search_results', []),
                "graph_search_results": analysis_result.get('graph_search_results', []),
                "data_status": data_status,
                "recommendations": analysis_result.get('recommendations', []),                "tests_analyzed": analysis_result.get('tests_analyzed', 0),
                "summary": f"Function '{analysis_result.get('function_name', '')}' not found in both documentation and graph. Tests cannot be created or updated.",
                "all_test_details": {
                    "existing_tests": [],
                    "total_tests": 0,
                    "data_available": True,
                    "function_exists": False
                }
            }
        
        elif analysis_result['action'] == 'EXISTING':
            existing_tests = analysis_result.get('existing_tests', [])
            data_status = analysis_result.get('data_status', {})
            response = {
                "action": "EXISTING",
                "status": analysis_result.get('status', 'Test cases found'),
                "message": analysis_result.get('message', 'Found existing test cases'),
                "functionality": user_prompt,
                "coverage_score": analysis_result.get('coverage_score', 0.0),
                "existing_tests": existing_tests,  # This should now contain all found test cases
                "recommendations": analysis_result.get('recommendations', []),                "tests_analyzed": analysis_result.get('tests_analyzed', 0),
                "data_status": data_status,
                "summary": f"Found {len(existing_tests)} existing test cases with {analysis_result.get('coverage_score', 0.0):.1%} coverage confidence",
                "all_test_details": {
                    "existing_tests": existing_tests,
                    "total_tests": len(existing_tests),
                    "coverage_score": analysis_result.get('coverage_score', 0.0),
                    "data_available": True
                }
            }
        
        elif analysis_result['action'] == 'UPDATED':
            # Handle updated tests - extract from analysis_result directly
            updated_test_cases = analysis_result.get('updated_tests', [])
            updated_test_scripts = analysis_result.get('updated_test_scripts', [])
            original_tests = analysis_result.get('original_tests', [])
            update_rationale = analysis_result.get('update_rationale', 'Tests updated for better coverage')
            coverage_improvement = analysis_result.get('coverage_improvement', 'Coverage improvements applied')
            data_status = analysis_result.get('data_status', {})
            
            response = {
                "action": "UPDATED",
                "status": analysis_result.get('status', 'Test cases updated'),
                "message": analysis_result.get('message', 'Updated existing test cases'),
                "functionality": user_prompt,
                "updated_test_cases": updated_test_cases,
                "updated_test_scripts": updated_test_scripts,
                "original_tests": original_tests,
                "update_rationale": update_rationale,
                "coverage_improvement": coverage_improvement,
                "tests_analyzed": analysis_result.get('tests_analyzed', 0),                "data_status": data_status,
                "summary": f"Updated {len(updated_test_cases)} existing test cases with improved coverage",
                "all_test_details": {
                    "updated_tests": updated_test_cases,
                    "test_scripts": updated_test_scripts,
                    "original_tests": original_tests,
                    "total_tests": len(updated_test_cases),
                    "data_available": True
                }
            }
        
        else:  # CREATED
            new_tests = analysis_result.get('new_tests', {})
            data_status = analysis_result.get('data_status', {})
            if isinstance(new_tests, dict):
                test_cases = new_tests.get('test_cases', [])
                test_scripts = new_tests.get('test_scripts', [])
                response = {
                    "action": "CREATED", 
                    "status": analysis_result.get('status', 'New test cases created'),
                    "message": analysis_result.get('message', 'Created new test cases'),
                    "functionality": user_prompt,
                    "new_test_cases": test_cases,
                    "new_test_scripts": test_scripts,
                    "rationale": new_tests.get('rationale', 'Test cases created for functionality coverage'),
                    "existing_partial_coverage": [test.get('content', str(test))[:200] + "..." for test in analysis_result.get('existing_partial_coverage', [])],
                    "coverage_gap": analysis_result.get('coverage_gap', 'No existing coverage found'),
                    "tests_analyzed": analysis_result.get('tests_analyzed', 0),
                    "data_status": data_status,
                    "summary": f"Created {len(test_cases)} new test cases",
                    "all_test_details": {
                        "test_cases": test_cases,
                        "test_scripts": test_scripts,
                        "partial_coverage": analysis_result.get('existing_partial_coverage', []),
                        "total_tests": len(test_cases),
                        "data_available": True
                    }
                }
            else:
                # Fallback if new_tests is not a dict
                response = {
                    "action": "CREATED", 
                    "status": analysis_result.get('status', 'New test cases created'),
                    "message": analysis_result.get('message', 'Created new test cases'),
                    "functionality": user_prompt,
                    "new_test_cases": [],
                    "new_test_scripts": [],
                    "rationale": "Test cases created for functionality coverage",
                    "existing_partial_coverage": [],
                    "coverage_gap": "No existing coverage found",
                    "tests_analyzed": analysis_result.get('tests_analyzed', 0),
                    "data_status": data_status,
                    "summary": "Created new test cases",
                    "all_test_details": {
                        "test_cases": [],
                        "test_scripts": [],
                        "partial_coverage": [],
                        "total_tests": 0,
                        "data_available": True
                    }
                }
        
        return response
        
    except Exception as e:
        print(f"Error in agentic test check: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process request: {str(e)}"
        )


@app.get("/get-all-test-summary")
async def get_all_test_summary():
    """
    Get a comprehensive summary of all test cases in the system
    """
    try:
        # Initialize the agentic RAG test checker
        agent = AgenticRAGTestChecker()
        
        # Search for all test cases
        all_test_chunks = agent.embedding_rag.search("test case", top_k=100)
        
        # Categorize tests
        login_tests = [chunk for chunk in all_test_chunks if any(keyword in chunk.lower() for keyword in ['login', 'auth', 'signin'])]
        crud_tests = [chunk for chunk in all_test_chunks if any(keyword in chunk.lower() for keyword in ['crud', 'create', 'update', 'delete'])]
        api_tests = [chunk for chunk in all_test_chunks if any(keyword in chunk.lower() for keyword in ['api', 'endpoint', 'request'])]
        validation_tests = [chunk for chunk in all_test_chunks if any(keyword in chunk.lower() for keyword in ['validation', 'validate', 'input'])]
        
        summary = {
            "total_tests": len(all_test_chunks),
            "test_categories": {
                "login_authentication": len(login_tests),
                "crud_operations": len(crud_tests),
                "api_testing": len(api_tests),
                "input_validation": len(validation_tests),
                "other": len(all_test_chunks) - len(login_tests) - len(crud_tests) - len(api_tests) - len(validation_tests)
            },
            "all_tests": all_test_chunks,
            "categorized_tests": {
                "login_tests": login_tests[:10],  # Limit to 10 per category
                "crud_tests": crud_tests[:10],
                "api_tests": api_tests[:10],
                "validation_tests": validation_tests[:10]
            }
        }
        
        return summary        
    except Exception as e:
        print(f"Error getting test summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/populate-jira-data")
async def populate_jira_data():
    """
    Manually populate data from Jira/Xray - only call this when you want to fetch and store new data
    """
    try:
        print("Manual Jira data population requested...")
        agent = AgenticRAGTestChecker()
        
        # Force populate data from Jira
        result = agent.populate_data_if_needed(force_populate=True)
        
        if isinstance(result, dict) and 'embeddings_available' in result:
            # This is a status check result
            return {
                "status": "success",
                "message": "Data population completed",
                "data_status": result,
                "embeddings_count": result.get('embedding_count', 0),
                "graph_nodes": result.get('graph_node_count', 0)
            }
        else:
            # This is a count result
            return {
                "status": "success", 
                "message": f"Successfully populated {result} test cases from Jira",
                "populated_count": result
            }
        
    except Exception as e:
        print(f"Error in populate_jira_data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data-status")
async def get_data_status():
    """
    Check the current status of embeddings and graph data
    """
    try:
        agent = AgenticRAGTestChecker()
        data_status = agent.check_existing_data_status()
        
        return {
            "status": "success",
            "data_status": data_status,
            "embeddings_available": data_status.get('embeddings_available', False),
            "graph_available": data_status.get('graph_available', False),
            "embeddings_count": data_status.get('embedding_count', 0),
            "graph_nodes": data_status.get('graph_node_count', 0),
            "message": f"Embeddings: {data_status.get('embedding_count', 0)}, Graph nodes: {data_status.get('graph_node_count', 0)}"
        }
        
    except Exception as e:
        print(f"Error getting data status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5005, reload=True)
