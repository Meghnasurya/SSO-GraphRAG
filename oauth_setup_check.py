#!/usr/bin/env python3
"""
OAuth Setup Verification Script
This script helps verify and troubleshoot OAuth configuration
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_github_oauth():
    """Check GitHub OAuth configuration"""
    print("\n🔍 Checking GitHub OAuth Configuration...")
    
    github_client_id = os.getenv("GITHUB_CLIENT_ID")
    github_client_secret = os.getenv("GITHUB_CLIENT_SECRET")
    
    if not github_client_id or not github_client_secret:
        print("❌ GitHub OAuth credentials not found in .env file")
        return False
    
    print(f"✅ GitHub Client ID: {github_client_id}")
    print(f"✅ GitHub Client Secret: {'*' * len(github_client_secret)}")
    
    # Check if the OAuth app is properly configured
    print("\n📋 Required GitHub OAuth App Settings:")
    print("   - Application name: Your choice")
    print("   - Homepage URL: http://localhost:3000")
    print("   - Authorization callback URL: http://localhost:8000/auth/github/callback")
    print("   - Enable 'Request user authorization (OAuth) during installation': ✓")
    
    print("\n🔗 To configure your GitHub OAuth app:")
    print("   1. Go to https://github.com/settings/developers")
    print("   2. Click 'OAuth Apps' → Select your app or create new")
    print("   3. Update the callback URL to: http://localhost:8000/auth/github/callback")
    print("   4. Save changes")
    
    return True

def check_jira_oauth():
    """Check Jira OAuth configuration"""
    print("\n🔍 Checking Jira OAuth Configuration...")
    
    jira_client_id = os.getenv("JIRA_CLIENT_ID")
    jira_client_secret = os.getenv("JIRA_CLIENT_SECRET")
    jira_url = os.getenv("JIRA_URL")
    
    if not jira_client_id or not jira_client_secret:
        print("❌ Jira OAuth credentials not found in .env file")
        return False
    
    print(f"✅ Jira Client ID: {jira_client_id}")
    print(f"✅ Jira Client Secret: {'*' * len(jira_client_secret)}")
    print(f"✅ Jira URL: {jira_url}")
    
    print("\n📋 Required Jira OAuth App Settings:")
    print("   - App name: Your choice")
    print("   - Callback URL: http://localhost:8000/auth/jira/callback")
    print("   - Scopes: read:jira-user, read:jira-work")
    
    print("\n🔗 To configure your Jira OAuth app:")
    print("   1. Go to https://developer.atlassian.com/console/myapps/")
    print("   2. Select your app or create new")
    print("   3. Add OAuth 2.0 (3LO) authorization")
    print("   4. Set callback URL to: http://localhost:8000/auth/jira/callback")
    print("   5. Add required scopes: read:jira-user, read:jira-work")
    
    return True

def check_server_status():
    """Check if servers are running"""
    print("\n🔍 Checking Server Status...")
    
    try:
        # Check backend server
        response = requests.get("http://localhost:8000/", timeout=5)
        print("✅ Backend server is running on http://localhost:8000")
    except requests.exceptions.RequestException:
        print("❌ Backend server is not running on http://localhost:8000")
        print("   Run: python main.py")
    
    try:
        # Check frontend server
        response = requests.get("http://localhost:3000/", timeout=5)
        print("✅ Frontend server is running on http://localhost:3000")
    except requests.exceptions.RequestException:
        print("❌ Frontend server is not running on http://localhost:3000")
        print("   Run: cd frontend && python serve.py")

def main():
    print("🚀 OAuth Setup Verification")
    print("=" * 50)
    
    check_github_oauth()
    check_jira_oauth()
    check_server_status()
    
    print("\n🎯 Common Issues and Solutions:")
    print("1. 'redirect_uri is not associated with this application'")
    print("   → Update GitHub OAuth app callback URL to: http://localhost:8000/auth/github/callback")
    print("\n2. 'Jira instance URL required'")
    print("   → Use the Jira configuration form in the frontend")
    print("\n3. 'Invalid state parameter'")
    print("   → Ensure backend runs on port 8000 (not 5005)")
    print("   → Clear browser cookies and try again")
    print("\n4. CORS errors")
    print("   → Ensure frontend runs on port 3000")
    print("   → Check CORS middleware is enabled in backend")

if __name__ == "__main__":
    main()
