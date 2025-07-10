#!/usr/bin/env python3
"""
Test script to verify OAuth setup and basic functionality
"""

import os
import sys
import requests
import subprocess
import time
from urllib.parse import urlencode

def test_environment_setup():
    """Test if required environment variables are set"""
    print("🔍 Checking environment setup...")
    
    required_vars = [
        'GEMINI_API_KEY',
        'NEO4J_URI', 
        'NEO4J_USER',
        'NEO4J_PASSWORD'
    ]
    
    optional_oauth_vars = [
        'GITHUB_CLIENT_ID',
        'GITHUB_CLIENT_SECRET', 
        'JIRA_OAUTH_CLIENT_ID',
        'JIRA_OAUTH_CLIENT_SECRET'
    ]
    
    missing_required = []
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    missing_oauth = []
    for var in optional_oauth_vars:
        if not os.getenv(var):
            missing_oauth.append(var)
    
    if missing_required:
        print(f"❌ Missing required environment variables: {', '.join(missing_required)}")
        return False
    else:
        print("✅ Required environment variables are set")
    
    if missing_oauth:
        print(f"⚠️ Missing OAuth variables (OAuth features will be disabled): {', '.join(missing_oauth)}")
    else:
        print("✅ OAuth environment variables are set")
    
    return True

def test_application_startup():
    """Test if the application can start up"""
    print("🚀 Testing application startup...")
    
    try:
        # Import the main module to check for syntax errors
        import main
        print("✅ Application imports successfully")
        return True
    except Exception as e:
        print(f"❌ Application import failed: {e}")
        return False

def test_basic_endpoints():
    """Test basic application endpoints"""
    print("🌐 Testing basic endpoints...")
    
    # Start the server in background
    server_process = None
    try:
        import uvicorn
        import threading
        import main
        
        def run_server():
            uvicorn.run(main.app, host="127.0.0.1", port=8001, log_level="error")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        time.sleep(3)
        
        # Test basic endpoints
        test_urls = [
            ("GET", "http://127.0.0.1:8001/", "Home page"),
            ("GET", "http://127.0.0.1:8001/auth/status", "Auth status"),
        ]
        
        for method, url, description in test_urls:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code < 500:
                    print(f"✅ {description}: Status {response.status_code}")
                else:
                    print(f"⚠️ {description}: Status {response.status_code}")
            except Exception as e:
                print(f"❌ {description}: Failed - {e}")
        
        return True
        
    except ImportError:
        print("❌ uvicorn not installed, skipping server test")
        return False
    except Exception as e:
        print(f"❌ Server test failed: {e}")
        return False

def test_oauth_configuration():
    """Test OAuth configuration"""
    print("🔐 Testing OAuth configuration...")
    
    github_client_id = os.getenv('GITHUB_CLIENT_ID')
    jira_client_id = os.getenv('JIRA_OAUTH_CLIENT_ID')
    
    if github_client_id:
        print(f"✅ GitHub OAuth Client ID configured: {github_client_id[:8]}...")
    else:
        print("⚠️ GitHub OAuth not configured")
    
    if jira_client_id:
        print(f"✅ Jira OAuth Client ID configured: {jira_client_id[:8]}...")
    else:
        print("⚠️ Jira OAuth not configured")
    
    return bool(github_client_id or jira_client_id)

def print_setup_instructions():
    """Print setup instructions for OAuth"""
    print("\n📋 OAuth Setup Instructions:")
    print("=" * 50)
    
    print("\n🐙 GitHub OAuth Setup:")
    print("1. Go to https://github.com/settings/developers")
    print("2. Click 'New OAuth App'")
    print("3. Set Authorization callback URL: http://localhost:8000/auth/github/callback")
    print("4. Add GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET to .env")
    
    print("\n🔷 Jira OAuth Setup:")
    print("1. Go to https://developer.atlassian.com/console")
    print("2. Create OAuth 2.0 integration")
    print("3. Set callback URL: http://localhost:8000/auth/jira/callback")
    print("4. Add JIRA_OAUTH_CLIENT_ID and JIRA_OAUTH_CLIENT_SECRET to .env")
    
    print("\n🔑 Environment Setup:")
    print("Copy .env.example to .env and fill in your credentials")
    
    print("\n🚀 Run Application:")
    print("python main.py")
    print("Then visit: http://localhost:8000")

def main():
    """Main test function"""
    print("🧪 Agentic RAG Test Generator - OAuth Setup Test")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 4
    
    # Run tests
    if test_environment_setup():
        tests_passed += 1
    
    if test_application_startup():
        tests_passed += 1
    
    if test_oauth_configuration():
        tests_passed += 1
    
    if test_basic_endpoints():
        tests_passed += 1
    
    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your OAuth setup is ready.")
        print("\n🚀 Next steps:")
        print("1. Run: python main.py")
        print("2. Open: http://localhost:8000")
        print("3. Go to 'SSO Setup' tab to connect your accounts")
    else:
        print("⚠️ Some tests failed. Please check the setup.")
        if tests_passed < 2:
            print_setup_instructions()

if __name__ == "__main__":
    main()
