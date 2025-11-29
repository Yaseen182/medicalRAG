"""
Test suite for Medical RAG Chatbot API
"""
import pytest
import json
from app import app


@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'status' in data
    assert data['status'] == 'healthy'


def test_chat_endpoint_missing_question(client):
    """Test chat endpoint with missing question"""
    response = client.post(
        '/api/v1/chat',
        data=json.dumps({}),
        content_type='application/json'
    )
    assert response.status_code == 400


def test_chat_endpoint_empty_question(client):
    """Test chat endpoint with empty question"""
    response = client.post(
        '/api/v1/chat',
        data=json.dumps({'question': ''}),
        content_type='application/json'
    )
    assert response.status_code == 400


def test_search_endpoint_missing_query(client):
    """Test search endpoint with missing query"""
    response = client.post(
        '/api/v1/search',
        data=json.dumps({}),
        content_type='application/json'
    )
    assert response.status_code == 400


def test_search_endpoint_invalid_top_k(client):
    """Test search endpoint with invalid top_k"""
    response = client.post(
        '/api/v1/search',
        data=json.dumps({'query': 'test', 'top_k': 100}),
        content_type='application/json'
    )
    assert response.status_code == 400


def test_stats_endpoint(client):
    """Test stats endpoint"""
    response = client.get('/api/v1/stats')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'pipeline_status' in data
    assert 'model_name' in data


def test_not_found(client):
    """Test 404 error handling"""
    response = client.get('/nonexistent')
    assert response.status_code == 404
