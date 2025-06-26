# RAG Service Testing Tools

This directory contains scripts to test the RAG (Retrieval-Augmented Generation) service running on port 5004.

## Available Test Scripts

### 1. Basic Test Script (`test_rag_service.py`)

A simple script to test basic functionality of the RAG service.

**Usage:**
```bash
# Run with default queries
python3 test_rag_service.py

# Run with custom queries
python3 test_rag_service.py "What is deep learning?" "Explain transformers"
```

### 2. Shell Script Wrapper (`test_rag.sh`)

A convenient shell script wrapper for the basic test.

**Usage:**
```bash
# Make executable (if not already)
chmod +x test_rag.sh

# Run with default queries
./test_rag.sh

# Run with custom queries
./test_rag.sh "What is deep learning?" "Explain transformers"
```

### 3. Comprehensive Test Script (`comprehensive_test.py`)

A more advanced test script with multiple testing options.

**Usage:**
```bash
# Make executable (if not already)
chmod +x comprehensive_test.py

# Run all tests
./comprehensive_test.py

# Run specific test type
./comprehensive_test.py --test basic  # Basic connectivity test
./comprehensive_test.py --test params  # Parameter variation test
./comprehensive_test.py --test perf  # Performance test
./comprehensive_test.py --test error  # Error handling test

# Run with custom queries
./comprehensive_test.py --queries "What is deep learning?" "Explain transformers"

# Customize top-k results
./comprehensive_test.py --queries "What is machine learning?" --topk 5

# Don't return scores
./comprehensive_test.py --queries "What is machine learning?" --no-scores

# Test against a different URL
./comprehensive_test.py --url http://different-host:5004/retrieve
```

## Troubleshooting

If you encounter connection errors:

1. Ensure the RAG service is running (check with `ps aux | grep retrieval_server.py`)
2. Verify the service is running on port 5004 (check with `netstat -tuln | grep 5004`)
3. Check if there are any firewall issues blocking the connection
4. Examine the RAG service logs for any errors

## Example Output

When the test is successful, you should see output similar to:

```
Sending request to RAG service at http://localhost:5004/retrieve
Queries: ['What is machine learning?']

===== RAG Service Response =====

Query 1: 'What is machine learning?'
  Result 1 (Score: 0.8765):
    Title: Machine Learning
    Text: Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn...