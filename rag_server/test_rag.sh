#!/bin/bash

# Test script for RAG service
# Usage: ./test_rag.sh [query1] [query2] ...

# If no arguments provided, run with default queries
if [ $# -eq 0 ]; then
  echo "Running with default queries..."
  CUDA_VISIBLE_DEVICES=7 python3 $(dirname "$0")/test_rag_service.py
else
  # Pass all arguments as queries to the test script
  CUDA_VISIBLE_DEVICES=7 python3 $(dirname "$0")/test_rag_service.py "$@"
fi