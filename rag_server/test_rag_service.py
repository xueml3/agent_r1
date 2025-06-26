import requests
import json
import sys

def test_rag_service(queries, topk=3, return_scores=True):
    """
    Test the RAG service by sending queries and displaying the retrieved documents.
    
    Args:
        queries (list): List of query strings
        topk (int): Number of documents to retrieve per query
        return_scores (bool): Whether to return relevance scores
    """
    # RAG service endpoint
    url = "http://localhost:5004/retrieve"
    
    # Prepare the request payload
    payload = {
        "queries": queries,
        "topk": topk,
        "return_scores": return_scores
    }
    
    print(f"Sending request to RAG service at {url}")
    print(f"Queries: {queries}")
    
    try:
        # Send POST request to the RAG service
        response = requests.post(url, json=payload)
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        print("\n===== RAG Service Response =====")
        
        # Process and display results
        for i, query_result in enumerate(result["result"]):
            print(f"\nQuery {i+1}: '{queries[i]}'")
            
            if return_scores:
                # Results with scores
                for j, item in enumerate(query_result):
                    doc = item["document"]
                    score = item["score"]
                    
                    # Extract title and text if available
                    title = doc.get("title", "No title")
                    text = doc.get("text", doc.get("contents", "No content"))
                    
                    # Truncate text for display
                    if len(text) > 200:
                        text = text[:200] + "..."
                    
                    print(f"  Result {j+1} (Score: {score:.4f}):")
                    print(f"    Title: {title}")
                    print(f"    Text: {text}")
            else:
                # Results without scores
                for j, doc in enumerate(query_result):
                    title = doc.get("title", "No title")
                    text = doc.get("text", doc.get("contents", "No content"))
                    
                    if len(text) > 200:
                        text = text[:200] + "..."
                    
                    print(f"  Result {j+1}:")
                    print(f"    Title: {title}")
                    print(f"    Text: {text}")
        
        return True
    
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the RAG service. Make sure it's running.")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    # Default queries if none provided
    default_queries = ["What is machine learning?", "Explain reinforcement learning"]
    
    # Use command line arguments as queries if provided
    if len(sys.argv) > 1:
        queries = sys.argv[1:]
    else:
        queries = default_queries
    
    # Test the RAG service
    test_rag_service(queries)