import sys
import torch
from pathlib import Path
import json

# Add the parent directory to sys.path to import the modules
sys.path.append(str(Path(__file__).parent))

from envs.search import SearchEnv
from verl import DataProto

class MockTokenizer:
    """Mock tokenizer for testing purposes"""
    def decode(self, tokens, **kwargs):
        # For testing, we'll just return a predefined string
        return "This is a mock decoded string"

class DataProtoItem:
    """Mock DataProtoItem for testing purposes"""
    def __init__(self, prompt, response, target):
        self.prompt = prompt
        self.response = response
        self.target = target
        self.data_source = "test"
        self.extra_info = {}

def create_test_data():
    """Create test data for demonstration"""
    test_data = []
    
    # Test case 1: Correct answer with proper format
    test_data.append(DataProtoItem(
        prompt="What is the capital of France?",
        response="I'll search for the capital of France.\n<tool_call>{\"name\": \"search\", \"input\": {\"query\": \"capital of France\"}}</tool_call>\nThe capital of France is Paris.\n<answer>Paris</answer>",
        target="Paris"
    ))
    
    # Test case 2: Incorrect answer with proper format
    test_data.append(DataProtoItem(
        prompt="What is the capital of Germany?",
        response="I'll search for the capital of Germany.\n<tool_call>{\"name\": \"search\", \"input\": {\"query\": \"capital of Germany\"}}</tool_call>\nThe capital of Germany is Berlin.\n<answer>Munich</answer>",
        target="Berlin"
    ))
    
    # Test case 3: Correct answer with improper format (missing tags)
    test_data.append(DataProtoItem(
        prompt="What is the capital of Italy?",
        response="I'll search for the capital of Italy.\nThe capital of Italy is Rome.",
        target="Rome"
    ))
    
    # Test case 4: Multiple answers with the last one correct
    test_data.append(DataProtoItem(
        prompt="What is the capital of Spain?",
        response="I'll search for the capital of Spain.\n<tool_call>{\"name\": \"search\", \"input\": {\"query\": \"capital of Spain\"}}</tool_call>\nLet me think...\n<answer>Barcelona</answer>\nActually, I made a mistake.\n<answer>Madrid</answer>",
        target="Madrid"
    ))
    
    # Test case 5: Answer with thinking tags
    test_data.append(DataProtoItem(
        prompt="What is the capital of Japan?",
        response="<think>I need to find the capital of Japan</think>\n<tool_call>{\"name\": \"search\", \"input\": {\"query\": \"capital of Japan\"}}</tool_call>\nThe capital of Japan is Tokyo.\n<answer>Tokyo</answer>",
        target="Tokyo"
    ))
    
    # Test case 6: Invalid JSON in tool call
    test_data.append(DataProtoItem(
        prompt="What is the capital of Canada?",
        response="<tool_call>{\"name\": \"search\", \"input\": {\"query\": \"capital of Canada\"}</tool_call>\nThe capital of Canada is Ottawa.\n<answer>Ottawa</answer>",
        target="Ottawa"
    ))
    
    return test_data

def test_search_env():
    """Test the SearchEnv class"""
    # Create a config for the SearchEnv
    config = {
        'name': 'base',
        'tool_manager': 'qwen3',
        'mcp_mode': 'sse',
        "tool_name_selected": ["search-query_rag"],
        'config_path': 'envs/configs/mcp_tools.pydata',
        'enable_thinking': True,
        'max_prompt_length': 4096,
        'enable_limiter': True,
        "parallel_sse_tool_call": {
            "is_enabled": True,
            "num_instances": 4
        }
    }
    # config = {
    #     "env_name": "search",
    #     "use_verify_tool": False
    # }
    
    # Create the SearchEnv
    env = SearchEnv(config)
    
    # Mock the _process_data method to return our test data
    def mock_process_data(data_item, tokenizer):
        return {
            'ground_truth': {'target': data_item.target},
            'response_str': data_item.response,
            'prompt_str': data_item.prompt,
            'data_source': data_item.data_source,
            'extra_info': data_item.extra_info
        }
    
    env._process_data = mock_process_data
    
    # Create test data
    test_data = create_test_data()
    
    # Create a mock tokenizer
    tokenizer = MockTokenizer()
    
    # Test the _compute_score_with_rules method
    scores = env._compute_score_with_rules(test_data, tokenizer)
    
    # Print the results
    print("\n===== TEST RESULTS =====")
    for i, data_item in enumerate(test_data):
        print(f"\nTest Case {i+1}:")
        print(f"Prompt: {data_item.prompt}")
        print(f"Target Answer: {data_item.target}")
        print(f"Response:")
        print(data_item.response)
        
        # Extract the answer for demonstration
        answer = None
        import re
        think_pattern = r'<think>.*?</think>'
        solution_str = re.sub(think_pattern, '', data_item.response, flags=re.DOTALL)
        answer_pattern = r'<answer>(.*?)</answer>'
        matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
        if len(matches) > 0:
            answer = matches[-1].group(1).strip()
        
        print(f"Extracted Answer: {answer}")
        print(f"Score: {scores[i][0]}")
        
        # Explain the score
        score = scores[i][0]
        if score == 1.05:  # Correct answer with good format
            print("Explanation: Correct answer with proper format (1.0 + 0.05 format bonus)")
        elif score == 0.05:  # Incorrect answer with good format
            print("Explanation: Incorrect answer with proper format (0.0 + 0.05 format bonus)")
        elif score == -0.05:  # Missing answer tags
            print("Explanation: Missing answer tags (-0.05 format penalty)")
        elif score > 0 and score < 1:  # Partial format score
            print(f"Explanation: Format score only: {score}")
        else:
            print(f"Explanation: Custom score: {score}")

def test_step_reward():
    """Test the get_step_reward method"""
    # Create a config for the SearchEnv
    config = {
        'name': 'base',
        'tool_manager': 'qwen3',
        'mcp_mode': 'sse',
        "tool_name_selected": ["search-query_rag"],
        'config_path': 'envs/configs/mcp_tools.pydata',
        'enable_thinking': True,
        'max_prompt_length': 4096,
        'enable_limiter': True,
        "parallel_sse_tool_call": {
            "is_enabled": True,
            "num_instances": 4
        }
    }
    
    # Create the SearchEnv
    env = SearchEnv(config)
    
    # Create a mock tool_manager
    class MockToolManager:
        def parse_response(self, response_content):
            if "answer" in response_content:
                return "answer", []
            elif "<empty>" in response_content:
                return "tool", [{"name": "<empty>"}]
            elif "<error>" in response_content:
                return "tool", [{"name": "search"}, {"name": "<error>"}, {"name": "search"}]
            else:
                return "tool", [{"name": "search"}, {"name": "search"}]
    
    env.tool_manager = MockToolManager()
    
    # Test responses
    responses = [
        "This is an answer response",
        "This is a response with <empty> tool",
        "This is a response with <error> tool",
        "This is a normal tool response"
    ]
    
    # Get step rewards
    rewards = env.get_step_reward(responses)
    
    # Print the results
    print("\n===== STEP REWARD TEST =====")
    for i, response in enumerate(responses):
        print(f"\nResponse {i+1}: {response}")
        print(f"Reward: {rewards[i]}")
        # Explain the reward
        if torch.isnan(torch.Tensor([rewards[i]])):
            print("Explanation: Answer response (NaN reward)")
        elif rewards[i] == -0.05:
            print("Explanation: Empty tool (-0.05 penalty)")
        elif rewards[i] == 0.033333:
            print("Explanation: Tool with error (partial reward)")
        elif rewards[i] == 0.1:
            print("Explanation: Normal tool (full format reward)")

if __name__ == "__main__":
    print("Testing SearchEnv score calculation...")
    test_search_env()
    
    print("\nTesting SearchEnv step reward calculation...")
    test_step_reward()