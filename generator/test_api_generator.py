import asyncio
from easydict import EasyDict
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generator.api_generator import APIGenerator

async def main():
    # Test message to send to the models
    test_message = [{"role": "user", "content": "Hello, please introduce yourself briefly."}]
    
    print("Testing API Generator functions...")
    
    # Test get_response_r1
    print("\n=== Testing get_response_r1 ===")
    config_r1 = EasyDict({
        "api_method": "deepseek-r1"
    })
    generator_r1 = APIGenerator(config_r1)
    try:
        response_r1 = generator_r1.get_response_r1(test_message)
        print(f"Response from DeepSeek-R1:\n{response_r1}")
    except Exception as e:
        print(f"Error testing get_response_r1: {e}")
    
    # Test get_response_qwen
    print("\n=== Testing get_response_qwen ===")
    config_qwen = EasyDict({
        "api_method": "qwen2.5-72b"
    })
    generator_qwen = APIGenerator(config_qwen)
    try:
        response_qwen = generator_qwen.get_response_qwen(test_message)
        print(f"Response from Qwen:\n{response_qwen}")
    except Exception as e:
        print(f"Error testing get_response_qwen: {e}")
    
    # Test async functions
    print("\n=== Testing async functions ===")
    
    # Test get_response_r1_async
    try:
        response_r1_async = await generator_r1.get_response_r1_async(test_message)
        print(f"\nAsync Response from DeepSeek-R1:\n{response_r1_async}")
    except Exception as e:
        print(f"Error testing get_response_r1_async: {e}")
    
    # Test get_response_qwen_async
    try:
        response_qwen_async = await generator_qwen.get_response_qwen_async(test_message)
        print(f"\nAsync Response from Qwen:\n{response_qwen_async}")
    except Exception as e:
        print(f"Error testing get_response_qwen_async: {e}")


if __name__ == "__main__":
    asyncio.run(main())