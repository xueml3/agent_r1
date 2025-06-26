# SearchEnv Test

This test script demonstrates how the `search.py` module parses predicted answers and calculates scores in the RL-Factory project.

## What the Test Demonstrates

1. **Answer Parsing**: Shows how the code extracts answers from responses using regex patterns
   - Extracts content within `<answer>...</answer>` tags
   - Removes content within `<think>...</think>` tags
   - Takes the last answer when multiple answers are provided

2. **Score Calculation**: Shows how scores are calculated based on:
   - Exact match between normalized predicted answer and ground truth
   - Format correctness (proper use of tags)
   - Tool call format validation (JSON parsing)

3. **Step Reward Calculation**: Shows how intermediate rewards are calculated during the search process
   - Rewards for proper tool usage
   - Penalties for empty or error tools

## How to Run the Test

Run the test script with Python:

```bash
cd Projects/RL-Factory
python test_search.py
```

## Test Cases

The test includes several scenarios:

1. Correct answer with proper format
2. Incorrect answer with proper format
3. Correct answer with improper format (missing tags)
4. Multiple answers with the last one correct
5. Answer with thinking tags
6. Invalid JSON in tool call

Each test case demonstrates different aspects of the parsing and scoring mechanism.

## Understanding the Results

The test output shows:
- The original prompt and response
- The extracted answer after parsing
- The calculated score
- An explanation of how the score was determined

For step rewards, it shows:
- Different types of responses (answer, empty tool, error tool, normal tool)
- The corresponding reward values
- Explanations of how rewards are calculated

This helps understand how the search environment evaluates agent responses and provides feedback during training.