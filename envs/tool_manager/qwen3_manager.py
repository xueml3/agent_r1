import re
import copy
import json
import json5
import asyncio
import traceback
from ast import literal_eval
from omegaconf import OmegaConf
from envs.tool_manager.base_manager import ToolManager
# from envs.utils.mcp_manager import MCPManager, BaseTool
from typing import Union, List, Tuple, Optional, Any, Dict
from envs.utils.util import ToolServiceError, DocParserError
from envs.utils.mcp_manager import MCPManager as SSEMCPManager
from qwen_agent.tools import TOOL_REGISTRY, MCPManager, BaseTool
from qwen_agent.llm.schema import ASSISTANT, SYSTEM, USER, FUNCTION, ContentItem
from envs.utils.concurrency_limiter import ConcurrencyLimiter
from envs.utils.async_mcp_manager import AsyncMCPManager


def parse_mcp_tools_config(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        # 使用 literal_eval 安全地解析 Python 字面量
        data = literal_eval(content)
        return data
    except Exception as e:
        print(f"解析错误: {e}")
        return None


class QwenManager(ToolManager):    
    def __init__(self, verl_config):
        if isinstance(verl_config, dict):
            verl_config = OmegaConf.create(verl_config)
        super().__init__(verl_config)
        self.generate_cfg = {
            'fncall_prompt_type': 'nous',
            'function_choice': 'auto',  # 注释掉这行
            'parallel_function_calls': False,
            'lang': 'en',
            'max_input_tokens': 10000
        }
        
        # 创建并发限制器
        if self.verl_config.enable_limiter:
            global_limit = getattr(verl_config, 'max_concurrency', 100)
            self._limiter = ConcurrencyLimiter(global_limit=global_limit)
        else:
            self._limiter = None

    def get_tool(self, name_or_short_name: str):
        """通过名称或简写获取工具
        
        Args:
            name_or_short_name: 工具名称或简写
            
        Returns:
            找到的工具，如果没找到则返回None
        """
        name_or_short_name = str(name_or_short_name)
        return self.tool_map.get(name_or_short_name, None)
    
    @property
    def all_tools(self):
        """获取所有工具
        
        Returns:
            所有工具的列表
        """
        return self.tool_map

    def _build_tools(self):
        config_path = self.verl_config.config_path
        if config_path is not None:
            function_list = parse_mcp_tools_config(config_path)

            if function_list:
                for tool in function_list:
                    self._init_tool(tool)
            
            self.functions = [func.function for func in self.tool_map.values()]
        else:
            print("The config_path is None!")
            self.functions = []

    async def execute_all_tools_with_limiter(self, actions, tools):
        """并行执行工具调用"""
        async def execute_single_tool(tool):
            """执行单个工具调用"""
            try:
                tool_name = tool.get("name", "default")
            
                async with self._limiter.limit(tool_name):
                    # 执行工具调用
                    result = await self._call_tool(tool_name, tool.get("args", ""))
                    return result
                
            except Exception as e:
                print(f"工具调用失败: {e}")
                return f"# 工具调用失败: {str(e)}"
        
        tasks = []
        for action, tool_list in zip(actions, tools):
            if action == "actions":
                for tool in tool_list:
                    tasks.append(execute_single_tool(tool))
        
        results = await asyncio.gather(*tasks)
        return results
    
    async def execute_all_tools(self, actions, tool_list):
        """异步并行执行所有工具列表
        
        Args:
            tool_list: 工具列表的列表
            
        Returns:
            所有工具执行结果的列表
        """
        
        # 并行执行每个工具列表
        tasks = []
        for temp_action, temp_tool_list in zip(actions, tool_list):
            tasks.append(self._execute_tool_batch(temp_action, temp_tool_list))
        
        results = await asyncio.gather(*tasks)
        
        return results
        
    async def _execute_tool_batch(self, action, tools):
        """异步并行执行一批工具
        
        Args:
            tools: 工具列表
            
        Returns:
            工具执行结果的列表
        """        
        async def execute_single_tool(tool):
            tool_instance = self.get_tool(tool["name"])
            args = tool["args"]
            if tool_instance is not None:
                try:
                    args = json.loads(args)
                except Exception as e:
                    pass

                if type(args) is dict:
                    try:
                        # 使用asyncio.to_thread包装self._call_tool以保持异步特性
                        tool_result = await asyncio.to_thread(
                            self._call_tool, 
                            tool["name"], json.dumps(args, ensure_ascii=False, indent=4)
                        )
                        result = """# Execute the tool {} successed
  - The args are: {}
  - The result is:
{}""".format(tool["name"], args, tool_result)
                    except Exception as e:
                        result = """# Execute the tool {} failed
  - The original args are: {}
  - Error message:
{}""".format(tool["name"], args, str(e))
                elif type(args) is str:
                    # Json decode error: xxx
                    result = args
                else:
                    result = 'Unexpected type of args: {} (args: {})'.format(type(args), args)
            else:
                if tool['name'] == '<empty>':
                    result = args
                else:
                    result = "# Failed to find the tool {} in the tool map".format(tool['name'])

            return result
        
        if action == 'answer':
            # 'tools' is a str (the answer)
            results = {'role': 'assitant', 'content': tools}
        elif action == 'error':
            # 'error' only occurs when there is no 'actions' tag or there is no 'action' tag after extraction
            # ('Cannot extract the actions tag' or 'There is no action after extraction')
            results = {'role': 'assitant', 'content': """# Extract the tools failed due to: {}""".format(tools)}
        elif action == 'actions':
            # 'tools' is the list of the 'Tool' instances
            tasks = [execute_single_tool(temp_tool) for temp_tool in tools]
            tool_results = await asyncio.gather(*tasks)
            results = [{'role': 'tool', 'content': temp_tool_result} for temp_tool_result in tool_results]
        else:
            raise ValueError('Unexpected action: {}'.format(action))

        return results
    
    def _call_tool(self, tool_name: str, tool_args: Union[str, dict] = '{}', **kwargs) -> Union[str, List[ContentItem]]:
        """The interface of calling tools for the agent.

        Args:
            tool_name: The name of one tool.
            tool_args: Model generated or user given tool parameters.

        Returns:
            The output of tools.
        """
        if tool_name not in self.tool_map:
            return f'Tool {tool_name} does not exists.'
        tool = self.get_tool(tool_name)
        try:
            tool_result = tool.call(tool_args, **kwargs)
        except (ToolServiceError, DocParserError) as ex:
            raise ex
        except Exception as ex:
            exception_type = type(ex).__name__
            exception_message = str(ex)
            traceback_info = ''.join(traceback.format_tb(ex.__traceback__))
            error_message = f'An error occurred when calling tool `{tool_name}`:\n' \
                            f'{exception_type}: {exception_message}\n' \
                            f'Traceback:\n{traceback_info}'
            return error_message

        if isinstance(tool_result, str):
            return tool_result
        else:
            return json.dumps(tool_result, ensure_ascii=False, indent=4)

    def _init_tool(self, tool: Union[str, BaseTool]):
        if isinstance(tool, BaseTool):
            tool_name = tool.name
            self.tool_map[tool_name] = tool
        elif isinstance(tool, dict) and 'mcpServers' in tool:
            print(f'MCP is using {self.verl_config.mcp_mode} mode')
            if self.verl_config.mcp_mode == 'sse':
                if self.verl_config.parallel_sse_tool_call.is_enabled:
                    tools = AsyncMCPManager(num_instances=self.verl_config.parallel_sse_tool_call.num_instances).initConfig(tool)
                else:
                    tools = SSEMCPManager().initConfig(tool)
            elif self.verl_config.mcp_mode == 'stdio':
                tools = MCPManager().initConfig(tool)
            else:
                raise ValueError(f"Unexpected mcp mode: {self.verl_config.mcp_mode}")
            
            for tool in tools:
                tool_name = tool.name
                self.tool_map[tool_name] = tool
        else:
            if isinstance(tool, dict):
                tool_name = tool['name']
                tool_cfg = tool
            else:
                tool_name = tool
                tool_cfg = None
            if tool_name not in TOOL_REGISTRY:
                raise ValueError(f'Tool {tool_name} is not registered.')

            self.tool_map[tool_name] = TOOL_REGISTRY[tool_name](tool_cfg)

        # select used tools within one sse link before tool learning
        if self.verl_config.mcp_mode == 'sse' and len(self.verl_config.tool_name_selected) != 0:
            try:
                self.tool_map = {each_tool_name: self.tool_map[each_tool_name] for each_tool_name in self.verl_config.tool_name_selected}
            except:
                raise ValueError('Selected tool names are not valid or available sse tool list error. Available tool names: {}'.format(self.tool_map.keys()))

    def execute_actions(self, responses: List[str]):
        actions, tools = [], []
        for response in responses:
            temp_action, temp_tool_list = self.parse_response(response_content=response)
            # temp_action: answer or tools
            # if temp_action is 'answer', temp_tool_list is the answer
            # else, temp_tool_list is the list of the 'Tool' instances
            actions.append(temp_action)
            tools.append(temp_tool_list)

        # 使用asyncio.run同步运行异步函数
        try:
            if self.verl_config.enable_limiter:
                assert self._limiter is not None, "Limiter is not enabled"
                tool_results = asyncio.run(self.execute_all_tools_with_limiter(actions, tools))
            else:
                tool_results = asyncio.run(self.execute_all_tools(actions, tools))
        except RuntimeError:
            # 如果事件循环已经在运行，则获取当前循环
            loop = asyncio.get_event_loop()
            if self.verl_config.enable_limiter:
                assert self._limiter is not None, "Limiter is not enabled"
                tool_results = loop.run_until_complete(self.execute_all_tools_with_limiter(actions, tools))
            else:
                tool_results = loop.run_until_complete(self.execute_all_tools(actions, tools))
        
        return actions, tool_results
    
    def parse_response(self, response_content: str) -> Optional[Dict[str, Any]]:
        """执行动作
        
        Args:
            response_content: 响应文本
            
        Returns:
            解析后的动作信息，包含name和args（如果存在）
            如果没有动作标签或格式不正确，返回None
        """
        # 提取answers
        if_answer, answer = self.parse_end_flag(response_content)
        if if_answer:
            return 'answer', answer
        
        # 提取tools
        tools = self.parse_tools(response_content)
        if type(tools) == list:
            return 'actions', tools
        else:
            assert type(tools) == str
            # if the response is not a tool call, it is an answer
            return 'answer', tools
    
    def parse_end_flag(self, response_content: str) -> bool:
        answer = None
        answer_section = re.findall(r'(<answer>.*?</answer>)', response_content, re.DOTALL)
        if len(answer_section) > 0:
            answer = answer_section[-1]
            return True, answer
        else:
            return False, None
    
    def parse_tools(self, response: str):
        parsed_tools = []
        i = response.find('<tool_call>')
        # If no function call:
        if i < 0:
            j = response.find('</tool_call>')
            if j < 0:
                return response
            else:
                parsed_tools.append({
                        "name": "<error>",
                        "args": "# Extract the tool name failed"
                    })
                return parsed_tools

        # split tool-call to separate assistant msg
        tool_call_list = response.split('<tool_call>')
        pre_thought = tool_call_list[0].strip()
        for txt in tool_call_list[1:]:
            if not txt.strip():
                continue

            if '</tool_call>' not in txt:
                # incomplete </tool_call>: This is to better represent incomplete tool calls in streaming output
                fn_name = '<empty>'
                fn_args = """# Extract the tool name failed"""
                parsed_tools.append(
                    {
                        "name": fn_name,
                        "args": fn_args,
                    }
                )
            else:
                one_tool_call_txt = txt.split('</tool_call>')
                try:
                    # 检查分割后是否有有效内容
                    if not one_tool_call_txt[0].strip():
                        raise ValueError("Empty tool call content")
                        
                    # 尝试解析JSON
                    fn = json5.loads(one_tool_call_txt[0].strip())
                    
                    # 检查必须字段是否存在
                    if type(fn) is not dict or 'name' not in fn or 'arguments' not in fn:
                        raise KeyError("Missing required fields")
                    
                    # 解析成功的情况
                    parsed_tools.append({
                        "name": fn['name'],
                        "args": json.dumps(fn['arguments'], ensure_ascii=False, indent=4),
                    })
                
                except (IndexError, KeyError, ValueError) as e:
                    # 所有可能的错误类型处理
                    parsed_tools.append({
                        "name": "<empty>",
                        "args": "# Extract the tool name failed"
                    })
        
        if len(parsed_tools) == 0 :
            # <tool_call> is last token
            fn_name = '<empty>'
            fn_args = """# Extract the tool name failed"""
            parsed_tools.append(
                {
                    "name": fn_name,
                    "args": fn_args,
                })

        return parsed_tools
    
    def get_prompt(self, input_data, tokenizer, mode='initial', add_generation_prompt=True):
        assert mode in ['initial', 'tool_call', 'assistant_response'], 'Invalid mode: {}'.format(mode)
        base_chat = [
            {'role': SYSTEM, 'content': 'base'},
            {'role': USER, 'content': 'base'},
        ]
        base_prompt = tokenizer.apply_chat_template(
            conversation=base_chat,
            tools=self.functions,
            tokenize=False, add_generation_prompt=False
        )

        if mode == 'initial':
            chat = input_data
            prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=chat, tokenize=False, tools=self.functions, 
                add_generation_prompt=add_generation_prompt, enable_thinking=self.verl_config.enable_thinking
            )
        elif mode in ['tool_call', 'assistant_response']:
            # NOTE: the assistant response might not be used
            role = 'tool' if mode == 'tool_call' else ASSISTANT
            if type(input_data) == str:
                chat = {'role': role, 'content': input_data}
            elif type(input_data) == list:
                chat = input_data
            else:
                raise ValueError('Unexpected type of input_data {} ({})'.format(type(input_data), input_data))
            
            temp_prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=base_chat + chat, tools=self.functions, 
                tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=self.verl_config.enable_thinking
            )
            prompt_with_chat_template = temp_prompt_with_chat_template.replace(base_prompt, '')
        else:
            raise ValueError('Invalid mode: {}'.format(mode))
        
        return prompt_with_chat_template
