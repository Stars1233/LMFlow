#!/usr/bin/env python
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.


from .base import ConversationTemplate, ConversationTemplateForTool, StringFormatter, TemplateComponent

QWEN2_TEMPLATE = ConversationTemplate(
    template_name="qwen2",
    user_formatter=StringFormatter(
        template=[TemplateComponent(type="string", content="<|im_start|>user\n{{content}}<|im_end|>\n")]
    ),
    assistant_formatter=StringFormatter(
        template=[TemplateComponent(type="string", content="<|im_start|>assistant\n{{content}}<|im_end|>\n")]
    ),
    system_formatter=StringFormatter(
        template=[TemplateComponent(type="string", content="<|im_start|>system\n{{content}}<|im_end|>\n")]
    ),
    separator=TemplateComponent(type="string", content="\n"),
)


QWEN2_TEMPLATE_FOR_TOOL = ConversationTemplateForTool(
    template_name="qwen2_for_tool",
    user_formatter=StringFormatter(
        template=[TemplateComponent(type="string", content="<|im_start|>user\n{{content}}<|im_end|>\n")]
    ),
    function_formatter=StringFormatter(
        template=[TemplateComponent(type="string", content="<|im_start|>assistant\n{{content}}<|im_end|>\n")]
    ),
    observation_formatter=StringFormatter(
        template=[TemplateComponent(type="string", content="<|im_start|>tool\n{{content}}<|im_end|>\n")]
    ),
    assistant_formatter=StringFormatter(
        template=[TemplateComponent(type="string", content="<|im_start|>assistant\n{{content}}<|im_end|>\n")]
    ),
    system_formatter=StringFormatter(
        template=[TemplateComponent(type="string", content="<|im_start|>system\n{{content}}<|im_end|>\n")]
    ),
    separator=TemplateComponent(type="string", content="\n"),
)


QWEN2_5_TEMPLATE = (
    "{%- if tools %}"
    "{{- '<|im_start|>system\\n' }}"
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- messages[0]['content'] }}"
    "{%- else %}"
    "{{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}"
    "{%- endif %}"
    '{{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}'
    "{%- for tool in tools %}"
    '{{- "\\n" }}'
    "{{- tool | tojson }}"
    "{%- endfor %}"
    '{{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}'
    "{%- else %}"
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- for message in messages %}"
    '{%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}'
    '{%- if message.role == "assistant" %}'
    "{{- '<|im_start|>' + message.role + '\\n' }}"
    "{% generation %}"
    "{{ message.content + '<|im_end|>' + '\\n' }}"
    "{% endgeneration %}"
    "{%- else %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
    "{%- endif %}"
    '{%- elif message.role == "assistant" %}'
    "{{- '<|im_start|>' + message.role }}"
    "{%- if message.content %}"
    "{% generation %}"
    "{{- '\\n' + message.content }}"
    "{% endgeneration %}"
    "{%- endif %}"
    "{%- for tool_call in message.tool_calls %}"
    "{%- if tool_call.function is defined %}"
    "{%- set tool_call = tool_call.function %}"
    "{%- endif %}"
    "{% generation %}"
    '{{- \'\\n<tool_call>\\n{"name": "\' }}'
    "{{- tool_call.name }}"
    '{{- \'", "arguments": \' }}'
    "{{- tool_call.arguments | tojson }}"
    "{{- '}\\n</tool_call>' }}"
    "{% endgeneration %}"
    "{%- endfor %}"
    "{% generation %}"
    "{{- '<|im_end|>\\n' }}"
    "{% endgeneration %}"
    '{%- elif message.role == "tool" %}'
    '{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}'
    "{{- '<|im_start|>user' }}"
    "{%- endif %}"
    "{{- '\\n<tool_response>\\n' }}"
    "{{- message.content }}"
    "{{- '\\n</tool_response>' }}"
    '{%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}'
    "{{- '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)


QWEN2_5_1M_TEMPLATE = (
    "{%- if tools %}"
    "{{- '<|im_start|>system\\n' }}"
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- messages[0]['content'] }}"
    "{%- else %}"
    "{{- 'You are a helpful assistant.' }}"
    "{%- endif %}"
    '{{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}'
    "{%- for tool in tools %}"
    '{{- "\\n" }}'
    "{{- tool | tojson }}"
    "{%- endfor %}"
    '{{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}'
    "{%- else %}"
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- for message in messages %}"
    '{%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}'
    '{%- if message.role == "assistant" %}'
    "{{- '<|im_start|>' + message.role + '\\n' }}"
    "{% generation %}"
    "{{ message.content + '<|im_end|>' + '\\n' }}"
    "{% endgeneration %}"
    "{%- else %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
    "{%- endif %}"
    '{%- elif message.role == "assistant" %}'
    "{{- '<|im_start|>' + message.role }}"
    "{%- if message.content %}"
    "{% generation %}"
    "{{- '\\n' + message.content }}"
    "{% endgeneration %}"
    "{%- endif %}"
    "{%- for tool_call in message.tool_calls %}"
    "{%- if tool_call.function is defined %}"
    "{%- set tool_call = tool_call.function %}"
    "{%- endif %}"
    "{% generation %}"
    '{{- \'\\n<tool_call>\\n{"name": "\' }}'
    "{{- tool_call.name }}"
    '{{- \'", "arguments": \' }}'
    "{{- tool_call.arguments | tojson }}"
    "{{- '}\\n</tool_call>' }}"
    "{% endgeneration %}"
    "{%- endfor %}"
    "{% generation %}"
    "{{- '<|im_end|>\\n' }}"
    "{% endgeneration %}"
    '{%- elif message.role == "tool" %}'
    '{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}'
    "{{- '<|im_start|>user' }}"
    "{%- endif %}"
    "{{- '\\n<tool_response>\\n' }}"
    "{{- message.content }}"
    "{{- '\\n</tool_response>' }}"
    '{%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}'
    "{{- '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)


QWEN2_5_MATH_TEMPLATE = (
    "{%- if tools %}"
    "{{- '<|im_start|>system\\n' }}"
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- messages[0]['content'] }}"
    "{%- else %}"
    "{{- 'Please reason step by step, and put your final answer within \\\\boxed{}.' }}"
    "{%- endif %}"
    '{{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}'
    "{%- for tool in tools %}"
    '{{- "\\n" }}'
    "{{- tool | tojson }}"
    "{%- endfor %}"
    '{{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}'
    "{%- else %}"
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>system\\nPlease reason step by step, and put your final answer within \\\\boxed{}.<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- for message in messages %}"
    '{%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}'
    '{%- if message.role == "assistant" %}'
    "{{- '<|im_start|>' + message.role + '\\n' }}"
    "{% generation %}"
    "{{ message.content + '<|im_end|>' + '\\n' }}"
    "{% endgeneration %}"
    "{%- else %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
    "{%- endif %}"
    '{%- elif message.role == "assistant" %}'
    "{{- '<|im_start|>' + message.role }}"
    "{%- if message.content %}"
    "{% generation %}"
    "{{- '\\n' + message.content }}"
    "{% endgeneration %}"
    "{%- endif %}"
    "{%- for tool_call in message.tool_calls %}"
    "{%- if tool_call.function is defined %}"
    "{%- set tool_call = tool_call.function %}"
    "{%- endif %}"
    "{% generation %}"
    '{{- \'\\n<tool_call>\\n{"name": "\' }}'
    "{{- tool_call.name }}"
    '{{- \'", "arguments": \' }}'
    "{{- tool_call.arguments | tojson }}"
    "{{- '}\\n</tool_call>' }}"
    "{% endgeneration %}"
    "{%- endfor %}"
    "{% generation %}"
    "{{- '<|im_end|>\\n' }}"
    "{% endgeneration %}"
    '{%- elif message.role == "tool" %}'
    '{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}'
    "{{- '<|im_start|>user' }}"
    "{%- endif %}"
    "{{- '\\n<tool_response>\\n' }}"
    "{{- message.content }}"
    "{{- '\\n</tool_response>' }}"
    '{%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}'
    "{{- '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)


QWEN_QWQ_TEMPLATE = (
    "{%- if tools %}"
    "{{- '<|im_start|>system\\n' }}"
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- messages[0]['content'] }}"
    "{%- else %}"
    "{{- 'You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.' }}"
    "{%- endif %}"
    '{{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}'
    "{%- for tool in tools %}"
    '{{- "\\n" }}'
    "{{- tool | tojson }}"
    "{%- endfor %}"
    '{{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}'
    "{%- else %}"
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>system\\nYou are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- for message in messages %}"
    '{%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}'
    '{%- if message.role == "assistant" %}'
    "{{- '<|im_start|>' + message.role + '\\n' }}"
    "{% generation %}"
    "{{ message.content + '<|im_end|>' + '\\n' }}"
    "{% endgeneration %}"
    "{%- else %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
    "{%- endif %}"
    '{%- elif message.role == "assistant" %}'
    "{{- '<|im_start|>' + message.role }}"
    "{%- if message.content %}"
    "{% generation %}"
    "{{- '\\n' + message.content }}"
    "{% endgeneration %}"
    "{%- endif %}"
    "{%- for tool_call in message.tool_calls %}"
    "{%- if tool_call.function is defined %}"
    "{%- set tool_call = tool_call.function %}"
    "{%- endif %}"
    "{% generation %}"
    '{{- \'\\n<tool_call>\\n{"name": "\' }}'
    "{{- tool_call.name }}"
    '{{- \'", "arguments": \' }}'
    "{{- tool_call.arguments | tojson }}"
    "{{- '}\\n</tool_call>' }}"
    "{% endgeneration %}"
    "{%- endfor %}"
    "{% generation %}"
    "{{- '<|im_end|>\\n' }}"
    "{% endgeneration %}"
    '{%- elif message.role == "tool" %}'
    '{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}'
    "{{- '<|im_start|>user' }}"
    "{%- endif %}"
    "{{- '\\n<tool_response>\\n' }}"
    "{{- message.content }}"
    "{{- '\\n</tool_response>' }}"
    '{%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}'
    "{{- '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)

QWEN3_TEMPLATE = (
    "{%- if tools %}"
    "{{- '<|im_start|>system\\n' }}"
    "{%- if messages[0].role == 'system' %}"
    "{{- messages[0].content + '\\n\\n' }}"
    "{%- endif %}"
    '{{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}'
    "{%- for tool in tools %}"
    '{{- "\\n" }}'
    "{{- tool | tojson }}"
    "{%- endfor %}"
    '{{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}'
    "{%- else %}"
    "{%- if messages[0].role == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}"
    "{%- for message in messages[::-1] %}"
    "{%- set index = (messages|length - 1) - loop.index0 %}"
    "{%- if ns.multi_step_tool and message.role == \"user\" and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}"
    "{%- set ns.multi_step_tool = false %}"
    "{%- set ns.last_query_index = index %}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- for message in messages %}"
    '{%- if (message.role == "user") or (message.role == "system" and not loop.first) %}'
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
    '{%- elif message.role == "assistant" %}'
    "{% generation %}"
    "{%- set content = message.content %}"
    "{%- set reasoning_content = '' %}"
    "{%- if message.reasoning_content is defined and message.reasoning_content is not none %}"
    "{%- set reasoning_content = message.reasoning_content %}"
    "{%- else %}"
    "{%- if '</think>' in message.content %}"
    "{%- set content = message.content.split('</think>')[-1].lstrip('\\n') %}"
    "{%- set reasoning_content = message.content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- if loop.index0 > ns.last_query_index %}"
    "{%- if loop.last or (not loop.last and reasoning_content) %}"
    "{{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}"
    "{%- else %}"
    "{{- '<|im_start|>' + message.role + '\\n' + content }}"
    "{%- endif %}"
    "{%- else %}"
    "{{- '<|im_start|>' + message.role + '\\n' + content }}"
    "{%- endif %}"
    "{%- if message.tool_calls %}"
    "{%- for tool_call in message.tool_calls %}"
    "{%- if (loop.first and content) or (not loop.first) %}"
    "{{- '\\n' }}"
    "{%- endif %}"
    "{%- if tool_call.function %}"
    "{%- set tool_call = tool_call.function %}"
    "{%- endif %}"
    '{{- \'<tool_call>\\n{"name": "\' }}'
    "{{- tool_call.name }}"
    '{{- \'", "arguments": \' }}'
    "{%- if tool_call.arguments is string %}"
    "{{- tool_call.arguments }}"
    "{%- else %}"
    "{{- tool_call.arguments | tojson }}"
    "{%- endif %}"
    "{{- '}\\n</tool_call>' }}"
    "{%- endfor %}"
    "{%- endif %}"
    "{{- '<|im_end|>\\n' }}"
    "{% endgeneration %}"
    '{%- elif message.role == "tool" %}'
    '{%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}'
    "{{- '<|im_start|>user' }}"
    "{%- endif %}"
    "{{- '\\n<tool_response>\\n' }}"
    "{{- message.content }}"
    "{{- '\\n</tool_response>' }}"
    '{%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}'
    "{{- '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}"
    "{{- '<think>\\n\\n</think>\\n\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
)
