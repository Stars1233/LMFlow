#!/usr/bin/env python
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
from .base import ConversationTemplate, StringFormatter, TemplateComponent

YI1_5_TEMPLATE = ConversationTemplate(
    template_name="yi1_5",
    user_formatter=StringFormatter(
        template=[TemplateComponent(type="string", content="<|im_start|>user\n{{content}}<|im_end|>\n")]
    ),
    assistant_formatter=StringFormatter(
        template=[TemplateComponent(type="string", content="<|im_start|>assistant\n{{content}}<|im_end|>\n")]
    ),
    system_formatter=StringFormatter(template=[TemplateComponent(type="string", content="{{content}}")]),
)
