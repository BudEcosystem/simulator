"""Tool-calling and reasoning parsers, chat template (engine_processors.py:330-347)."""

from __future__ import annotations

from typing import Tuple

from ..context import Fragment, PlanContext
from .base import Module, register


@register
class ToolCalling(Module):
    name = "parsers.tool_calling"
    version = "v1"
    after = ("engine.base",)

    def applies(self, ctx: PlanContext) -> Tuple[bool, str]:
        if not ctx.intent.enable_tool_calling:
            return False, ""
        if not ctx.intent.tool_calling_parser_type:
            return (
                False,
                "tool calling asked for, but the engine record names no tool parser for this model",
            )
        return True, ""

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        f.arg("enable-auto-tool-choice", True, self.name, "tool calling on")
        f.arg(
            "tool-call-parser",
            ctx.intent.tool_calling_parser_type,
            self.name,
            "the engine record's parser for the model",
        )
        if ctx.intent.chat_template:
            f.arg(
                "chat-template", ctx.intent.chat_template, self.name, "the record's chat template"
            )
        return f


@register
class Reasoning(Module):
    name = "parsers.reasoning"
    version = "v1"
    after = ("engine.base",)

    def applies(self, ctx: PlanContext) -> Tuple[bool, str]:
        if not ctx.intent.enable_reasoning:
            return False, ""
        if not ctx.intent.reasoning_parser_type:
            return (
                False,
                "reasoning asked for, but the engine record names no reasoning parser for this model",
            )
        return True, ""

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        f.arg("reasoning-parser", ctx.intent.reasoning_parser_type, self.name, "reasoning on")
        return f
