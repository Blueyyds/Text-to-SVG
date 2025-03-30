from .base import BaseAgent
from ..config import DEFAULT_SVG_WIDTH, DEFAULT_SVG_HEIGHT


class PromptEngineerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="PromptEngineer",
            instructions="""You are a prompt engineering agent specialized in:
            1. Optimizing user text descriptions for SVG generation
            2. Adding necessary SVG attributes (dimensions, colors, coordinate system)
            3. Ensuring all required details are specified
            4. Maintaining consistency in style and format
            Focus on creating clear, detailed, and technically precise prompts.""",
        )

    async def optimize_prompt(self, user_input: str, supervisor_plan: str) -> str:
        """优化用户输入的提示"""
        prompt = f"""Optimize this description for SVG generation:
        
        User Request: {user_input}
        
        Supervisor's Plan: {supervisor_plan}
        
        Please create a detailed specification including:
        1. SVG viewport dimensions (default: {DEFAULT_SVG_WIDTH}x{DEFAULT_SVG_HEIGHT})
        2. Color scheme and styling
        3. Animation specifications (if needed)
        4. Coordinate system and positioning
        5. Required SVG elements and attributes
        
        Format the response as a structured technical specification."""

        return await self.execute(prompt)
