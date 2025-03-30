from .base import BaseAgent


class SVGGeneratorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="SVGGenerator",
            instructions="""You are an SVG generation expert responsible for:
            1. Creating valid SVG code based on optimized prompts
            2. Supporting basic shapes, paths, gradients, and animations
            3. Following SVG best practices and standards
            4. Implementing requested visual effects and interactions
            Ensure generated code is clean, efficient, and well-structured.""",
        )

    async def generate_svg(self, optimized_prompt: str) -> str:
        """生成SVG代码"""
        prompt = f"""Generate SVG code based on this specification:
        
        {optimized_prompt}
        
        Requirements:
        1. Use valid SVG syntax
        2. Include all necessary attributes and namespaces
        3. Implement animations if specified
        4. Add comments for complex sections
        5. Follow SVG best practices
        
        Return only the SVG code without any additional text."""

        return await self.execute(prompt)
