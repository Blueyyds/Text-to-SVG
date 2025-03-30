from .base import BaseAgent


class SupervisorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Supervisor",
            instructions="""You are a supervisor agent responsible for:
            1. Coordinating the overall SVG generation process
            2. Receiving and understanding user text descriptions
            3. Assigning tasks to other agents
            4. Managing iteration loops and feedback
            Always maintain clear communication and ensure all requirements are met.""",
        )

    async def create_plan(self, user_input: str) -> str:
        """创建处理计划"""
        prompt = f"""Analyze this user request and create a detailed plan:
        User Request: {user_input}
        
        Please provide:
        1. Task breakdown
        2. Required SVG elements
        3. Potential challenges
        4. Success criteria"""

        return await self.execute(prompt)
