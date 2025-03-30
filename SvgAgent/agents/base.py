from typing import Optional, Dict, Any
from agents import Agent, Runner
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel
import asyncio
from ..config import API_BASE_URL, API_KEY, MODEL_NAME, MAX_RETRY_ATTEMPTS, RETRY_DELAY


class BaseAgent:
    def __init__(self, name: str, instructions: str):
        self.client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        self.model = OpenAIChatCompletionsModel(
            model=MODEL_NAME,
            openai_client=self.client,
        )
        self.agent = Agent(name=name, instructions=instructions, model=self.model)

    async def execute(self, prompt: str, retry_count: int = 0) -> Optional[str]:
        """执行智能体任务，包含重试机制"""
        try:
            response = await Runner.run(self.agent, prompt)
            return response
        except Exception as e:
            if retry_count < MAX_RETRY_ATTEMPTS:
                await asyncio.sleep(RETRY_DELAY)
                return await self.execute(prompt, retry_count + 1)
            raise e

    def get_agent_info(self) -> Dict[str, Any]:
        """获取智能体信息"""
        return {
            "name": self.agent.name,
            "instructions": self.agent.instructions,
        }
