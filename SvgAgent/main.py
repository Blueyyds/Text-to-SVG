import asyncio
from typing import Dict, Any
from agents.supervisor import SupervisorAgent
from agents.prompt_engineer import PromptEngineerAgent
from agents.svg_generator import SVGGeneratorAgent
from agents.svg_validator import SVGValidatorAgent
from utils.cache import Cache
from utils.preview import SVGPreview


class SVGAgentSystem:
    def __init__(self):
        self.supervisor = SupervisorAgent()
        self.prompt_engineer = PromptEngineerAgent()
        self.svg_generator = SVGGeneratorAgent()
        self.svg_validator = SVGValidatorAgent()
        self.cache = Cache()

    async def process_request(self, user_input: str) -> Dict[str, Any]:
        """处理用户请求"""
        # 1. 检查缓存
        cache_key = self.cache._get_cache_key(user_input)
        cached_result = self.cache.get(cache_key)
        if cached_result:
            print("Using cached result...")
            return cached_result

        try:
            # 2. Supervisor创建计划
            print("\n1. Creating plan...")
            plan = await self.supervisor.create_plan(user_input)

            # 3. Prompt Engineer优化提示
            print("\n2. Optimizing prompt...")
            optimized_prompt = await self.prompt_engineer.optimize_prompt(user_input, plan)

            # 4. SVG Generator生成代码
            print("\n3. Generating SVG...")
            svg_code = await self.svg_generator.generate_svg(optimized_prompt)

            # 5. SVG Validator验证
            print("\n4. Validating SVG...")
            validation_result = await self.svg_validator.validate_svg(svg_code)

            # 6. 如果验证失败，尝试修复
            if not validation_result["syntax_valid"] or not validation_result["render_valid"]:
                print("\nValidation failed, attempting to fix issues...")
                svg_code = await self.svg_generator.generate_svg(
                    optimized_prompt + "\nPlease fix these issues: " + str(validation_result["issues"])
                )
                validation_result = await self.svg_validator.validate_svg(svg_code)

            # 7. 创建预览
            print("\n5. Creating preview...")
            preview_path = SVGPreview.preview(svg_code)

            # 8. 整理结果
            result = {
                "plan": plan,
                "optimized_prompt": optimized_prompt,
                "svg_code": svg_code,
                "validation": validation_result,
                "preview_path": preview_path,
            }

            # 9. 缓存结果
            self.cache.set(cache_key, result)

            return result

        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return {"error": str(e), "status": "failed"}


async def main():
    # 创建系统实例
    system = SVGAgentSystem()

    # 示例用户输入
    user_input = "Please create an SVG of a simple animated circle that pulses."

    # 处理请求
    result = await system.process_request(user_input)

    # 打印结果
    print("\nFinal Results:")
    for key, value in result.items():
        if key != "svg_code":  # SVG代码可能很长，单独打印
            print(f"\n{key.upper()}:")
            print(value)

    print("\nSVG CODE:")
    print(result["svg_code"][:200] + "..." if len(result["svg_code"]) > 200 else result["svg_code"])


if __name__ == "__main__":
    asyncio.run(main())
