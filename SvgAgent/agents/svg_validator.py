import asyncio
from typing import Dict, Any
from .base import BaseAgent
from lxml import etree
import cairosvg
import tempfile
import os
from ..config import VALIDATION_TIMEOUT


class SVGValidatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="SVGValidator",
            instructions="""You are a validation specialist focused on:
            1. Checking SVG syntax and XML compliance
            2. Validating rendering effects
            3. Testing functionality and animations
            4. Providing detailed feedback on issues
            Ensure all SVG output meets quality standards and works across browsers.""",
        )

    async def validate_svg(self, svg_code: str) -> Dict[str, Any]:
        """验证SVG代码的完整性和正确性"""
        validation_results = {
            "syntax_valid": False,
            "render_valid": False,
            "issues": [],
            "suggestions": [],
            "warnings": [],
        }

        # 1. 基本语法检查
        try:
            # 确保代码是XML格式
            if not svg_code.strip().startswith("<?xml") and not svg_code.strip().startswith("<svg"):
                svg_code = f'<?xml version="1.0" encoding="UTF-8"?>\n{svg_code}'

            # 解析XML
            parser = etree.XMLParser(recover=True)
            tree = etree.fromstring(svg_code.encode(), parser=parser)

            # 检查根元素是否为svg
            if tree.tag != "{http://www.w3.org/2000/svg}svg" and tree.tag != "svg":
                validation_results["issues"].append("Root element is not <svg>")
            else:
                validation_results["syntax_valid"] = True

            # 检查必要的属性
            if "width" not in tree.attrib and "viewBox" not in tree.attrib:
                validation_results["warnings"].append("Missing width/viewBox attribute")
            if "height" not in tree.attrib and "viewBox" not in tree.attrib:
                validation_results["warnings"].append("Missing height/viewBox attribute")

        except Exception as e:
            validation_results["issues"].append(f"XML syntax error: {str(e)}")
            return validation_results

        # 2. 渲染测试
        if validation_results["syntax_valid"]:
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    cairosvg.svg2png(bytestring=svg_code.encode(), write_to=tmp_file.name)
                    validation_results["render_valid"] = True
                os.unlink(tmp_file.name)
            except Exception as e:
                validation_results["issues"].append(f"Render error: {str(e)}")

        # 3. 动画和交互元素检查
        try:
            # 检查动画元素
            animation_elements = tree.xpath(
                '//*[local-name()="animate" or local-name()="animateTransform" or '
                'local-name()="animateMotion" or local-name()="set"]'
            )
            if animation_elements:
                validation_results["warnings"].append(
                    f"Found {len(animation_elements)} animation elements - ensure they work across browsers"
                )

            # 检查脚本和事件处理
            script_elements = tree.xpath('//*[@*[contains(local-name(), "on")]]')
            if script_elements:
                validation_results["warnings"].append(
                    f"Found {len(script_elements)} event handlers - ensure proper event handling"
                )

        except Exception as e:
            validation_results["warnings"].append(f"Animation check error: {str(e)}")

        # 4. AI分析验证
        try:
            async with asyncio.timeout(VALIDATION_TIMEOUT):
                ai_analysis = await self._analyze_svg(svg_code, validation_results)
                validation_results.update(ai_analysis)
        except asyncio.TimeoutError:
            validation_results["warnings"].append("AI analysis timeout")
        except Exception as e:
            validation_results["warnings"].append(f"AI analysis error: {str(e)}")

        return validation_results

    async def _analyze_svg(self, svg_code: str, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """使用AI分析SVG代码的质量和潜在问题"""
        prompt = f"""Analyze this SVG code for potential issues and improvements:

        SVG Code:
        {svg_code}

        Current validation results:
        - Syntax valid: {current_results["syntax_valid"]}
        - Render valid: {current_results["render_valid"]}
        - Current issues: {current_results["issues"]}
        - Current warnings: {current_results["warnings"]}

        Please provide a detailed analysis including:
        1. Code quality assessment
        2. Browser compatibility concerns
        3. Performance optimization suggestions
        4. Accessibility considerations
        5. Best practices compliance

        Format the response as a structured report."""

        analysis = await self.execute(prompt)

        return {"ai_analysis": analysis, "ai_validation_complete": True}
