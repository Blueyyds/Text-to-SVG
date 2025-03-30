import tempfile
import os
import webbrowser
from typing import Optional
import cairosvg


class SVGPreview:
    @staticmethod
    def save_temp_svg(svg_code: str) -> str:
        """保存SVG到临时文件"""
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_file:
            tmp_file.write(svg_code.encode())
            return tmp_file.name

    @staticmethod
    def save_temp_png(svg_code: str) -> str:
        """将SVG转换为PNG并保存"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            cairosvg.svg2png(bytestring=svg_code.encode(), write_to=tmp_file.name)
            return tmp_file.name

    @staticmethod
    def create_preview_html(svg_code: str) -> str:
        """创建预览HTML"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SVG Preview</title>
            <style>
                body {{
                    margin: 0;
                    padding: 20px;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    background: #f0f0f0;
                }}
                .svg-container {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
            </style>
        </head>
        <body>
            <div class="svg-container">
                {svg_code}
            </div>
        </body>
        </html>
        """
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_file:
            tmp_file.write(html.encode())
            return tmp_file.name

    @staticmethod
    def preview(svg_code: str, format: str = "html") -> Optional[str]:
        """预览SVG"""
        try:
            if format == "html":
                preview_path = SVGPreview.create_preview_html(svg_code)
            elif format == "svg":
                preview_path = SVGPreview.save_temp_svg(svg_code)
            elif format == "png":
                preview_path = SVGPreview.save_temp_png(svg_code)
            else:
                raise ValueError(f"Unsupported format: {format}")

            # 在浏览器中打开预览
            webbrowser.open(f"file://{preview_path}")
            return preview_path

        except Exception as e:
            print(f"Preview failed: {str(e)}")
            return None
