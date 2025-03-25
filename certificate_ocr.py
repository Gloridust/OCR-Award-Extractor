import os
import json
import cv2
import numpy as np
import logging
from paddleocr import PaddleOCR
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
import re
from pathlib import Path
import argparse

# 配置日志记录
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CertificateOCR:
    def __init__(self, img_dir="data/img/", result_dir="data/result/", use_llm=True):
        """
        初始化 CertificateOCR 类
        
        Args:
            img_dir (str): 包含证书图像的目录
            result_dir (str): 保存 JSON 结果的目录
            use_llm (bool): 是否使用大语言模型
        """
        self.img_dir = img_dir
        self.result_dir = result_dir
        self.use_llm = use_llm
        
        # 创建目录（如果不存在）
        Path(img_dir).mkdir(parents=True, exist_ok=True)
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化 PaddleOCR，使用中文语言模型
        logger.info("正在初始化 PaddleOCR...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch", 
                             use_gpu=torch.cuda.is_available())
        
        # 初始化大语言模型
        self.llm_available = False
        if use_llm:
            logger.info("正在初始化大语言模型...")
            try:
                model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto"
                )
                self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.llm_available = True
                logger.info("大语言模型初始化成功")
            except Exception as e:
                logger.error(f"大语言模型初始化失败: {e}")
                logger.warning("将使用备用简单提取方法")
    
    def enhance_image(self, image):
        """
        增强图像质量以获得更好的 OCR 结果
        
        Args:
            image (numpy.ndarray): 输入图像
            
        Returns:
            numpy.ndarray: 增强后的图像
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 应用自适应阈值处理
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 去噪
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        # 应用形态学操作以增强文本
        kernel = np.ones((1, 1), np.uint8)
        enhanced = cv2.dilate(denoised, kernel, iterations=1)
        
        return enhanced
        
    def process_image(self, image_path):
        """
        处理单个证书图像
        
        Args:
            image_path (str): 证书图像的路径
            
        Returns:
            dict: JSON 格式的结果
        """
        logger.info(f"正在处理图像: {image_path}")
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return {
                "status": "error",
                "message": "无法读取图像",
                "image_path": image_path
            }
        
        # 尝试多种预处理方法以获得最佳结果
        results = []
        
        # 原始图像
        result_original = self.ocr.ocr(image, cls=True)
        if result_original and len(result_original) > 0:
            results.append(result_original)
            
        # 增强图像
        enhanced = self.enhance_image(image)
        result_enhanced = self.ocr.ocr(enhanced, cls=True)
        if result_enhanced and len(result_enhanced) > 0:
            results.append(result_enhanced)
        
        # 如果没有检测到文本，返回错误
        if not results:
            logger.error(f"OCR 未能在图像中检测到文本: {image_path}")
            return {
                "status": "error",
                "message": "OCR 未能检测到文本",
                "image_path": image_path
            }
            
        # 选择检测到更多文本的结果
        selected_result = max(results, key=lambda r: sum(len(line) for line in r))
        
        # 提取文本和置信度
        text_lines = []
        confidence_scores = []
        
        for line in selected_result:
            for word_info in line:
                text = word_info[1][0]
                confidence = word_info[1][1]
                if text and len(text.strip()) > 0:
                    text_lines.append(text)
                    confidence_scores.append(confidence)
        
        # 计算平均置信度分数
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # 合并所有文本，保留适当的换行符
        full_text = "\n".join(text_lines)
        
        # 使用大语言模型提取结构化信息
        if self.llm_available:
            certificate_info = self.extract_with_llm(full_text)
        else:
            # 如果大语言模型不可用，使用备用提取方法
            certificate_info = self.simple_fallback_extract(full_text, text_lines)
        
        # 准备结果
        json_result = {
            "status": "success",
            "certificate_info": certificate_info,
            "ocr_confidence": round(avg_confidence, 4),
            "image_path": image_path,
            "ocr_text": full_text  # 添加OCR识别的原始文本
        }
        
        return json_result
    
    def extract_with_llm(self, text):
        """
        使用大语言模型从OCR文本中提取结构化信息
        
        Args:
            text (str): OCR提取的文本
            
        Returns:
            dict: 结构化的证书信息
        """
        try:
            # 构建提示词
            prompt = f"""
你是一位专业的证书信息提取助手。请帮我从以下证书文本中提取关键信息。
请严格按照JSON格式输出以下字段:
1. competition_name: 竞赛名称
2. award_level: 奖项级别（如一等奖、二等奖、金奖等）
3. project_name: 项目名称
4. people: 包含两个列表
   - winner: 获奖者姓名列表
   - teacher: 指导教师姓名列表

以下是证书的OCR识别文本:
```
{text}
```

只输出JSON格式结果，不要包含任何解释。遵循以下格式：
{{
  "competition_name": "竞赛名称",
  "award_level": "奖项级别",
  "project_name": "项目名称",
  "people": {{
    "winner": ["获奖者1", "获奖者2", ...],
    "teacher": ["指导教师1", "指导教师2", ...]
  }}
}}

如果某个字段无法提取，请将其设置为空字符串或空列表。
请全面分析证书文本，识别竞赛名称时注意查找包含"大赛"、"比赛"、"竞赛"、"挑战赛"等关键词的完整名称；识别奖项级别时注意查找包含"等奖"、"金奖"、"银奖"等级别词语；识别项目名称时注意引号内的文本或带有连字符的标题；识别人员时区分获奖者和指导教师。
"""

            # 使用大语言模型分析文本
            messages = [
                {"role": "system", "content": "你是一个专业的证书信息提取助手，擅长结构化分析OCR文本并以JSON格式输出。"},
                {"role": "user", "content": prompt}
            ]
            
            text_input = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.llm_tokenizer([text_input], return_tensors="pt").to(self.llm_model.device)
            
            generated_ids = self.llm_model.generate(
                **model_inputs,
                max_new_tokens=1024,
                temperature=0.1,  # 使用较低的温度以获得更确定的输出
                do_sample=False   # 禁用采样以获得确定性输出
            )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 尝试从响应中提取JSON部分
            json_matches = re.findall(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_matches:
                json_str = json_matches[0]
            else:
                # 尝试查找JSON对象，假设它是以{开始，以}结束的
                json_matches = re.findall(r'({.*})', response, re.DOTALL)
                if json_matches:
                    json_str = json_matches[0]
                else:
                    json_str = response
            
            # 从响应中解析JSON
            try:
                result = json.loads(json_str)
                # 格式验证和修复
                result = self.validate_and_fix_result(result)
                logger.info("LLM成功提取信息")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"LLM返回的JSON无效: {e}")
                logger.error(f"原始响应: {response}")
                # 使用简单提取方法作为后备方案
                return self.simple_fallback_extract(text, text.split('\n'))
                
        except Exception as e:
            logger.error(f"LLM提取过程中出错: {e}")
            # 使用简单提取方法作为后备方案
            return self.simple_fallback_extract(text, text.split('\n'))
    
    def validate_and_fix_result(self, result):
        """
        验证并修复LLM返回的结果格式
        
        Args:
            result (dict): LLM返回的结果
            
        Returns:
            dict: 验证并修复后的结果
        """
        # 确保所有必要的键都存在
        expected_keys = ["competition_name", "award_level", "project_name", "people"]
        for key in expected_keys:
            if key not in result:
                result[key] = "" if key != "people" else {"winner": [], "teacher": []}
        
        # 确保people字典具有正确的结构
        if "people" in result:
            if not isinstance(result["people"], dict):
                result["people"] = {"winner": [], "teacher": []}
            else:
                if "winner" not in result["people"]:
                    result["people"]["winner"] = []
                if "teacher" not in result["people"]:
                    result["people"]["teacher"] = []
        
        # 确保字符串字段确实是字符串
        for key in ["competition_name", "award_level", "project_name"]:
            if not isinstance(result[key], str):
                result[key] = str(result[key]) if result[key] is not None else ""
        
        # 确保列表字段确实是列表
        for key in ["winner", "teacher"]:
            if not isinstance(result["people"][key], list):
                result["people"][key] = [result["people"][key]] if result["people"][key] else []
        
        # 清除可能的重复项
        result["people"]["winner"] = list(set(result["people"]["winner"]))
        result["people"]["teacher"] = list(set(result["people"]["teacher"]))
        
        # 确保列表中的所有项都是字符串
        result["people"]["winner"] = [str(item) for item in result["people"]["winner"] if item]
        result["people"]["teacher"] = [str(item) for item in result["people"]["teacher"] if item]
        
        return result
    
    def simple_fallback_extract(self, full_text, text_lines):
        """
        使用简单关键词方法从OCR文本中提取结构化信息，作为大语言模型的备用方案
        
        Args:
            full_text (str): 完整OCR文本
            text_lines (list): 文本行列表
            
        Returns:
            dict: 结构化的证书信息
        """
        # 初始化结果结构
        info = {
            "competition_name": "",
            "award_level": "",
            "project_name": "",
            "people": {
                "winner": [],
                "teacher": []
            }
        }
        
        competition_keywords = ["大赛", "比赛", "竞赛", "挑战赛"]
        award_keywords = ["一等奖", "二等奖", "三等奖", "特等奖", "金奖", "银奖", "铜奖", "优秀奖"]
        
        # 简单检测竞赛名称 - 包含关键词的最长行
        competition_lines = []
        for line in text_lines:
            if any(keyword in line for keyword in competition_keywords):
                competition_lines.append(line)
        
        if competition_lines:
            info["competition_name"] = max(competition_lines, key=len)
        
        # 简单检测奖项级别 - 包含关键词的行
        for line in text_lines:
            for keyword in award_keywords:
                if keyword in line:
                    info["award_level"] = keyword
                    break
            if info["award_level"]:
                break
        
        # 简单检测项目名称 - 查找引号内的内容
        project_matches = re.findall(r'[《""](.+?)[》""]', full_text)
        if project_matches:
            info["project_name"] = project_matches[0]
        
        # 简单检测人员信息
        for i, line in enumerate(text_lines):
            # 检测获奖者
            if "获奖学生" in line or "负责人" in line or "队员" in line or "获奖者" in line:
                parts = line.split("：")
                if len(parts) > 1 and parts[1].strip():
                    names = re.split(r'[、，,；;]', parts[1])
                    info["people"]["winner"].extend([n.strip() for n in names if n.strip()])
            
            # 检测教师
            if "指导教师" in line or "导师" in line:
                parts = line.split("：")
                if len(parts) > 1 and parts[1].strip():
                    names = re.split(r'[、，,；;]', parts[1])
                    info["people"]["teacher"].extend([n.strip() for n in names if n.strip()])
        
        # 清理文本
        info["competition_name"] = self.clean_text(info["competition_name"])
        info["award_level"] = self.clean_text(info["award_level"])
        info["project_name"] = self.clean_text(info["project_name"])
        
        return info
    
    def clean_text(self, text):
        """
        清理文本字符串
        
        Args:
            text (str): 输入文本
            
        Returns:
            str: 清理后的文本
        """
        if not text:
            return ""
            
        # 移除多余的空格
        text = re.sub(r"\s+", " ", text).strip()
        
        # 移除不需要的前后缀
        text = re.sub(r"^(?:关于|获得|授予)(.+)$", r"\1", text)
        
        # 移除引号（如果仍然存在）
        text = re.sub(r"^[\"\'《](.+?)[\"\'》]$", r"\1", text).strip()
        
        return text
    
    def process_all_images(self):
        """
        处理指定目录中的所有证书图像
        
        Returns:
            list: JSON 格式结果的列表
        """
        results = []
        
        # 获取目录中的所有图像文件
        image_files = [f for f in os.listdir(self.img_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        if not image_files:
            logger.warning(f"目录 {self.img_dir} 中未找到图像文件")
            return results
        
        for image_file in image_files:
            image_path = os.path.join(self.img_dir, image_file)
            result = self.process_image(image_path)
            results.append(result)
            
            # 保存单个结果
            result_path = os.path.join(self.result_dir, f"{os.path.splitext(image_file)[0]}.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"已处理图像: {image_file}")
        
        # 保存合并结果
        combined_result_path = os.path.join(self.result_dir, "all_results.json")
        with open(combined_result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"已处理总共 {len(results)} 个图像")
        return results

def main():
    """主程序入口点"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='证书OCR信息提取工具')
    parser.add_argument('--img_dir', type=str, default="data/img/", help='包含证书图像的目录路径')
    parser.add_argument('--result_dir', type=str, default="data/result/", help='保存结果的目录路径')
    parser.add_argument('--no_llm', action='store_true', help='不使用大语言模型，而是使用简单关键词提取方法')
    
    args = parser.parse_args()
    
    # 创建OCR处理器实例，默认启用大语言模型
    ocr = CertificateOCR(img_dir=args.img_dir, result_dir=args.result_dir, use_llm=not args.no_llm)
    results = ocr.process_all_images()
    
    # 记录摘要
    success_count = sum(1 for result in results if result.get("status") == "success")
    error_count = len(results) - success_count
    
    logger.info(f"处理了 {len(results)} 个图像。成功: {success_count}, 错误: {error_count}")
    
    # 打印示例结果（如果有）
    if results and len(results) > 0 and results[0].get("status") == "success":
        logger.info("示例结果:")
        print(json.dumps(results[0], ensure_ascii=False, indent=2))
    
if __name__ == "__main__":
    main()