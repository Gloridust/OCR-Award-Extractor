import os
import json
import cv2
import numpy as np
import logging
from paddleocr import PaddleOCR
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re
from pathlib import Path

# 配置日志记录
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CertificateOCR:
    def __init__(self, img_dir="data/img/", result_dir="data/result/"):
        """
        初始化 CertificateOCR 类
        
        Args:
            img_dir (str): 包含证书图像的目录
            result_dir (str): 保存 JSON 结果的目录
        """
        self.img_dir = img_dir
        self.result_dir = result_dir
        
        # 创建目录（如果不存在）
        Path(img_dir).mkdir(parents=True, exist_ok=True)
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化 PaddleOCR，使用中文语言模型
        logger.info("正在初始化 PaddleOCR...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch", 
                             use_gpu=torch.cuda.is_available())
    
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
        
        # 从文本中提取结构化信息
        certificate_info = self.extract_information(full_text, text_lines)
        
        # 准备结果
        json_result = {
            "status": "success",
            "certificate_info": certificate_info,
            "ocr_confidence": round(avg_confidence, 4),
            "image_path": image_path
        }
        
        return json_result
    
    def extract_competition_name(self, text, text_lines):
        """
        提取竞赛名称
        
        Args:
            text (str): 完整文本
            text_lines (list): 单独文本行列表
            
        Returns:
            str: 提取的竞赛名称
        """
        # 竞赛名称模式
        competition_patterns = [
            r"第[\u4e00-\u9fa5\d]+届[\u4e00-\u9fa5\d]+(?:大学生|青年|全国|国际)?[\u4e00-\u9fa5\d]+(?:大赛|比赛|竞赛|创新创业大赛|创客大赛|挑战赛)",
            r"[\u4e00-\u9fa5\d]+(?:大学生|青年|全国|国际)?[\u4e00-\u9fa5\d]+(?:大赛|比赛|竞赛|创新创业大赛|创客大赛|挑战赛)",
            r"(?:大学生|青年|全国|国际)[\u4e00-\u9fa5\d]+(?:大赛|比赛|竞赛|创新创业大赛|创客大赛|挑战赛)"
        ]
        
        for pattern in competition_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # 按长度排序以获取最完整的匹配
                matches.sort(key=len, reverse=True)
                return matches[0]
                
        # 如果未找到竞赛名称，检查可能包含竞赛名称的单独行
        for line in text_lines:
            if any(keyword in line for keyword in ["大赛", "比赛", "竞赛", "挑战赛"]) and len(line) > 5 and len(line) < 40:
                return line.strip()
                
        return ""
        
    def extract_award_level(self, text, text_lines):
        """
        提取奖项级别
        
        Args:
            text (str): 完整文本
            text_lines (list): 单独文本行列表
            
        Returns:
            str: 提取的奖项级别
        """
        # 奖项级别模式
        award_patterns = [
            r"[国省市区校](?:家|级|赛区)?(?:特等|一等|二等|三等|优秀|金|银|铜)?奖",
            r"(?:特等|一等|二等|三等|金|银|铜)奖",
            r"(?:优秀|突出贡献)奖",
            r"(?:第[一二三]名)"
        ]
        
        for pattern in award_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # 使用最长的匹配
                matches.sort(key=len, reverse=True)
                return matches[0]
                
        # 如果未找到奖项级别，检查单独行
        for line in text_lines:
            if "奖" in line and len(line) < 15:
                return line.strip()
                
        return ""
        
    def extract_project_name(self, text, text_lines):
        """
        提取项目名称
        
        Args:
            text (str): 完整文本
            text_lines (list): 单独文本行列表
            
        Returns:
            str: 提取的项目名称
        """
        # 项目名称模式
        project_patterns = [
            r"[《\"](.+?)[》\"]",  # 引号或特殊括号内的文本
            r"(?:项目|作品|获奖项目|获奖作品|题目)[：:]\s*([\u4e00-\u9fa5a-zA-Z0-9\-_+（）\(\)]+)",
            r"(.+?(?:-{1,2}|\s*[-—]\s*)[\u4e00-\u9fa5a-zA-Z0-9\-_+]+)"  # 项目名称通常包含连字符
        ]
        
        for pattern in project_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # 使用不太长的最长匹配
                valid_matches = [m for m in matches if isinstance(m, str) and 3 < len(m) < 50]
                if valid_matches:
                    valid_matches.sort(key=len, reverse=True)
                    return valid_matches[0]
        
        # 检查以冒号结尾的模式，项目名称可能在下一行
        for i, line in enumerate(text_lines):
            if re.search(r"(?:项目|作品|获奖项目|获奖作品|题目)[：:]$", line) and i + 1 < len(text_lines):
                return text_lines[i+1].strip()
                
        # 在许多证书中，如果一行包含"--"或"—"，它很可能是一个项目名称
        for line in text_lines:
            if "--" in line or "—" in line or "-" in line:
                if len(line) > 5 and len(line) < 50 and not any(keyword in line for keyword in ["指导教师", "导师", "负责人"]):
                    return line.strip()
                    
        return ""
        
    def extract_people(self, text, text_lines):
        """
        提取人员信息（获奖者和教师）
        
        Args:
            text (str): 完整文本
            text_lines (list): 单独文本行列表
            
        Returns:
            dict: 包含获奖者和教师的字典
        """
        people = {
            "winner": [],
            "teacher": []
        }
        
        # 识别名单的模式
        winner_patterns = [
            r"(?:获奖学生|负责人|队员|参赛者|作者|获奖人|学生|成员)[：:]\s*(.*)",
            r"(?:获奖学生|负责人|队员|参赛者|作者|获奖人|学生|成员)[：:]\s*$"
        ]
        
        teacher_patterns = [
            r"(?:指导教师|导师|辅导教师|指导老师|指导员)[：:]\s*(.*)",
            r"(?:指导教师|导师|辅导教师|指导老师|指导员)[：:]\s*$"
        ]
        
        # 处理每一行以提取人员
        winner_section = False
        teacher_section = False
        
        for i, line in enumerate(text_lines):
            # 标记部分开始
            if re.search(r"(?:获奖学生|负责人|队员|参赛者|作者|获奖人|学生|成员)[：:]", line):
                winner_section = True
                teacher_section = False
            elif re.search(r"(?:指导教师|导师|辅导教师|指导老师|指导员)[：:]", line):
                teacher_section = True
                winner_section = False
            
            # 检查获奖者模式
            for pattern in winner_patterns:
                match = re.search(pattern, line)
                if match:
                    if match.group(1) and len(match.group(1).strip()) > 0:
                        # 从这一行提取名字
                        names = self.extract_names(match.group(1))
                        people["winner"].extend(names)
                    elif i + 1 < len(text_lines) and not any(keyword in text_lines[i+1] for keyword in ["指导教师", "导师", "辅导教师", "指导老师"]):
                        # 名字可能在下一行
                        names = self.extract_names(text_lines[i+1])
                        people["winner"].extend(names)
                    break
            
            # 如果当前在获奖者部分，且当前行可能是名字列表
            if winner_section and not teacher_section and len(line.strip()) > 0:
                if self.is_name_list(line):
                    names = self.extract_names(line)
                    people["winner"].extend(names)
            
            # 检查教师模式
            if not winner_section:  # 确保不在获奖者部分时才检查教师模式
                for pattern in teacher_patterns:
                    match = re.search(pattern, line)
                    if match:
                        if match.group(1) and len(match.group(1).strip()) > 0:
                            # 从这一行提取名字
                            names = self.extract_names(match.group(1))
                            people["teacher"].extend(names)
                        elif i + 1 < len(text_lines):
                            # 名字可能在下一行
                            names = self.extract_names(text_lines[i+1])
                            people["teacher"].extend(names)
                        break
            
            # 如果当前在教师部分，且当前行可能是名字列表
            if teacher_section and len(line.strip()) > 0:
                if self.is_name_list(line):
                    names = self.extract_names(line)
                    people["teacher"].extend(names)
        # 如果以上基于部分的提取失败，尝试基于位置的提取
        if not people["winner"] and not people["teacher"]:
            for line in text_lines:
                # 先查找负责人/获奖学生
                if "负责人" in line and "：" in line:
                    parts = line.split("：")
                    if len(parts) > 1 and parts[1].strip():
                        name = parts[1].strip()
                        if self.is_valid_name(name):
                            people["winner"].append(name)
                
                # 检查这一行是否可能是名字列表
                elif len(line.strip()) > 0 and len(line.strip()) < 20:
                    names = self.extract_names(line)
                    
                    if names:
                        # 判断是否是教师名字
                        if any(teacher_kw in text for teacher_kw in ["指导教师", "导师"]) and len(people["teacher"]) == 0:
                            people["teacher"].extend(names)
                        else:
                            people["winner"].extend(names)
                    
        # 从"获奖学生"部分提取名字
        for line in text_lines:
            if "获奖学生：" in line:
                student_part = line.split("获奖学生：")[1].strip()
                student_names = self.extract_names(student_part)
                people["winner"].extend(student_names)
                
        # 删除重复项和非名字项
        people["winner"] = [name for name in list(set(people["winner"])) if self.is_valid_name(name)]
        people["teacher"] = [name for name in list(set(people["teacher"])) if self.is_valid_name(name)]
        
        return people
        
    def is_name_list(self, text):
        """
        检查文本行是否可能是名字列表
        
        Args:
            text (str): 文本行
            
        Returns:
            bool: 如果可能是名字列表则为 True
        """
        # 排除常见的非名字文本
        non_name_keywords = [
            "负责人", "指导教师", "导师", "指导老师", "获奖学生", "队员", "参赛者", 
            "作者", "获奖人", "学生", "成员", "年", "月", "日", "二零", "学院"
        ]
        
        # 检查是否包含非名字关键词作为独立词语
        for keyword in non_name_keywords:
            if keyword == text.strip():
                return False
        
        # 中文名字通常是 2-4 个字符
        name_pattern = r"[\u4e00-\u9fa5]{2,4}"
        
        # 名字列表中的常见分隔符
        separator_pattern = r"[、，,；;]"
        
        # 检查文本是否包含由常见分隔符分隔的多个名字
        names = re.findall(f"{name_pattern}(?:{separator_pattern}|$)", text)
        
        # 确保找到的名字不是上面列表中的关键词
        filtered_names = [name for name in names if name not in non_name_keywords]
        
        return len(filtered_names) > 0 and len(filtered_names) * 4 >= len(text) * 0.5
        
    def extract_names(self, text):
        """
        从文本字符串中提取名字
        
        Args:
            text (str): 包含名字的文本
            
        Returns:
            list: 提取的名字列表
        """
        if not text or len(text.strip()) == 0:
            return []
            
        # 删除不是名字的常见前缀和后缀
        text = re.sub(r"(?:教授|老师|博士|硕士|先生|女士|同学)$", "", text)
        
        # 检查常见分隔符
        if any(sep in text for sep in ["、", "，", ",", "；", ";"]):
            # 按分隔符拆分
            parts = re.split(r"[、，,；;]", text)
            # 清理每个部分
            names = [part.strip() for part in parts if self.is_valid_name(part.strip())]
            return names
        else:
            # 检查文本是否是单个名字
            if self.is_valid_name(text.strip()):
                return [text.strip()]
            else:
                # 尝试使用模式匹配提取名字
                name_pattern = r"[\u4e00-\u9fa5]{2,4}"
                names = re.findall(name_pattern, text)
                return [name for name in names if self.is_valid_name(name)]
    
    def is_valid_name(self, text):
        """
        检查字符串是否可能是有效的中文名字
        
        Args:
            text (str): 要检查的文本
            
        Returns:
            bool: 如果可能是名字则为 True
        """
        # 中文名字通常是 2-4 个字符
        if not text or not text.strip():
            return False
            
        text = text.strip()
        
        # 检查长度
        if not (2 <= len(text) <= 4):
            return False
            
        # 检查是否只包含中文字符
        if not re.match(r"^[\u4e00-\u9fa5]+$", text):
            return False
            
        # 排除明显不是名字的词语
        non_name_keywords = [
            "负责人", "指导教师", "导师", "指导老师", "获奖学生", "队员", "参赛者", 
            "作者", "获奖人", "学生", "成员", "年", "月", "日", "二零", "学院",
            "证书", "奖状", "二零二四", "年出月", "年出月院"
        ]
        
        if text in non_name_keywords:
            return False
            
        # 排除只包含数字的中文字符（如"二零二四"）
        chinese_digits = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
        if all(char in chinese_digits for char in text):
            return False
            
        # 如果所有测试都通过但我们不确定，倾向于接受它
        return True
    
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
    
    def extract_information(self, full_text, text_lines):
        """
        从证书文本中提取结构化信息
        
        Args:
            full_text (str): 合并的 OCR 文本
            text_lines (list): 单独文本行列表
            
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
        
        # 提取每个组件
        info["competition_name"] = self.clean_text(self.extract_competition_name(full_text, text_lines))
        info["award_level"] = self.clean_text(self.extract_award_level(full_text, text_lines))
        info["project_name"] = self.clean_text(self.extract_project_name(full_text, text_lines))
        info["people"] = self.extract_people(full_text, text_lines)
        
        # 确保获奖者和教师之间没有重叠
        common_names = set(info["people"]["winner"]) & set(info["people"]["teacher"])
        if common_names:
            # 如果名字出现在两个列表中，只保留在获奖者中
            info["people"]["teacher"] = [name for name in info["people"]["teacher"] if name not in common_names]
        
        return info
    
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
    ocr = CertificateOCR()
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