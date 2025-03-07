import os
import cv2
import numpy as np
import json
import re
import jieba
import jieba.posseg as pseg
from paddleocr import PaddleOCR
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
from pathlib import Path

class AwardCertificateOCR:
    def __init__(self, use_gpu=False):
        """
        初始化奖状OCR识别与信息提取系统
        
        Args:
            use_gpu: 是否使用GPU加速
        """
        # 初始化OCR模型
        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=use_gpu)
        
        # 初始化BERT模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.bert_model = BertModel.from_pretrained("bert-base-chinese")
        
        # 加载jieba自定义词典
        self.load_custom_dict()
        
        # 定义奖状信息模板
        self.award_patterns = {
            "award_name": [r"([\u4e00-\u9fa5]{2,10}(大赛|比赛|竞赛|锦标赛|联赛))",
                          r"([\u4e00-\u9fa5]{2,15}奖)",
                          r"([\u4e00-\u9fa5]{2,20}(证书|奖状))"],
            "award_level": [r"(一等奖|二等奖|三等奖|特等奖|金奖|银奖|铜奖|优秀奖|优胜奖|一级|二级|三级)",
                           r"(第[一二三四五六七八九十]名)",
                           r"(冠军|亚军|季军)"],
            "winner_name": [r"(兹有|授予|获得者|颁发给|证明|同学|学生)([\u4e00-\u9fa5]{2,5})",
                          r"([\u4e00-\u9fa5]{2,5})(同学|老师|教授|博士)"],
            "organization": [r"([\u4e00-\u9fa5]{2,20}(委员会|协会|学会|部|局|中心|单位|组委会))",
                           r"([\u4e00-\u9fa5]{2,15}(大学|学院|学校))"],
            "date": [r"(\d{4}[年/-]\d{1,2}[月/-]\d{1,2}[日号]?)",
                    r"(\d{4}[年/-]\d{1,2}[月/-])",
                    r"(\d{4}[年/-])"]
        }
    
    def load_custom_dict(self):
        """加载自定义词典到jieba"""
        award_terms = [
            "一等奖", "二等奖", "三等奖", "特等奖", "金奖", "银奖", "铜奖", 
            "优秀奖", "全国大赛", "国际竞赛", "省级比赛", "市级比赛",
            "竞赛委员会", "组织委员会", "颁奖单位", "荣誉证书"
        ]
        
        for term in award_terms:
            jieba.add_word(term)
    
    def preprocess_image(self, image_path):
        """
        图像预处理
        
        Args:
            image_path: 图像路径
        
        Returns:
            处理后的图像
        """
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 获取图像尺寸并计算缩放比例
        h, w = img.shape[:2]
        max_dim = 1500
        scale = min(max_dim / w, max_dim / h)
        
        # 如果图像太大则缩放
        if scale < 1.0:
            new_width = int(w * scale)
            new_height = int(h * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 双边滤波去噪
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # 使用自适应阈值进行二值化
        binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # 使用形态学操作清除噪点
        kernel = np.ones((2, 2), np.uint8)
        morphology = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 校正图像倾斜
        corrected = self.correct_skew(morphology)
        
        return corrected, img
    
    def correct_skew(self, image):
        """
        校正图像倾斜
        
        Args:
            image: 输入图像
        
        Returns:
            校正后的图像
        """
        # 检测边缘
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # 霍夫变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is None or len(lines) == 0:
            return image
        
        # 计算角度
        angles = []
        for line in lines:
            for rho, theta in line:
                # 只考虑水平附近的线条
                if 0.7 < theta < 2.4:  # 大约 40° 到 140°
                    angle = theta - np.pi/2
                    angles.append(angle)
        
        if not angles:
            return image
        
        # 取中位数作为旋转角度
        median_angle = np.median(angles)
        angle_degrees = np.degrees(median_angle)
        
        # 如果角度太大，可能是误检测，不进行校正
        if abs(angle_degrees) > 20:
            return image
        
        # 旋转图像
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    def recognize_text(self, image):
        """
        使用OCR识别图像中的文字
        
        Args:
            image: 预处理后的图像
        
        Returns:
            识别出的文本列表
        """
        result = self.ocr.ocr(image, cls=True)
        
        # PaddleOCR的结果结构可能因版本而异
        # 适应不同版本的输出格式
        if result is None:
            return []
        
        # 统一返回结果格式
        texts = []
        for line in result:
            if isinstance(line, list):
                for item in line:
                    if len(item) >= 2:
                        text = item[1][0]  # 获取文本内容
                        confidence = item[1][1]  # 获取置信度
                        if confidence > 0.7:  # 只保留置信度高的结果
                            texts.append(text)
            elif isinstance(line, tuple) and len(line) >= 2:
                text = line[1][0]
                confidence = line[1][1]
                if confidence > 0.7:
                    texts.append(text)
        
        return texts
    
    def extract_entities_with_bert(self, texts):
        """
        使用BERT模型进行命名实体识别
        
        Args:
            texts: OCR识别出的文本列表
        
        Returns:
            识别出的实体
        """
        # 将文本转换为一个字符串
        full_text = "，".join(texts)
        
        # BERT输入准备
        inputs = self.tokenizer(full_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # 获取BERT输出
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            embeddings = outputs.last_hidden_state
        
        # 使用jieba进行分词和词性标注
        words = pseg.cut(full_text)
        
        # 基于词性标注进行实体提取
        entities = {
            "person": [],
            "organization": [],
            "date": [],
            "award": []
        }
        
        for word, flag in words:
            if flag == 'nr' and len(word) >= 2:  # 人名
                entities["person"].append(word)
            elif flag == 'nt' and len(word) >= 2:  # 机构名
                entities["organization"].append(word)
            elif flag == 't':  # 时间
                entities["date"].append(word)
            elif 'n' in flag and len(word) >= 2:  # 可能是奖项
                if "奖" in word or "赛" in word or "证书" in word:
                    entities["award"].append(word)
        
        return entities
    
    def extract_info_with_rules(self, texts):
        """
        使用规则匹配提取奖状信息
        
        Args:
            texts: OCR识别出的文本列表
        
        Returns:
            匹配出的信息
        """
        # 将文本转换为一个字符串以便于规则匹配
        full_text = "，".join(texts)
        
        extracted_info = {
            "award_name": None,
            "award_level": None,
            "winner_name": None,
            "organization": None,
            "date": None
        }
        
        # 使用正则表达式匹配各类信息
        for info_type, patterns in self.award_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, full_text)
                if matches:
                    if info_type == "winner_name" and len(matches[0]) > 1:
                        # 对于获奖者姓名，需要特殊处理
                        extracted_info[info_type] = matches[0][1]
                    else:
                        match_str = matches[0]
                        if isinstance(match_str, tuple):
                            match_str = match_str[0]
                        extracted_info[info_type] = match_str
                    break
        
        return extracted_info
    
    def merge_extraction_results(self, bert_entities, rule_based_info):
        """
        合并BERT和规则提取的结果
        
        Args:
            bert_entities: BERT提取的实体
            rule_based_info: 规则提取的信息
        
        Returns:
            合并后的信息
        """
        merged_info = {
            "award_name": rule_based_info["award_name"],
            "award_level": rule_based_info["award_level"],
            "winner_name": rule_based_info["winner_name"],
            "organization": rule_based_info["organization"],
            "date": rule_based_info["date"]
        }
        
        # 如果规则没提取到，尝试使用BERT结果
        if not merged_info["winner_name"] and bert_entities["person"]:
            merged_info["winner_name"] = bert_entities["person"][0]
        
        if not merged_info["organization"] and bert_entities["organization"]:
            merged_info["organization"] = bert_entities["organization"][0]
        
        if not merged_info["date"] and bert_entities["date"]:
            merged_info["date"] = bert_entities["date"][0]
        
        if not merged_info["award_name"] and bert_entities["award"]:
            merged_info["award_name"] = bert_entities["award"][0]
        
        return merged_info
    
    def process_certificate(self, image_path):
        """
        处理单个奖状
        
        Args:
            image_path: 奖状图像路径
        
        Returns:
            提取的信息JSON字符串
        """
        try:
            # 预处理图像
            processed_img, original_img = self.preprocess_image(image_path)
            
            # 识别文字
            texts = self.recognize_text(processed_img)
            
            # 如果文字识别效果不好，尝试使用原图
            if len(texts) < 5:
                texts = self.recognize_text(original_img)
            
            # 使用BERT进行实体识别
            bert_entities = self.extract_entities_with_bert(texts)
            
            # 使用规则提取信息
            rule_based_info = self.extract_info_with_rules(texts)
            
            # 合并两种方法的结果
            merged_info = self.merge_extraction_results(bert_entities, rule_based_info)
            
            # 添加原始OCR识别文本以便人工确认
            merged_info["raw_text"] = texts
            
            # 添加置信度评分
            merged_info["confidence"] = self.calculate_confidence(merged_info)
            
            return json.dumps(merged_info, ensure_ascii=False)
        
        except Exception as e:
            error_info = {
                "error": str(e),
                "status": "failed"
            }
            return json.dumps(error_info, ensure_ascii=False)
    
    def calculate_confidence(self, info):
        """
        计算提取信息的置信度
        
        Args:
            info: 提取的信息
        
        Returns:
            0-1之间的置信度分数
        """
        # 计算核心字段的完整性
        key_fields = ["award_name", "winner_name", "organization"]
        filled_fields = sum(1 for field in key_fields if info[field])
        
        # 基本置信度基于字段完整性
        base_confidence = filled_fields / len(key_fields)
        
        # 根据字段长度调整置信度
        length_adjustment = 0
        if info["winner_name"] and 2 <= len(info["winner_name"]) <= 5:
            length_adjustment += 0.1
        if info["award_name"] and len(info["award_name"]) >= 4:
            length_adjustment += 0.1
        
        final_confidence = min(base_confidence + length_adjustment, 1.0)
        
        return round(final_confidence, 2)
    
    def batch_process(self, folder_path, output_dir=None):
        """
        批量处理奖状图像
        
        Args:
            folder_path: 包含奖状图像的文件夹路径
            output_dir: 输出JSON结果的目录路径，如果为None则不保存单独的JSON文件
        
        Returns:
            包含所有提取信息的JSON字符串
        """
        results = []
        
        # 获取文件夹中所有图像文件
        image_files = [f for f in os.listdir(folder_path) 
                     if os.path.isfile(os.path.join(folder_path, f)) and 
                     f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
        
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            result_json = self.process_certificate(image_path)
            result = json.loads(result_json)
            result["filename"] = image_file
            results.append(result)
            
            # 如果指定了输出目录，保存单独的JSON文件
            if output_dir:
                json_filename = f"{os.path.splitext(image_file)[0]}.json"
                output_path = os.path.join(output_dir, json_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False, indent=2))
        
        return json.dumps(results, ensure_ascii=False, indent=2)


# 确保目录结构存在
def ensure_directories():
    """创建数据目录结构"""
    # 创建data目录及其子目录
    data_dir = Path("data")
    img_dir = data_dir / "img"
    result_dir = data_dir / "result"
    
    # 确保所有目录都存在
    for directory in [data_dir, img_dir, result_dir]:
        directory.mkdir(exist_ok=True, parents=True)
    
    return {
        "data_dir": data_dir,
        "img_dir": img_dir,
        "result_dir": result_dir
    }

# 保存JSON结果到文件
def save_json_result(json_data, filename, result_dir):
    """将JSON结果保存到指定文件"""
    output_path = result_dir / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json_data)
    print(f"结果已保存到: {output_path}")

# 示例使用
def main():
    # 创建必要的目录
    dirs = ensure_directories()
    
    # 创建OCR系统实例
    ocr_system = AwardCertificateOCR(use_gpu=False)
    
    # 批量处理奖状
    img_dir = dirs["img_dir"]
    result_dir = dirs["result_dir"]
    
    # 检查是否有图片
    image_files = [f for f in os.listdir(img_dir) 
                 if os.path.isfile(os.path.join(img_dir, f)) and 
                 f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
    
    if not image_files:
        print(f"在 {img_dir} 目录中没有找到图像文件。请将奖状图片放入该目录。")
        return
    
    print(f"发现 {len(image_files)} 个图像文件, 开始处理...")
    
    # 处理每个图像并保存结果
    all_results = []
    
    for image_file in image_files:
        print(f"处理图像: {image_file}")
        image_path = os.path.join(img_dir, image_file)
        
        # 处理单个奖状
        result_json = ocr_system.process_certificate(image_path)
        result = json.loads(result_json)
        result["filename"] = image_file
        all_results.append(result)
        
        # 保存单个结果
        json_filename = f"{os.path.splitext(image_file)[0]}.json"
        save_json_result(json.dumps(result, ensure_ascii=False, indent=2), 
                        json_filename, result_dir)
    
    # 保存所有结果到一个汇总文件
    save_json_result(json.dumps(all_results, ensure_ascii=False, indent=2), 
                    "all_results.json", result_dir)
    
    print("所有处理完成!")


if __name__ == "__main__":
    main()