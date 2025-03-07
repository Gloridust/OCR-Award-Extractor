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
        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=use_gpu, 
                             rec_char_dict_path=None,  # 使用默认字典
                             rec_batch_num=6,  # 批处理数量
                             rec_model_dir=None,  # 使用默认模型
                             det_model_dir=None,  # 使用默认检测模型
                             cls_model_dir=None,  # 使用默认方向分类模型
                             enable_mkldnn=True,  # 启用MKL-DNN加速
                             det_db_thresh=0.3,  # 降低检测阈值，提高召回率
                             det_db_box_thresh=0.3,  # 降低框阈值
                             max_batch_size=24,  # 最大批处理大小
                             use_dilation=True,  # 使用膨胀操作
                             det_db_unclip_ratio=1.6)  # 提高unclip比例，更容易检测小文字
        
        # 初始化BERT模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.bert_model = BertModel.from_pretrained("bert-base-chinese")
        
        # 加载jieba自定义词典
        self.load_custom_dict()
        
        # 定义奖状信息模板
        self.award_patterns = {
            "award_name": [r"([\u4e00-\u9fa5]{2,10}(大赛|比赛|竞赛|锦标赛|联赛))",
                          r"([\u4e00-\u9fa5]{2,15}奖)",
                          r"([\u4e00-\u9fa5]{2,20}(证书|奖状))",
                          r"第[\u4e00-\u9fa5零一二三四五六七八九十百届]+[\u4e00-\u9fa5]+"],
            "award_level": [r"([国省市校区]?级[特一二三]等奖)",
                           r"([国省市校区]?级[金银铜优]奖)",
                           r"(特等奖|一等奖|二等奖|三等奖|金奖|银奖|铜奖|优秀奖|优胜奖|一级|二级|三级)",
                           r"(第[一二三四五六七八九十]名)",
                           r"(冠军|亚军|季军)"],
            "winner_name": [r"(兹有|授予|获得者|颁发给|证明|同学|学生)([\u4e00-\u9fa5]{2,5})",
                          r"([\u4e00-\u9fa5]{2,5})(同学|老师|教授|博士)",
                          r"负责人[：:]([\u4e00-\u9fa5]{2,5})",
                          r"获奖学生[：:]([\u4e00-\u9fa5、，,]+)"],
            "project_name": [r"获奖项目[：:](.*?)[；;。\n]",
                            r"项目名称[：:](.*?)[；;。\n]",
                            r"作品名称[：:](.*?)[；;。\n]"],
            "teachers": [r"指导教师[：:]([\u4e00-\u9fa5、，,]+)",
                        r"([\u4e00-\u9fa5]{2,5})(教师|老师|导师|指导|教授)"],
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
            "优秀奖", "全国大赛", "国际竞赛", "省级比赛", "市级比赛", "校级比赛",
            "竞赛委员会", "组织委员会", "颁奖单位", "荣誉证书", "指导教师",
            "获奖项目", "负责人", "获奖学生", "互联网+", "国际创新大赛",
            "国家级", "省级", "市级", "校级", "区级"
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
            识别出的文本列表和位置信息
        """
        result = self.ocr.ocr(image, cls=True)
        
        if result is None or len(result) == 0:
            return [], []
        
        # 统一返回结果格式
        texts = []
        boxes = []
        
        # 处理PaddleOCR不同版本的输出格式
        if isinstance(result, list) and len(result) > 0:
            # 对于新版本的输出
            if isinstance(result[0], list):
                for line in result:
                    for item in line:
                        if len(item) >= 2:
                            text = item[1][0]  # 获取文本内容
                            confidence = item[1][1]  # 获取置信度
                            box = item[0]  # 获取文本框坐标
                            if confidence > 0.6:  # 只保留置信度高的结果
                                texts.append(text)
                                boxes.append(box)
            # 对于旧版本的输出
            elif isinstance(result[0], dict) and 'text' in result[0]:
                for item in result:
                    text = item['text'][0]
                    confidence = item['text'][1]
                    box = item['box']
                    if confidence > 0.6:
                        texts.append(text)
                        boxes.append(box)
            # 对于其他格式的输出
            else:
                for line in result:
                    if isinstance(line, tuple) and len(line) >= 2:
                        text = line[1][0]
                        confidence = line[1][1]
                        box = line[0]
                        if confidence > 0.6:
                            texts.append(text)
                            boxes.append(box)
        
        # 根据文本框的位置对文本进行排序（从上到下，从左到右）
        if boxes and texts:
            # 计算每个文本框的中心点y坐标
            centers_y = [sum(box[i][1] for i in range(4)) / 4 for box in boxes]
            
            # 首先按照y坐标排序（粗略分行）
            sorted_indices = sorted(range(len(centers_y)), key=lambda i: centers_y[i])
            
            # 对排序后的文本和框进行重排
            texts = [texts[i] for i in sorted_indices]
            boxes = [boxes[i] for i in sorted_indices]
            
            # 尝试合并同一行的相邻文本
            merged_texts = []
            merged_boxes = []
            i = 0
            while i < len(texts):
                current_text = texts[i]
                current_box = boxes[i]
                
                # 检查下一个文本是否在同一行（y坐标接近）
                if i + 1 < len(texts) and abs(centers_y[sorted_indices[i]] - centers_y[sorted_indices[i+1]]) < 15:
                    # 获取下一个文本框的左侧x坐标
                    next_box = boxes[i+1]
                    next_left_x = min(point[0] for point in next_box)
                    current_right_x = max(point[0] for point in current_box)
                    
                    # 如果两个文本框水平距离较近，且没有明显的标点符号结束，合并它们
                    if next_left_x - current_right_x < 50 and not current_text.endswith(("。", "，", "；", "！", "？", "：", "、")):
                        current_text += texts[i+1]
                        # 合并框坐标（简单合并为外接矩形）
                        all_points = current_box + next_box
                        min_x = min(point[0] for point in all_points)
                        min_y = min(point[1] for point in all_points)
                        max_x = max(point[0] for point in all_points)
                        max_y = max(point[1] for point in all_points)
                        current_box = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
                        i += 1  # 跳过下一个文本
                
                merged_texts.append(current_text)
                merged_boxes.append(current_box)
                i += 1
            
            return merged_texts, merged_boxes
        
        return texts, boxes
    
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
    
    def extract_info_with_rules(self, texts, boxes=None):
        """
        使用规则匹配提取奖状信息
        
        Args:
            texts: OCR识别出的文本列表
            boxes: 文本框位置信息列表（可选）
        
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
            "date": None,
            "project_name": None,
            "team_members": None,
            "teachers": None
        }
        
        # 使用正则表达式匹配各类信息
        for info_type, patterns in self.award_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, full_text)
                if matches:
                    if info_type == "winner_name":
                        if "负责人" in pattern:
                            # 处理负责人匹配
                            responsible_person = matches[0]
                            if isinstance(responsible_person, tuple):
                                responsible_person = responsible_person[0]
                            extracted_info[info_type] = responsible_person
                        elif "获奖学生" in pattern:
                            # 处理团队成员匹配
                            team_members = matches[0]
                            if isinstance(team_members, tuple):
                                team_members = team_members[0]
                            extracted_info["team_members"] = team_members
                        else:
                            # 处理其他获奖者匹配
                            if len(matches[0]) > 1 and isinstance(matches[0], tuple):
                                extracted_info[info_type] = matches[0][1]
                            else:
                                match_str = matches[0]
                                if isinstance(match_str, tuple):
                                    match_str = match_str[0]
                                extracted_info[info_type] = match_str
                    elif info_type == "project_name":
                        # 处理项目名称匹配
                        project_name = matches[0]
                        if isinstance(project_name, tuple):
                            project_name = project_name[0]
                        extracted_info[info_type] = project_name.strip()
                    elif info_type == "teachers":
                        # 处理指导教师匹配
                        teachers = matches[0]
                        if isinstance(teachers, tuple):
                            teachers = teachers[0]
                        extracted_info[info_type] = teachers
                    else:
                        # 处理其他类型的匹配
                        match_str = matches[0]
                        if isinstance(match_str, tuple):
                            match_str = match_str[0]
                        extracted_info[info_type] = match_str
                    
                    # 如果已经匹配到了，就跳过后续模式
                    if extracted_info[info_type]:
                        break
        
        # 使用文本框布局信息进行特殊处理
        if boxes and len(boxes) == len(texts):
            self.extract_info_from_layout(texts, boxes, extracted_info)
        
        return extracted_info
    
    def extract_info_from_layout(self, texts, boxes, extracted_info):
        """
        基于文本框布局信息提取额外信息
        
        Args:
            texts: OCR识别出的文本列表
            boxes: 对应的文本框位置信息
            extracted_info: 已提取的信息字典，将被更新
        """
        # 检查是否有疑似不完整的人名
        for i, text in enumerate(texts):
            # 检查指导教师
            if "指导教师" in text and "：" in text:
                teacher_part = text.split("：")[1].strip()
                if 1 <= len(teacher_part) <= 2:  # 名字可能不完整
                    # 查找水平相邻的文本框
                    current_box = boxes[i]
                    current_center_y = sum(point[1] for point in current_box) / 4
                    
                    for j, other_box in enumerate(boxes):
                        if i != j:
                            other_center_y = sum(point[1] for point in other_box) / 4
                            # 如果两个框在同一水平线上（Y坐标接近）
                            if abs(current_center_y - other_center_y) < 20:
                                # 获取当前框的右边界和下一个框的左边界
                                current_right = max(point[0] for point in current_box)
                                other_left = min(point[0] for point in other_box)
                                
                                # 如果下一个框就在当前框右侧不远处
                                if 0 < other_left - current_right < 100:
                                    # 合并文本
                                    complete_teacher = teacher_part + texts[j]
                                    extracted_info["teachers"] = complete_teacher
                                    break
            
            # 检查项目名称
            if "获奖项目：" in text or "项目名称：" in text or "作品名称：" in text:
                project_part = text.split("：")[1].strip()
                # 如果项目名称看起来不完整
                if project_part and not project_part.endswith(("器", "法", "统", "台", "件")):
                    # 寻找下一行可能的剩余部分
                    for j in range(len(texts)):
                        if i != j:
                            # 简单启发式判断：如果另一个文本不包含明显的标签
                            other_text = texts[j]
                            if "：" not in other_text and len(other_text) > 1:
                                # 合并项目名称
                                complete_project = project_part + other_text
                                extracted_info["project_name"] = complete_project
                                break
    
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
            "date": rule_based_info["date"],
            "project_name": rule_based_info["project_name"],
            "team_members": rule_based_info["team_members"],
            "teachers": rule_based_info["teachers"]
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
        
        # 处理团队成员信息
        if merged_info["team_members"]:
            # 尝试从team_members字符串中提取所有人名
            team_members_str = merged_info["team_members"]
            # 分割字符串，处理各种可能的分隔符
            members = re.split(r'[、，,；;]', team_members_str)
            members = [m.strip() for m in members if m.strip()]
            
            # 如果负责人信息不在team_members中，保留两个不同的字段
            if merged_info["winner_name"] and merged_info["winner_name"] not in team_members_str:
                # 保持字段分开
                pass
            elif members:
                # 如果没有单独的负责人信息，但有团队成员，将第一个成员设为负责人
                if not merged_info["winner_name"]:
                    merged_info["winner_name"] = members[0]
        
        # 处理指导教师信息
        if merged_info["teachers"]:
            # 清理指导教师字符串
            teachers_str = merged_info["teachers"]
            # 处理可能的分隔符
            teachers = re.split(r'[、，,；;]', teachers_str)
            teachers = [t.strip() for t in teachers if t.strip()]
            
            # 检查是否有名字看起来很短（可能被截断）
            for i, teacher in enumerate(teachers):
                if len(teacher) <= 2 and i < len(teachers) - 1:
                    # 尝试合并相邻的两个短名字
                    next_teacher = teachers[i+1]
                    if len(next_teacher) <= 2:
                        teachers[i] = teacher + next_teacher
                        teachers.pop(i+1)
            
            merged_info["teachers"] = "、".join(teachers)
        
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
            
            # 首先尝试使用处理后的图像识别文字
            texts, boxes = self.recognize_text(processed_img)
            
            # 如果文字识别效果不好，尝试使用原图
            if len(texts) < 5:
                texts, boxes = self.recognize_text(original_img)
                
                # 如果还是效果不好，尝试直接使用灰度图像
                if len(texts) < 5:
                    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                    texts, boxes = self.recognize_text(gray)
            
            # 使用BERT进行实体识别
            bert_entities = self.extract_entities_with_bert(texts)
            
            # 使用规则提取信息，传入文本框位置信息
            rule_based_info = self.extract_info_with_rules(texts, boxes)
            
            # 合并两种方法的结果
            merged_info = self.merge_extraction_results(bert_entities, rule_based_info)
            
            # 清理和标准化信息
            merged_info = self.clean_results(merged_info, texts)
            
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
    
    def clean_results(self, info, texts):
        """
        清理和标准化提取的信息
        
        Args:
            info: 提取的信息字典
            texts: 原始OCR文本
        
        Returns:
            清理后的信息字典
        """
        # 创建返回结果的副本
        cleaned = info.copy()
        
        # 清理项目名称
        if cleaned.get("project_name"):
            # 删除多余的冒号
            project = cleaned["project_name"]
            project = re.sub(r'：+', '：', project)
            project = re.sub(r':+', '：', project)
            
            # 去除项目名称中可能的干扰词
            project = project.replace("获奖项目：", "").replace("项目名称：", "").replace("作品名称：", "")
            cleaned["project_name"] = project.strip()
        
        # 处理获奖者和团队成员
        if cleaned.get("team_members") and cleaned.get("winner_name"):
            # 如果已经有负责人和团队成员，构建完整的获奖者信息
            all_winners = f"{cleaned['winner_name']}(负责人), {cleaned['team_members']}"
            cleaned["full_team"] = all_winners
        elif cleaned.get("team_members"):
            # 如果只有团队成员
            cleaned["full_team"] = cleaned["team_members"]
        elif cleaned.get("winner_name"):
            # 如果只有获奖者
            cleaned["full_team"] = cleaned["winner_name"]
        
        # 重新检查完整的奖项级别
        if cleaned.get("award_level"):
            level = cleaned["award_level"]
            
            # 在原始文本中查找更完整的奖项级别
            for text in texts:
                if level in text and len(text) > len(level):
                    # 尝试提取更完整的奖项级别
                    full_level_match = re.search(r'([国省市校区]?级[特一二三]等奖|[国省市校区]?级[金银铜优]奖)', text)
                    if full_level_match:
                        cleaned["award_level"] = full_level_match.group(0)
                        break
        
        # 清理指导教师信息
        if cleaned.get("teachers"):
            # 删除可能的标签
            teachers = cleaned["teachers"]
            teachers = teachers.replace("指导教师：", "").replace("指导教师:", "")
            
            # 检查是否有短名字（可能截断）
            if 1 <= len(teachers) <= 2:
                # 在原始文本中查找可能匹配的更完整的教师名字
                for text in texts:
                    if teachers in text and "老师" in text or "教师" in text:
                        # 尝试提取出完整的名字
                        name_match = re.search(f'{teachers}[\u4e00-\u9fa5]{{1,2}}', text)
                        if name_match:
                            cleaned["teachers"] = name_match.group(0)
                            break
            
            cleaned["teachers"] = teachers.strip()
        
        return cleaned
    
    def calculate_confidence(self, info):
        """
        计算提取信息的置信度
        
        Args:
            info: 提取的信息
        
        Returns:
            0-1之间的置信度分数
        """
        # 计算核心字段的完整性
        key_fields = ["award_name", "award_level", "winner_name"]
        optional_fields = ["organization", "date", "project_name", "team_members", "teachers"]
        
        # 基本置信度基于核心字段的完整性
        filled_core_fields = sum(1 for field in key_fields if info[field])
        filled_optional_fields = sum(1 for field in optional_fields if info[field])
        
        # 基本置信度
        base_confidence = filled_core_fields / len(key_fields)
        
        # 附加置信度来自可选字段
        optional_bonus = min(0.2, filled_optional_fields * 0.05)
        
        # 根据字段内容质量调整置信度
        quality_adjustment = 0
        
        # 检查获奖者信息是否合理
        if info["winner_name"] and 2 <= len(info["winner_name"]) <= 5:
            quality_adjustment += 0.1
        
        # 检查奖项名称是否合理
        if info["award_name"] and len(info["award_name"]) >= 4:
            quality_adjustment += 0.1
        
        # 检查是否有团队成员信息
        if info.get("team_members") and len(info["team_members"]) >= 5:
            quality_adjustment += 0.05
        
        # 检查是否有项目名称
        if info.get("project_name") and len(info["project_name"]) >= 5:
            quality_adjustment += 0.05
        
        # 计算最终置信度，限制在0-1范围内
        final_confidence = min(base_confidence + optional_bonus + quality_adjustment, 1.0)
        
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