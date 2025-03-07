import os
import cv2
import numpy as np
import json
import re
from PIL import Image
import torch
from paddleocr import PaddleOCR
from transformers import AutoTokenizer, AutoModelForTokenClassification
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CertificateOCR:
    def __init__(self, use_gpu=False, img_dir="data/img", result_dir="data/result"):
        """
        初始化证书OCR处理器
        
        参数:
            use_gpu: 是否使用GPU加速
            img_dir: 输入图像目录
            result_dir: 结果输出目录
        """
        logger.info("初始化CertificateOCR系统...")
        
        # 设置目录结构
        self.img_dir = img_dir
        self.result_dir = result_dir
        
        # 创建目录（如果不存在）
        self._ensure_directories()
        
        # 初始化PaddleOCR模型 - 专为中文优化
        self.ocr = PaddleOCR(
            use_angle_cls=True,     # 使用方向分类器
            lang="ch",              # 中文模型
            use_gpu=use_gpu,        # GPU加速
            det_model_dir=None,     # 使用默认检测模型
            rec_model_dir=None,     # 使用默认识别模型
            cls_model_dir=None,     # 使用默认方向分类模型
            det_limit_side_len=2400,# 检测模型最大尺寸
            det_db_thresh=0.3,      # 文本检测阈值
            det_db_box_thresh=0.5,  # 文本检测框阈值
            det_db_unclip_ratio=1.6,# 文本检测框扩张比例
            rec_batch_num=6,        # 识别批次大小
            max_text_length=25,     # 最大文本长度
            rec_char_dict_path=None,# 使用默认字典
            drop_score=0.5          # 低于此置信度的结果将被丢弃
        )
        
        # 初始化BERT模型用于信息提取
        # 使用中文BERT变体，更适合中文实体识别
        model_name = "hfl/chinese-roberta-wwm-ext"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # 注意：在实际应用中，应该使用微调过的NER模型
            # 这里为了演示，使用预训练模型
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            
            # GPU加速
            self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
            self.model.to(self.device)
            
            logger.info(f"NLP模型已加载，使用设备: {self.device}")
        except Exception as e:
            logger.warning(f"NLP模型加载失败: {str(e)}，将仅使用规则提取信息")
            self.tokenizer = None
            self.model = None
            self.device = None
        
        logger.info("CertificateOCR系统初始化完成")
    
    def preprocess_image(self, image_path):
        """
        预处理证书图像
        
        参数:
            image_path: 图像文件路径
            
        返回:
            处理后的图像
        """
        logger.info(f"开始预处理图像: {image_path}")
        
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise Exception(f"无法读取图像: {image_path}")
        
        # 保存原始图像尺寸
        original_height, original_width = img.shape[:2]
        logger.info(f"原始图像尺寸: {original_width}x{original_height}")
        
        # 调整大小但保持比例
        max_size = 2400
        if max(original_height, original_width) > max_size:
            scale = max_size / max(original_height, original_width)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            logger.info(f"图像已调整大小，新尺寸: {img.shape[1]}x{img.shape[0]}")
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 增强对比度 - 使用自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 降噪 - 双边滤波保留边缘的同时去除噪声
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 尝试进行文档透视校正
        # 1. 边缘检测
        edges = cv2.Canny(denoised, 50, 150, apertureSize=3)
        
        # 2. 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 3. 如果找到了足够大的轮廓，尝试进行透视变换
        corrected_img = None
        if contours:
            # 找到最大的轮廓（假设这是文档的边缘）
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 如果轮廓足够大（占图像面积至少40%）
            if cv2.contourArea(largest_contour) > 0.4 * img.shape[0] * img.shape[1]:
                # 近似轮廓为多边形
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # 如果轮廓近似为四边形，则尝试透视变换
                if len(approx) == 4:
                    # 进行透视变换
                    src_pts = approx.reshape(4, 2).astype(np.float32)
                    
                    # 按照左上、右上、右下、左下的顺序排列点
                    # 计算每个点到原点的距离
                    s = src_pts.sum(axis=1)
                    # 左上角点 (最小和)
                    tl = src_pts[np.argmin(s)]
                    # 右下角点 (最大和)
                    br = src_pts[np.argmax(s)]
                    
                    # 计算差值 (右上点和左下点)
                    diff = np.diff(src_pts, axis=1)
                    # 右上角点 (最小差)
                    tr = src_pts[np.argmin(diff)]
                    # 左下角点 (最大差)
                    bl = src_pts[np.argmax(diff)]
                    
                    src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
                    
                    # 目标图像大小
                    width, height = original_width, original_height
                    dst_pts = np.array([
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1]
                    ], dtype=np.float32)
                    
                    # 计算变换矩阵并应用
                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    warped = cv2.warpPerspective(img, M, (width, height))
                    
                    logger.info("已应用透视校正")
                    corrected_img = warped
        
        # 如果没有进行透视变换或变换失败，使用原始图像
        processed_img = corrected_img if corrected_img is not None else img
        
        # 最终的图像增强
        processed_gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        final_enhanced = clahe.apply(processed_gray)
        
        # 锐化处理
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(final_enhanced, -1, kernel)
        
        # 转回彩色图像用于OCR
        color_img = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        
        logger.info("图像预处理完成")
        return color_img
    
    def recognize_text(self, image):
        """
        使用PaddleOCR识别图像中的文本
        
        参数:
            image: 预处理后的图像
            
        返回:
            识别的文本列表，每项包含文本内容、置信度和位置信息
        """
        logger.info("开始OCR文本识别")
        
        # 调用PaddleOCR进行识别
        result = self.ocr.ocr(image, cls=True)
        
        # 提取识别结果
        texts = []
        if not result or len(result) == 0:
            logger.warning("OCR未识别到任何文本")
            return texts
        
        # 处理OCR结果
        # 检查是否是2.0版本的结果格式
        if isinstance(result[0], list) and len(result[0]) > 0 and isinstance(result[0][0], list):
            # PaddleOCR 2.0版本
            for line in result:
                for item in line:
                    box, (text, prob) = item
                    # 确保box有4个点
                    if len(box) != 4:
                        continue
                    
                    # 计算边界框
                    x1 = min(point[0] for point in box)
                    y1 = min(point[1] for point in box)
                    x2 = max(point[0] for point in box)
                    y2 = max(point[1] for point in box)
                    
                    texts.append({
                        'text': text,
                        'confidence': prob,
                        'position': {
                            'x1': int(x1),
                            'y1': int(y1),
                            'x2': int(x2),
                            'y2': int(y2),
                            'points': [[int(p[0]), int(p[1])] for p in box]
                        }
                    })
        else:
            # PaddleOCR 1.0版本或其他格式
            for item in result:
                if isinstance(item, list) and len(item) == 2:
                    box, (text, prob) = item
                    x1 = min(point[0] for point in box)
                    y1 = min(point[1] for point in box)
                    x2 = max(point[0] for point in box)
                    y2 = max(point[1] for point in box)
                    
                    texts.append({
                        'text': text,
                        'confidence': prob,
                        'position': {
                            'x1': int(x1),
                            'y1': int(y1),
                            'x2': int(x2),
                            'y2': int(y2),
                            'points': [[int(p[0]), int(p[1])] for p in box]
                        }
                    })
        
        # 过滤掉空文本和低置信度的结果
        texts = [item for item in texts if item['text'].strip() and item['confidence'] > 0.5]
        
        # 按照垂直位置排序，模拟从上到下的阅读顺序
        texts.sort(key=lambda x: x['position']['y1'])
        
        logger.info(f"OCR识别完成，共识别{len(texts)}个文本块")
        return texts
    
    def extract_certificate_info(self, texts):
        """
        从OCR识别的文本中提取证书信息
        结合规则和NLP模型
        
        参数:
            texts: OCR识别的文本列表
            
        返回:
            提取的证书信息字典
        """
        logger.info("开始提取证书信息")
        
        # 1. 合并所有文本并排序（按垂直位置）
        sorted_texts = sorted(texts, key=lambda x: (x['position']['y1'], x['position']['x1']))
        full_text = " ".join([item['text'] for item in sorted_texts])
        
        logger.info(f"完整文本: {full_text[:100]}..." if len(full_text) > 100 else full_text)
        
        # 2. 使用规则进行初步提取
        certificate_info = self._extract_with_rules(full_text, sorted_texts)
        
        # 3. 使用NLP模型进行实体识别和分类增强
        if self.tokenizer is not None and self.model is not None:
            enhanced_info = self._enhance_with_nlp(sorted_texts, certificate_info)
        else:
            enhanced_info = certificate_info
        
        # 4. 清理结果中的噪声并进行格式化
        finalized_info = self._clean_and_format(enhanced_info)
        
        logger.info("证书信息提取完成")
        return finalized_info
    
    def _extract_with_rules(self, text, sorted_texts):
        """
        使用规则提取基本信息
        
        参数:
            text: 完整文本
            sorted_texts: 排序后的文本块列表
            
        返回:
            初步提取的证书信息
        """
        logger.info("使用规则提取信息")
        
        # 初始化结果字典
        certificate_info = {
            "certificate_type": "",
            "competition_name": "",
            "award_level": "",
            "people": {
                "winner": [],       # 获奖者
                "teammate": [],     # 团队成员
                "teacher": []       # 指导教师
            },
            "award_date": "",
            "issuing_organization": "",
            "certificate_number": ""
        }
        
        # 1. 证书类型
        cert_type_patterns = [
            r'(.*?)(证书|奖状|荣誉证书)',
            r'(获奖证书)',
            r'(荣誉证书)',
            r'(奖\s*状)'
        ]
        
        for pattern in cert_type_patterns:
            matches = re.findall(pattern, text)
            if matches:
                if isinstance(matches[0], tuple):
                    certificate_info["certificate_type"] = ''.join(matches[0]).strip()
                else:
                    certificate_info["certificate_type"] = matches[0].strip()
                logger.info(f"提取证书类型: {certificate_info['certificate_type']}")
                break
        
        # 2. 竞赛名称 (通常是标题或第一行大字)
        # 首先尝试从前几个文本块中获取
        top_text_blocks = sorted_texts[:3]
        for block in top_text_blocks:
            if len(block['text']) >= 6 and any(keyword in block['text'] for keyword in ["大赛", "竞赛", "比赛"]):
                certificate_info["competition_name"] = block['text'].strip()
                logger.info(f"从顶部文本块提取比赛名称: {certificate_info['competition_name']}")
                break
        
        # 如果上面的方法未提取到，则使用模式匹配
        if not certificate_info["competition_name"]:
            comp_patterns = [
                r'(.*?)(比赛|竞赛|大赛)(?!\s*获)',
                r'(第\s*[一二三四五六七八九十]+\s*届.*?)(比赛|竞赛|大赛)',
                r'([\u4e00-\u9fa5]{5,20}?(?:竞赛|大赛))'
            ]
            
            for pattern in comp_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    if isinstance(matches[0], tuple):
                        certificate_info["competition_name"] = ''.join(matches[0]).strip()
                    else:
                        certificate_info["competition_name"] = matches[0].strip()
                    logger.info(f"使用模式匹配提取比赛名称: {certificate_info['competition_name']}")
                    break
        
        # 3. 奖项级别
        award_patterns = [
            r'(一等奖|二等奖|三等奖|特等奖)',
            r'(金[奖|牌]|银[奖|牌]|铜[奖|牌])',
            r'(冠军|亚军|季军)',
            r'(特等奖|优秀奖|优胜奖|入围奖)',
            r'(第\s*[一二三四五六七八九十]\s*名)',
            r'(全国|省级|市级)\s*(一等奖|二等奖|三等奖)'
        ]
        
        for pattern in award_patterns:
            matches = re.findall(pattern, text)
            if matches:
                if isinstance(matches[0], tuple):
                    certificate_info["award_level"] = ''.join(matches[0]).strip()
                else:
                    certificate_info["award_level"] = matches[0].strip()
                logger.info(f"提取奖项级别: {certificate_info['award_level']}")
                break
        
        # 4. 获奖者
        winner_patterns = [
            r'授予\s*([\u4e00-\u9fa5]{2,10})\s*(?:同学|先生|女士)?',
            r'([\u4e00-\u9fa5]{2,10})\s*同学(?:在|获|荣)',
            r'颁发给\s*([\u4e00-\u9fa5]{2,10})\s*(?:同学|先生|女士)?',
            r'获奖人[:|：]?\s*([\u4e00-\u9fa5]{2,10})\s*',
            r'(?:兹授予|授予|颁发给)\s*([\u4e00-\u9fa5]{2,10}(?:同学)?)\s*'
        ]
        
        for pattern in winner_patterns:
            matches = re.findall(pattern, text)
            if matches:
                for match in matches:
                    name = match.strip()
                    if name and name not in certificate_info["people"]["winner"]:
                        certificate_info["people"]["winner"].append(name)
                        logger.info(f"提取获奖者: {name}")
        
        # 5. 团队成员
        teammate_patterns = [
            r'团队成员[：:]\s*([\u4e00-\u9fa5]{2,10}(?:\s*[,，、]\s*[\u4e00-\u9fa5]{2,10})*)',
            r'参赛队员[：:]\s*([\u4e00-\u9fa5]{2,10}(?:\s*[,，、]\s*[\u4e00-\u9fa5]{2,10})*)',
            r'(?:队员|成员)[：:]\s*([\u4e00-\u9fa5]{2,10}(?:\s*[,，、]\s*[\u4e00-\u9fa5]{2,10})*)'
        ]
        
        for pattern in teammate_patterns:
            matches = re.findall(pattern, text)
            if matches:
                for match in matches:
                    # 分割多个名字
                    names = re.split(r'[,，、]\s*', match)
                    for name in names:
                        name = name.strip()
                        if name and name not in certificate_info["people"]["teammate"]:
                            certificate_info["people"]["teammate"].append(name)
                            logger.info(f"提取团队成员: {name}")
        
        # 6. 指导教师
        teacher_patterns = [
            r'指导教师[：:]\s*([\u4e00-\u9fa5]{2,10}(?:\s*[,，、]\s*[\u4e00-\u9fa5]{2,10})*)',
            r'教师[：:]\s*([\u4e00-\u9fa5]{2,10}(?:\s*[,，、]\s*[\u4e00-\u9fa5]{2,10})*)',
            r'导师[：:]\s*([\u4e00-\u9fa5]{2,10}(?:\s*[,，、]\s*[\u4e00-\u9fa5]{2,10})*)',
            r'([\u4e00-\u9fa5]{2,10})\s*老师'
        ]
        
        for pattern in teacher_patterns:
            matches = re.findall(pattern, text)
            if matches:
                for match in matches:
                    # 分割多个名字
                    names = re.split(r'[,，、]\s*', match)
                    for name in names:
                        name = name.strip()
                        if name and name not in certificate_info["people"]["teacher"]:
                            certificate_info["people"]["teacher"].append(name)
                            logger.info(f"提取指导教师: {name}")
        
        # 5. 颁奖日期
        date_patterns = [
            r'(\d{4}\s*[年/-]\s*\d{1,2}\s*[月/-]\s*\d{1,2}\s*[日号]?)',
            r'(\d{4}\s*[年/-]\s*\d{1,2}\s*[月/-])',
            r'(\d{4}\s*年\s*\d{1,2}\s*月)',
            r'二[○〇零](\d{2})\s*年\s*\d{1,2}\s*月'  # 处理"二〇二三年"这种格式
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # 处理"二〇二三年"这种格式
                if pattern.startswith('二[○〇零]'):
                    year = "20" + matches[0]
                    certificate_info["award_date"] = year
                else:
                    certificate_info["award_date"] = matches[0].strip()
                logger.info(f"提取颁奖日期: {certificate_info['award_date']}")
                break
        
        # 6. 发证机构
        org_patterns = [
            r'([\u4e00-\u9fa5]{2,20})(委员会|协会|学会|中心|部|院|校|司|局|组委会)(?:\s*?印)?',
            r'发证单位[:|：]?\s*([\u4e00-\u9fa5]{3,20})\s*',
            r'颁发单位[:|：]?\s*([\u4e00-\u9fa5]{3,20})\s*',
            r'([\u4e00-\u9fa5]{3,20})(大学|学院|学校)'
        ]
        
        for pattern in org_patterns:
            matches = re.findall(pattern, text)
            if matches:
                if isinstance(matches[0], tuple):
                    certificate_info["issuing_organization"] = ''.join(matches[0]).strip()
                else:
                    certificate_info["issuing_organization"] = matches[0].strip()
                logger.info(f"提取发证机构: {certificate_info['issuing_organization']}")
                break
        
        # 7. 证书编号
        num_patterns = [
            r'证书编号[:|：]?\s*([a-zA-Z0-9-]+)',
            r'编号[:|：]?\s*([a-zA-Z0-9-]+)',
            r'No[\.:]?\s*([a-zA-Z0-9-]+)',
            r'[:|：]?\s*([A-Z][0-9]{6,})'
        ]
        
        for pattern in num_patterns:
            matches = re.findall(pattern, text)
            if matches:
                certificate_info["certificate_number"] = matches[0].strip()
                logger.info(f"提取证书编号: {certificate_info['certificate_number']}")
                break
        
        return certificate_info
    
    def _enhance_with_nlp(self, sorted_texts, certificate_info):
        """
        使用NLP模型增强信息提取的准确性
        
        参数:
            sorted_texts: 排序后的文本块列表
            certificate_info: 初步提取的证书信息
            
        返回:
            增强后的证书信息
        """
        logger.info("使用NLP模型增强信息提取")
        
        # 强化竞赛名称提取 - 检查文本的大小和位置
        # 通常，标题（竞赛名称）会在页面顶部并且字体较大
        if not certificate_info["competition_name"]:
            # 查找页面顶部的文本块
            top_texts = [item for item in sorted_texts[:5] if item['position']['y1'] < sorted_texts[0]['position']['y1'] + 150]
            
            # 根据文本长度和位置判断可能的标题
            for item in top_texts:
                if len(item['text']) >= 6 and len(item['text']) <= 30:
                    # 计算文本框宽度，可能表示字体大小
                    width = item['position']['x2'] - item['position']['x1']
                    # 如果宽度大，可能是标题
                    if width > 200:
                        certificate_info["competition_name"] = item['text']
                        logger.info(f"通过位置和大小提取比赛名称: {certificate_info['competition_name']}")
                        break
        
        # 强化人员信息提取
        if not certificate_info["people"]["winner"] and not certificate_info["people"]["teammate"]:
            # 查找包含关键词的文本行
            for item in sorted_texts:
                # 获奖者识别
                if any(keyword in item['text'] for keyword in ["授予", "颁发", "获奖", "同学"]):
                    # 尝试提取人名 - 假设是2-4个汉字
                    names = re.findall(r'([\u4e00-\u9fa5]{2,4})(?:同学|老师|先生|女士|获|在|的)', item['text'])
                    if names and names[0] not in certificate_info["people"]["winner"]:
                        certificate_info["people"]["winner"].append(names[0])
                        logger.info(f"通过上下文分析提取获奖者: {names[0]}")
                
                # 团队成员识别 - 查找包含多个人名的文本
                if any(keyword in item['text'] for keyword in ["团队", "成员", "队员"]):
                    # 尝试提取多个连续人名
                    team_text = item['text']
                    # 查找"XXX、XXX、XXX"这样的模式
                    team_names = re.findall(r'([\u4e00-\u9fa5]{2,4})[,，、]([\u4e00-\u9fa5]{2,4})[,，、]?([\u4e00-\u9fa5]{2,4})?', team_text)
                    if team_names:
                        for name_group in team_names:
                            for name in name_group:
                                if name and name not in certificate_info["people"]["teammate"]:
                                    certificate_info["people"]["teammate"].append(name)
                                    logger.info(f"通过上下文分析提取团队成员: {name}")
                
                # 指导教师识别
                if any(keyword in item['text'] for keyword in ["指导", "教师", "老师", "导师"]):
                    # 尝试提取教师名
                    teacher_names = re.findall(r'([\u4e00-\u9fa5]{2,4})(?:老师|教授|博士)', item['text'])
                    if teacher_names:
                        for name in teacher_names:
                            if name and name not in certificate_info["people"]["teacher"]:
                                certificate_info["people"]["teacher"].append(name)
                                logger.info(f"通过上下文分析提取指导教师: {name}")
        
        # 强化奖项级别提取 - 通常与获奖者或比赛名称在同一行或相邻行
        if not certificate_info["award_level"]:
            # 如果知道获奖者，查找其附近的文本
            if certificate_info["people"]["winner"]:
                for winner in certificate_info["people"]["winner"]:
                    for i, item in enumerate(sorted_texts):
                        if winner in item['text']:
                            # 获取获奖者所在行及其前后行
                            context_start = max(0, i-1)
                            context_end = min(len(sorted_texts), i+2)
                            context = " ".join([sorted_texts[j]['text'] for j in range(context_start, context_end)])
                            
                            # 查找常见奖项级别
                            award_levels = re.findall(r'(一等奖|二等奖|三等奖|金奖|银奖|铜奖|冠军|亚军|季军|特等奖|优秀奖)', context)
                            if award_levels:
                                certificate_info["award_level"] = award_levels[0]
                                logger.info(f"通过上下文分析提取奖项级别: {certificate_info['award_level']}")
                                return certificate_info  # 找到一个就返回
        
        return certificate_info
    
    def _ensure_directories(self):
        """
        确保必要的目录结构存在
        """
        logger.info(f"检查并创建目录结构: {self.img_dir}, {self.result_dir}")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
    
    def _clean_and_format(self, info):
        """
        清理和格式化提取的信息
        
        参数:
            info: 提取的证书信息
            
        返回:
            清理后的证书信息
        """
        logger.info("清理和格式化证书信息")
        
        # 移除每个字段中的多余空格
        for key in info:
            if isinstance(info[key], str):
                info[key] = re.sub(r'\s+', ' ', info[key]).strip()
        
        # 特定字段的后处理
        # 1. 比赛名称 - 移除可能重复的"比赛"、"竞赛"字样
        if info["competition_name"]:
            # 如果末尾有两个或以上的"比赛"、"竞赛"等字样，保留一个
            info["competition_name"] = re.sub(r'(比赛|竞赛|大赛){2,}$', r'\1', info["competition_name"])
            
            # 如果开头有不必要的字符，去除
            info["competition_name"] = re.sub(r'^[第届].*(赛事|活动)', '', info["competition_name"])
        
        # 2. 获奖日期 - 标准化格式
        if info["award_date"]:
            # 将各种格式统一为YYYY-MM-DD
            info["award_date"] = re.sub(r'(\d{4})\s*[年]\s*(\d{1,2})\s*[月]\s*(\d{1,2})\s*[日号]?', r'\1-\2-\3', info["award_date"])
            info["award_date"] = re.sub(r'(\d{4})\s*[年]\s*(\d{1,2})\s*[月]', r'\1-\2', info["award_date"])
            
            # 处理简写年份
            if re.match(r'^(\d{2})-', info["award_date"]):
                info["award_date"] = "20" + info["award_date"]
        
        # 3. 发证机构 - 清理可能的噪声
        if info["issuing_organization"]:
            # 移除末尾的"印"字、括号等
            info["issuing_organization"] = re.sub(r'[印\(\)（）]$', '', info["issuing_organization"])
        
        # 4. 证书类型 - 统一格式
        if info["certificate_type"]:
            # 如果只是"证书"或"奖状"，并且知道奖项级别，可以组合为更完整的描述
            if info["certificate_type"] in ["证书", "奖状"] and info["award_level"]:
                info["certificate_type"] = f"{info['award_level']}{info['certificate_type']}"
        
        # 5. 人员名称中可能包含的"同学"等称谓处理
        for role in info["people"]:
            cleaned_names = []
            for name in info["people"][role]:
                cleaned_name = re.sub(r'(同学|老师|先生|女士)$', '', name)
                cleaned_names.append(cleaned_name)
            info["people"][role] = cleaned_names
        
        return info
    
    def process_certificate(self, image_path):
        """
        处理单个证书图像
        
        参数:
            image_path: 图像文件路径
            
        返回:
            处理结果字典
        """
        try:
            logger.info(f"开始处理证书: {image_path}")
            
            # 1. 预处理图像
            processed_image = self.preprocess_image(image_path)
            
            # 2. OCR识别文本
            recognized_texts = self.recognize_text(processed_image)
            
            # 3. 提取证书信息
            certificate_info = self.extract_certificate_info(recognized_texts)
            
            # 4. 返回结果字典
            result = {
                "status": "success",
                "certificate_info": certificate_info,
                "ocr_confidence": self._calculate_average_confidence(recognized_texts),
                "image_path": image_path
            }
            
            logger.info("证书处理完成")
            return result
        
        except Exception as e:
            logger.error(f"处理证书时出错: {str(e)}")
            # 错误处理
            result = {
                "status": "error",
                "error_message": str(e),
                "image_path": image_path
            }
            return result
    
    def process_directory(self):
        """
        处理图像目录中的所有证书文件
        
        返回:
            JSON格式的处理结果
        """
        logger.info(f"开始处理目录: {self.img_dir}")
        
        results = []
        
        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        
        for file in os.listdir(self.img_dir):
            file_path = os.path.join(self.img_dir, file)
            if os.path.isfile(file_path) and os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(file_path)
        
        logger.info(f"找到{len(image_files)}个图像文件")
        
        # 处理每个图像
        for image_path in image_files:
            result = self.process_certificate(image_path)
            results.append(result)
            
            # 将结果保存到单独的JSON文件
            base_name = os.path.basename(image_path)
            file_name = os.path.splitext(base_name)[0]
            result_path = os.path.join(self.result_dir, f"{file_name}.json")
            
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"保存结果到: {result_path}")
        
        # 返回所有结果的JSON
        logger.info(f"目录处理完成，共处理{len(results)}个文件")
        return json.dumps(results, ensure_ascii=False, indent=2)
    
    def _calculate_average_confidence(self, texts):
        """
        计算OCR识别的平均置信度
        
        参数:
            texts: OCR识别的文本列表
            
        返回:
            平均置信度
        """
        if not texts:
            return 0
        
        total_confidence = sum(item['confidence'] for item in texts)
        return round(total_confidence / len(texts), 4)


# 示例使用方法
def main():
    # 命令行参数解析
    import argparse
    parser = argparse.ArgumentParser(description='中文奖状OCR识别与信息提取系统')
    
    parser.add_argument('--img_dir', type=str, default='data/img',
                        help='输入图像目录 (默认: data/img)')
    
    parser.add_argument('--result_dir', type=str, default='data/result',
                        help='结果输出目录 (默认: data/result)')
    
    parser.add_argument('--use_gpu', action='store_true',
                        help='是否使用GPU加速 (如果可用)')
    
    parser.add_argument('--single_file', type=str, default=None,
                        help='处理单个文件而非整个目录')
    
    args = parser.parse_args()
    
    # 初始化OCR处理器
    processor = CertificateOCR(
        use_gpu=args.use_gpu,
        img_dir=args.img_dir,
        result_dir=args.result_dir
    )
    
    if args.single_file:
        # 处理单个证书
        result = processor.process_certificate(args.single_file)
        
        # 输出结果
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # 保存结果到文件
        base_name = os.path.basename(args.single_file)
        file_name = os.path.splitext(base_name)[0]
        result_path = os.path.join(args.result_dir, f"{file_name}.json")
        
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到 {result_path}")
    else:
        # 处理整个目录
        results_json = processor.process_directory()
        
        # 输出汇总结果
        summary_path = os.path.join(args.result_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(results_json)
        
        print(f"所有结果已保存到 {args.result_dir} 目录")
        print(f"汇总结果已保存到 {summary_path}")


if __name__ == "__main__":
    main()
    