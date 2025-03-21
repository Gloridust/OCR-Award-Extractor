# 📜 中文证书OCR内容提取工具

![版本](https://img.shields.io/badge/版本-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.10-green)
![License](https://img.shields.io/badge/许可证-MIT-yellow)

## 📋 项目简介

这是一个基于Python的工具，用于自动识别和提取中文奖状/证书中的关键信息，如比赛名称、奖项级别、项目名称、获奖者和指导教师姓名等。特别适用于大学生竞赛证书，如蓝桥杯、互联网+、计算机设计大赛等。

## 🚀 特性

- ✅ 自动识别证书图像中的文本
- ✅ 智能提取关键信息
- ✅ 处理各种格式的证书
- ✅ 高精度中文文本识别
- ✅ 支持批量处理多张证书
- ✅ JSON格式输出结果

## 🛠️ 技术栈

- **Python 3.10** - 编程语言
- **PaddleOCR** - 高性能中文OCR引擎
- **OpenCV** - 图像处理
- **正则表达式** - 文本模式匹配
- **规则匹配** - 信息提取算法

## ⚙️ 安装说明

### 前提条件

- Python 3.10
- 足够的磁盘空间用于OCR模型

### 步骤

1. 克隆仓库或下载源代码:

```bash
git clone https://github.com/Gloridust/OCR-Award-Extractor.git
cd OCR-Award-Extractor
```

2. 安装依赖项:

```bash
pip install -r requirements.txt
```

这将安装所有必要的库，包括PaddleOCR、PyTorch、OpenCV和NumPy。

## 📂 目录结构

```
.
├── certificate_ocr.py   # 主程序文件
├── requirements.txt     # 依赖列表
├── data                 # 数据目录
│   ├── img              # 证书图像目录
│   └── result           # 输出结果目录
└── README.md            # 说明文档
```

## 🔍 使用方法

1. **准备证书图像**

   将您需要处理的证书图像放入 `data/img/` 目录。支持的格式包括JPG、JPEG、PNG、TIF和TIFF。

2. **运行程序**

   ```bash
   python certificate_ocr.py
   ```

3. **查看结果**

   处理完成后，您可以在 `data/result/` 目录中找到以下文件:
   - 针对每个输入图像的单独JSON文件
   - `all_results.json` 包含所有结果的汇总文件

## 🧠 工作原理

该项目结合了计算机视觉和文本分析技术，主要分为以下几个步骤:

1. **图像预处理** - 使用OpenCV增强图像质量
2. **文本识别** - 使用PaddleOCR提取证书中的文本
3. **信息提取** - 通过规则匹配和模式识别提取关键信息
4. **结果输出** - 生成结构化的JSON结果

### 核心技术点:

- **多策略OCR** - 尝试多种预处理方法以获得最佳结果
- **模式匹配** - 使用针对证书特点定制的正则表达式
- **文本分析** - 智能识别人名、项目名称和其他关键信息
- **错误处理** - 完善的异常处理和日志记录

## 📊 输出格式

输出的JSON格式示例:

```json
{
  "status": "success",
  "certificate_info": {
    "competition_name": "第十届中国国际创新大赛",
    "award_level": "校级铜奖",
    "project_name": "项目名称",
    "people": {
      "winner": ["张三", "李四", "王五", "赵六"],
      "teacher": ["陈老师"]
    }
  },
  "ocr_confidence": 0.9146,
  "image_path": "data/img/pic.jpg"
}
```

## 🔧 常见问题解决

- **OCR识别不准确?** 
  - 尝试提高图像质量或调整图像对比度
  - 确保图像没有严重的扭曲或模糊

- **找不到特定信息?**
  - 检查证书格式是否特殊
  - 可能需要添加额外的模式匹配规则

- **处理速度慢?**
  - 首次运行时需要下载OCR模型
  - 考虑使用GPU加速处理

## 📝 注意事项

- 本工具主要针对中国大学生竞赛证书优化
- 对于非标准格式的证书可能需要调整规则
- 最佳图像质量可以获得最准确的结果

## 📄 许可证

本项目采用MIT许可证。详情请参阅LICENSE文件。