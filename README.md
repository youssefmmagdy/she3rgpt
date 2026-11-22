<div align="center">

![trans_thumbnail](https://github.com/user-attachments/assets/2f19917c-488e-409e-9326-a3391eff225d)

# شِعرGPT

**نموذج لغوي توليدي مبني على معمارية Transformer لتوليد الشعر العربي**

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

<div dir="rtl" align="right">

## 📌 نبذة عن المشروع

شِعرGPT هو نموذج لغوي توليدي (Generative Pre-trained Transformer) مُدرَّب على مجموعة بيانات من الشعر العربي الكلاسيكي. الهدف من المشروع هو بناء نموذج Decoder-only Transformer من الصفر باستخدام PyTorch، قادر على توليد أبيات شعرية عربية جديدة بأسلوب يحاكي الشعر الكلاسيكي.

يتضمن المشروع أيضاً دفترَين تعليميَّين يشرحان أساسيات الـ Neural Networks وكيفية بنائها من الصفر، بدءاً من مفهوم الـ Neuron وصولاً إلى تدريب نموذج Transformer كامل.

---

## 🏗️ معمارية النموذج

يعتمد النموذج على معمارية **Decoder-only Transformer** بالمواصفات التالية:

| المعامل | القيمة |
|---|---|
| Token Embedding Dimension | `500` |
| Position Embedding | `256` (Block Size) |
| عدد طبقات الـ Transformer Blocks | `12` |
| عدد رؤوس الـ Attention (Heads) | `10` |
| حجم كل Head | `50` |
| الـ FeedForward الداخلي | `500 → 2000 → 500` |
| دالة التنشيط (Activation) | `ReLU` |
| Dropout | `0.2` |
| حجم المفردات (Vocab Size) | `~30` حرف عربي + مسافة + سطر جديد |
| الـ Tokenization | على مستوى الحروف (Character-level) |

### آلية العمل

1. **الـ Tokenization**: يتم تحويل النص العربي إلى سلسلة من الأرقام الصحيحة، حيث يُمثَّل كل حرف عربي برقم فريد
2. **الـ Embedding**: يمر كل Token عبر طبقتي Token Embedding و Position Embedding ثم تُجمعان
3. **الـ Self-Attention**: تستخدم كل طبقة Causal Self-Attention (مقنّعة) لمنع النموذج من النظر إلى الحروف المستقبلية
4. **الـ FeedForward Network**: شبكة أمامية بطبقتين مع ReLU و Dropout
5. **الـ Residual Connections**: وصلات متبقية حول كل من الـ Attention و الـ FeedForward
6. **الـ Layer Normalization**: تطبيع قبل كل طبقة فرعية (Pre-Norm)
7. **التوليد**: توليد تسلسلي (Autoregressive) باستخدام Multinomial Sampling

---

## 📊 تفاصيل التدريب

| المعامل | القيمة |
|---|---|
| عدد الـ Iterations | `15,000` |
| Batch Size | `64` |
| Block Size (Context Window) | `256` |
| Learning Rate | `3e-4` |
| Optimizer | `AdamW` |
| Loss Function | `Cross-Entropy` |
| التقييم | كل `100` خطوة |
| الجهاز | `CUDA` (GPU) |

### مجموعة البيانات

تم التدريب على مجموعة بيانات الشعر العربي من Kaggle، تتضمن آلاف الأبيات من الشعر العربي الكلاسيكي. تمت معالجة البيانات بتصفية الحروف غير العربية والاحتفاظ بالحروف العربية والمسافات وعلامات السطر الجديد فقط. تم تقسيم البيانات بنسبة 90% للتدريب و 10% للتحقق.

---

## 📝 عيّنة من الشعر المُولَّد

</div>

<div dir="rtl" align="right">

```
جيش لقاتله لا يرام
وقد أشننت شقتيه الرمال
يهاب فؤادك أمر همام
فبم هاج يشكو وابتسم
وقصرت الأسى والهجر ذنبى
ألا لقيت الديار القفرة
فأين الكمال أسود من أحد
وخيرت للقبر والصحبه
أجب لا تساعد عند الكربه
ذوو الأدب الثاني عند الصحبه
ومن لا ينام بعين القرب
فإن فكم لا تفك في الكثر
وإن لم يزد واقعا بالطمع
فإنسانا علىلاه غنى
بغ وانقطاعا لعمري بلمي
ومن أشكر الله نعمى الوامي
في حلى الحر سحب أثيل سهام
وعن الله أنزه الكلام
وأعين أفكار العيوب
```

</div>

<div dir="rtl" align="right">

---

## 📂 هيكل المشروع

</div>

```
she3rgpt/
├── she3rgpt.ipynb                    # الدفتر الرئيسي - بناء وتدريب النموذج
├── Intro_Neural_Networks.ipynb       # مقدمة في الـ Neural Networks من الصفر
├── Intro_NN_ET_Prediction.ipynb      # تطبيق عملي: التنبؤ بالـ Evapotranspiration
├── poetry_generated_5K_36M_1.3.txt   # عينة من الشعر المُولَّد (5000 حرف)
├── data/
│   └── arabic_poetry_text.txt        # مجموعة بيانات الشعر العربي
├── slides/                           # شرائح العرض
└── README.md
```

<div dir="rtl" align="right">

---

## 📓 الدفاتر التعليمية

### 1. مقدمة في الـ Neural Networks

يشرح هذا الدفتر أساسيات الشبكات العصبية من الصفر بدون أي مكتبة خارجية، ويتضمن:
- بناء Neuron واحد يدوياً (الـ Forward Pass)
- تجميع عدة Neurons في Layer
- بناء Multi-Layer Perceptron (MLP) كامل
- فهم الـ Weights والـ Biases والـ Linear Transformations

### 2. التنبؤ بالـ Evapotranspiration

تطبيق عملي على مهمة Regression لتوقع كمية التبخر-نتح (Evapotranspiration) من بيانات مناخية، ويتضمن:
- بناء MLP يدوي مع Backpropagation
- الانتقال إلى PyTorch والمقارنة
- استخدام Activation Functions (ReLU)
- تقييم الأداء باستخدام MSE و MAE و RMSE
- تطبيق Data Normalization والـ Train/Test Split

### 3. شِعرGPT (الدفتر الرئيسي)

بناء وتدريب نموذج Transformer كامل لتوليد الشعر العربي:
- معالجة البيانات والـ Character-level Tokenization
- بناء Self-Attention و Multi-Head Attention
- تركيب الـ Transformer Blocks مع Residual Connections و Layer Normalization
- التدريب مع AdamW Optimizer والـ Cross-Entropy Loss
- توليد نصوص جديدة بأسلوب Autoregressive Generation
- حفظ الـ Model Checkpoint

---

## 🚀 كيفية الاستخدام

### المتطلبات

</div>

```bash
pip install torch numpy pandas matplotlib
```

<div dir="rtl" align="right">

### التشغيل

1. قم باستنساخ المستودع:

</div>

```bash
git clone https://github.com/YOUR_USERNAME/she3rgpt.git
cd she3rgpt
```

<div dir="rtl" align="right">

2. افتح الدفتر الرئيسي `she3rgpt.ipynb` في Jupyter Notebook أو VS Code
3. شغّل جميع الخلايا بالترتيب
4. سيتم تدريب النموذج وتوليد أبيات شعرية جديدة تلقائياً

---

## 🎯 المسار التعليمي المقترح

</div>

```
Intro_Neural_Networks.ipynb → Intro_NN_ET_Prediction.ipynb → she3rgpt.ipynb
        (الأساسيات)                 (تطبيق عملي)              (النموذج الكامل)
```



<div align="center">

![trans_thumbnail](https://github.com/user-attachments/assets/2f19917c-488e-409e-9326-a3391eff225d)

# She3rGPT

**A Transformer-Based Generative Language Model for Arabic Poetry Generation**

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📌 Project Overview

**She3rGPT** is a generative language model based on the Transformer architecture, trained on a dataset of classical Arabic poetry.

The primary goal of this project is to **build a Decoder-only Transformer from scratch using PyTorch**, capable of generating original Arabic poetry that resembles the linguistic patterns and style of classical Arabic literature.

In addition to the main model, the repository includes two educational notebooks that explore the fundamentals of Neural Networks and their implementation from scratch, starting with individual Neurons and progressing toward training a complete Transformer model.

---

## 🏗️ Model Architecture

The model follows a **Decoder-only Transformer** architecture with the following specifications:

| Parameter | Value |
|---|---|
| Token Embedding Dimension | `500` |
| Position Embedding | `256` (Block Size) |
| Transformer Blocks | `12` |
| Attention Heads | `10` |
| Head Dimension | `50` |
| FeedForward Network | `500 → 2000 → 500` |
| Activation Function | `ReLU` |
| Dropout | `0.2` |
| Vocabulary Size | `~30` Arabic characters + space + newline |
| Tokenization | Character-level |

### How It Works

1. **Tokenization**: Arabic text is converted into a sequence of integer tokens, where each character is assigned a unique numerical ID.
2. **Embeddings**: Each token is passed through Token Embedding and Positional Embedding layers, and their representations are added together.
3. **Self-Attention**: Each Transformer block uses Masked (Causal) Self-Attention to prevent the model from attending to future characters.
4. **FeedForward Network**: A two-layer feedforward network with ReLU activation and Dropout.
5. **Residual Connections**: Skip connections are applied around both the Attention and FeedForward sublayers.
6. **Layer Normalization**: Pre-Layer Normalization (Pre-Norm) is applied before each sublayer.
7. **Text Generation**: New characters are generated autoregressively using Multinomial Sampling.

---

## 📊 Training Configuration

| Parameter | Value |
|---|---|
| Training Iterations | `15,000` |
| Batch Size | `64` |
| Block Size (Context Window) | `256` |
| Learning Rate | `3e-4` |
| Optimizer | `AdamW` |
| Loss Function | `Cross-Entropy` |
| Evaluation Interval | Every `100` steps |
| Computing Device | `CUDA` (GPU) |

### Dataset

The model was trained on an **Arabic Poetry Dataset from Kaggle**, containing thousands of verses from classical Arabic poetry.

The dataset underwent preprocessing to remove non-Arabic characters while preserving Arabic letters, spaces, and newline characters.

The processed dataset was divided into:

- **90% Training Data**
- **10% Validation Data**

---

## 📝 Sample Generated Poetry

The following is an example of Arabic poetry generated by She3rGPT.

<div dir="rtl" align="right">

    جيش لقاتله لا يرام
    وقد أشننت شقتيه الرمال
    يهاب فؤادك أمر همام
    فبم هاج يشكو وابتسم
    وقصرت الأسى والهجر ذنبى
    ألا لقيت الديار القفرة
    فأين الكمال أسود من أحد
    وخيرت للقبر والصحبه
    أجب لا تساعد عند الكربه
    ذوو الأدب الثاني عند الصحبه
    ومن لا ينام بعين القرب
    فإن فكم لا تفك في الكثر
    وإن لم يزد واقعا بالطمع
    فإنسانا علىلاه غنى
    بغ وانقطاعا لعمري بلمي
    ومن أشكر الله نعمى الوامي
    في حلى الحر سحب أثيل سهام
    وعن الله أنزه الكلام
    وأعين أفكار العيوب

</div>

> **Note:** The generated text demonstrates the model's ability to learn Arabic character sequences and poetic language patterns. However, as a character-level model, its output may contain grammatical inconsistencies, incomplete meanings, and irregular poetic structures.

---

## 📂 Project Structure

```text
she3rgpt/
├── she3rgpt.ipynb                    # Main notebook: Building and training the Transformer
├── Intro_Neural_Networks.ipynb       # Introduction to Neural Networks from scratch
├── Intro_NN_ET_Prediction.ipynb      # Practical application: Evapotranspiration prediction
├── poetry_generated_5K_36M_1.3.txt   # Generated Arabic poetry sample (5,000 characters)
├── data/
│   └── arabic_poetry_text.txt        # Arabic poetry dataset
├── slides/                           # Presentation slides
└── README.md
```

---

## 📓 Educational Notebooks

### 1. Introduction to Neural Networks

This notebook introduces the fundamental concepts of Neural Networks by implementing their core components from scratch, without relying on external machine learning frameworks.

Topics covered:

- Implementing a single Neuron manually (Forward Pass)
- Combining multiple Neurons into a Layer
- Building a complete Multi-Layer Perceptron (MLP)
- Understanding Weights, Biases, and Linear Transformations

### 2. Evapotranspiration Prediction

A practical Regression application that predicts **Evapotranspiration (ET)** using climate and weather data.

Topics covered:

- Implementing an MLP manually with Backpropagation
- Transitioning to PyTorch and comparing implementations
- Applying Activation Functions (ReLU)
- Evaluating model performance using MSE, MAE, and RMSE
- Applying Data Normalization and Train/Test Splitting

### 3. She3rGPT (Main Notebook)

The main notebook focuses on building and training a complete Transformer model for Arabic poetry generation.

Topics covered:

- Data preprocessing and Character-level Tokenization
- Implementing Self-Attention and Multi-Head Attention
- Constructing Transformer Blocks with Residual Connections and Layer Normalization
- Training with the AdamW Optimizer and Cross-Entropy Loss
- Generating new Arabic text using Autoregressive Generation
- Saving the trained Model Checkpoint

---

## 🚀 Getting Started

### Prerequisites

Install the required Python dependencies:

```bash
pip install torch numpy pandas matplotlib
```

### Installation and Execution

**1. Clone the repository:**

```bash
git clone https://github.com/YOUR_USERNAME/she3rgpt.git
cd she3rgpt
```

**2. Open the main notebook:**

Open `she3rgpt.ipynb` using Jupyter Notebook or Visual Studio Code.

**3. Execute the notebook:**

Run all cells sequentially to initialize the model, preprocess the dataset, and begin training.

**4. Generate Arabic poetry:**

After training, the model will automatically generate new Arabic poetry based on the patterns learned from the dataset.

---

## 🎯 Recommended Learning Path

For those interested in understanding Neural Networks and Transformers from the ground up, the following learning sequence is recommended:

```text
Intro_Neural_Networks.ipynb
            ↓
Intro_NN_ET_Prediction.ipynb
            ↓
       she3rgpt.ipynb
```

| Stage | Notebook | Key Concepts |
|---|---|---|
| 1 | Intro Neural Networks | Neuron, Layer, MLP, Forward Pass |
| 2 | ET Prediction | Backpropagation, Loss Functions, Optimization, PyTorch |
| 3 | She3rGPT | Transformer, Self-Attention, Embeddings, Text Generation |

This progression provides a structured learning experience, beginning with fundamental Neural Network concepts and advancing toward implementing a complete Generative Transformer Language Model.

---

## 📄 License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute the code in accordance with the terms of the license.

<div dir="rtl" align="right">

| المرحلة | الدفتر | المفاهيم |
|---|---|---|
| 1 | Intro Neural Networks | Neuron, Layer, MLP, Forward Pass |
| 2 | ET Prediction | Backpropagation, Loss Functions, Optimization, PyTorch |
| 3 | شِعرGPT | Transformer, Self-Attention, Embeddings, Text Generation |

---

## 📄 الرخصة

هذا المشروع مرخص تحت رخصة MIT. يمكنك استخدامه وتعديله ومشاركته بحرية.

</div>
