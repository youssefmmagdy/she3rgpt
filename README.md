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
