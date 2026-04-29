# 🔄 BNB4 Model Converter v2.0

Удобный конвертер AI-моделей в 4-битный формат (bnb4) с графическим интерфейсом.

Квантует модели для запуска на потребительских GPU с уменьшением размера в ~4 раза.

🇷 [Русская документация](#-русская-документация) | 🇬🇧 [English](#-english-documentation)

---

## ✨ Возможности

* **Поддержка моделей:** Vision-Language (MinerU, Qwen-VL, MiniCPM-V), LLM (Qwen 3.5/3/2.5, Llama, Mistral, Gemma, Phi), перевод (Hunyuan-MT, NLLB), Embedding
* **Автоопределение типа:** vision / text / embedding по архитектуре модели
* **3-этапный fallback:** AutoModelForImageTextToText → AutoModelForCausalLM → AutoModel
* **Настройки:** контекст 512–262k, квантование NF4/FP4, устройство GPU/CPU/Auto
* **Мониторинг VRAM** через nvidia-smi в реальном времени
* **Каталог моделей** с Qwen 3.5, Qwen 3, Qwen2.5, Llama 3.x, Gemma 2, Phi-3.5 и др.

## 📋 Требования

| | Минимум | Рекомендация |
|---|---|---|
| ОС | Windows 10/11 | Windows 10/11 |
| Python | 3.10+ | 3.10+ |
| ОЗУ | 8 GB | 16+ GB |
| Диск | 50+ GB | SSD 50+ GB |
| GPU | NVIDIA CUDA 8+ GB VRAM | NVIDIA CUDA 12+ GB VRAM |

## 🛠 Установка

```bash
git clone https://github.com/LeDXIII/BNB4-Model-converter.git
cd BNB4-Model-converter
install.bat
```

`install.bat` автоматически: проверит Python, создаст venv, определит CUDA, установит PyTorch и зависимости.

## 🚀 Запуск

```bash
Run.bat
```

## 🎯 Использование

1. **Выберите модель** из списка в `models.json` или введите URL вручную
2. **Настройте параметры:** контекст, квантование (NF4/FP4), устройство
3. **Нажмите «Запустить конвертацию»** — следите за прогрессом
4. **Готовая модель** сохранится в папку `output/<model>-bnb4`

## 📦 Каталог моделей

Модели хранятся в `models.json` — редактируйте для добавления новых:

```json
{
  "groups": {
    " Название категории": {
      "Название модели": {
        "url": "author/model-name",
        "type": "auto|vision|text|embedding",
        "params": "7B",
        "desc": "Описание"
      }
    }
  }
}
```

| Поле | Значение |
|------|----------|
| `url` | Репозиторий на HuggingFace (`author/name`) |
| `type` | `auto` (автоопределение), `vision`, `text`, `embedding` |
| `params` | Размер параметров для отображения |
| `desc` | Краткое описание в интерфейсе |

### Текущие категории
* **Qwen 3.5:** 0.8B, 2B, 4B, 9B
* **Qwen 3:** 0.6B, 1.7B, 4B, 8B + Embedding
* **Vision-Language:** MinerU, Qwen-VL, MiniCPM-V, GLM-4V, InternVL2, NuMarkdown, DeepSeek-OCR
* **LLM:** Qwen2.5, Mistral, Llama, Gemma, Phi
* **Перевод:** Hunyuan-MT, NLLB, M2M100

## 💡 Преимущества BNB4

| | FP16 | BNB4 |
|---|---|---|
| Размер | 100% | ~25% |
| VRAM | Полный | ~25% |
| Качество | Базовое | Минимальная потеря |

## 🔧 Формат вывода

Квантованная модель сохраняется в папку `<model>-bnb4/` с файлами:
- `model.safetensors` — веса в 4-бит (nf4/fp4)
- `config.json` — конфигурация
- `tokenizer.*` — токенизатор
- `preprocessor_config.json` — для VL-моделей

## 📦 Поддерживаемые модели

### Qwen 3.5 (новые)
Qwen3.5-0.8B, Qwen3.5-2B, Qwen3.5-4B, Qwen3.5-9B

### Qwen 3
Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Qwen3-Embedding

### Vision-Language
MinerU2.5-Pro, Qwen2.5-VL, MiniCPM-V, GLM-4V, InternVL2, NuMarkdown, DeepSeek-OCR

### LLM
Qwen2.5 (0.5B–14B), Mistral-7B, Llama 3.1/3.2, Gemma 2, Phi-3.5

### Перевод
Hunyuan-MT, NLLB-200, M2M100

---

## 🇧 English Documentation

### Overview
A user-friendly converter for quantizing AI models to 4-bit BNB format with a graphical interface. Reduces model size by ~4× for running on consumer GPUs.

### Features
* **Model support:** Vision-Language (MinerU, Qwen-VL, MiniCPM-V), LLM (Qwen 3.5/3/2.5, Llama, Mistral, Gemma, Phi), Translation (Hunyuan-MT, NLLB), Embedding
* **Auto-detection:** vision / text / embedding based on model architecture
* **3-stage fallback:** AutoModelForImageTextToText → AutoModelForCausalLM → AutoModel
* **Settings:** context length 512–262k, quantization NF4/FP4, device GPU/CPU/Auto
* **Real-time VRAM monitoring** via nvidia-smi
* **Model catalog** with Qwen 3.5, Qwen 3, Qwen2.5, Llama 3.x, Gemma 2, Phi-3.5, and more

### Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Windows 10/11 | Windows 10/11 |
| Python | 3.10+ | 3.10+ |
| RAM | 8 GB | 16+ GB |
| Disk | 50+ GB | SSD 50+ GB |
| GPU | NVIDIA CUDA 8+ GB VRAM | NVIDIA CUDA 12+ GB VRAM |

### Installation

```bash
git clone https://github.com/LeDXIII/BNB4-Model-converter.git
cd BNB4-Model-converter
install.bat
```

`install.bat` automatically: checks Python, creates venv, detects CUDA version, installs PyTorch and dependencies.

### Launch

```bash
Run.bat
```

### Usage

1. **Select a model** from the catalog in `models.json` or enter a URL manually
2. **Configure parameters:** context length, quantization (NF4/FP4), device
3. **Click "Start Conversion"** — monitor progress in real time
4. **The quantized model** is saved to `output/<model>-bnb4`

### Model Catalog

Models are stored in `models.json` — edit it to add new ones:

```json
{
  "groups": {
    " Category Name": {
      "Model Name": {
        "url": "author/model-name",
        "type": "auto|vision|text|embedding",
        "params": "7B",
        "desc": "Description"
      }
    }
  }
}
```

| Field | Description |
|-------|-------------|
| `url` | HuggingFace repository (`author/name`) |
| `type` | `auto` (auto-detect), `vision`, `text`, `embedding` |
| `params` | Parameter count for display |
| `desc` | Short description shown in the UI |

### Current Categories
* **Qwen 3.5:** 0.8B, 2B, 4B, 9B
* **Qwen 3:** 0.6B, 1.7B, 4B, 8B + Embedding
* **Vision-Language:** MinerU, Qwen-VL, MiniCPM-V, GLM-4V, InternVL2, NuMarkdown, DeepSeek-OCR
* **LLM:** Qwen2.5, Mistral, Llama, Gemma, Phi
* **Translation:** Hunyuan-MT, NLLB, M2M100

### BNB4 Benefits

| | FP16 | BNB4 |
|---|---|---|
| Size | 100% | ~25% |
| VRAM | Full | ~25% |
| Quality | Baseline | Minimal loss |

### Output Format

The quantized model is saved to `<model>-bnb4/` with the following files:
- `model.safetensors` — 4-bit weights (nf4/fp4)
- `config.json` — model configuration
- `tokenizer.*` — tokenizer files
- `preprocessor_config.json` — for Vision-Language models

### Supported Models

#### Qwen 3.5 (new)
Qwen3.5-0.8B, Qwen3.5-2B, Qwen3.5-4B, Qwen3.5-9B

#### Qwen 3
Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Qwen3-Embedding

#### Vision-Language
MinerU2.5-Pro, Qwen2.5-VL, MiniCPM-V, GLM-4V, InternVL2, NuMarkdown, DeepSeek-OCR

#### LLM
Qwen2.5 (0.5B–14B), Mistral-7B, Llama 3.1/3.2, Gemma 2, Phi-3.5

#### Translation
Hunyuan-MT, NLLB-200, M2M100

---

⭐ Star this repository if it was helpful!
