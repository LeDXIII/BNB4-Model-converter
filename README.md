## 📖 Быстрая навигация / Quick Navigation

🇷🇺 [Русская документация](#-русская-документация) | 🇬🇧 [English Documentation](#-english-documentation)

---

# 🇷🇺 Русская документация

# 🔄 vLLM/LLM Model BNB4 Converter

**Удобный конвертер AI-моделей в 4-битный формат (bnb4) с графическим интерфейсом**

Программа позволяет легко квантовать современные AI-модели для эффективного использования на потребительских видеокартах. Поддерживает Vision-Language модели и обычные LLM с автоматическим определением типа и оптимальными настройками.

## ✨ Основные возможности

- **🖼️ Поддержка различных типов моделей:**
  - Vision-Language модели (NuMarkdown, Qwen2.5-VL, MiniCPM-V и др.)
  - Модели машинного перевода (Hunyuan-MT, NLLB, M2M100)
  - Большие языковые модели (Qwen, Mistral, Llama и др.)

- **⚙️ Гибкие настройки конвертации:**
  - Выбор устройства обработки (GPU/CPU/Auto)
  - Настройка длины контекста (от 512 до 262k токенов)
  - Тип квантования (NF4/FP4)
  - Формат SafeTensors

- **🎮 Простой графический интерфейс:**
  - Интуитивно понятный интерфейс
  - Группированный список популярных моделей
  - Подробный журнал операций с временными метками
  - Индикатор прогресса выполнения
  - Автосохранение пользовательских настроек

- **🚀 Автоматическая установка:**
  - Автоматическое определение версии CUDA
  - Установка совместимой версии PyTorch
  - Разрешение конфликтов зависимостей
  - Поддержка как GPU, так и CPU-режимов

<img width="1102" height="832" alt="image" src="https://github.com/user-attachments/assets/4a86929a-40e3-48d5-8f2e-3343325ab18c" />


## 📋 Системные требования

### Минимальные требования:
- **ОС:** Windows 10/11
- **Python:** 3.8 или новее
- **ОЗУ:** 8 GB (рекомендуется 16+ GB)
- **Свободное место:** 50+ GB для моделей

### Для работы на GPU:
- **Видеокарта:** NVIDIA с поддержкой CUDA
- **Видеопамять:** 8+ GB
- **Драйверы:** Актуальные NVIDIA драйверы

## 🛠️ Установка и запуск

1. **Клонирование репозитория:**
```bash
git clone https://github.com/username/BNB4-Model-converter.git
cd BNB4-Model-converter
```

2. **Автоматическая установка зависимостей:**
```bash
install.bat
```
Скрипт автоматически:
- Проверит версию Python
- Создаст виртуальное окружение
- Определит версию CUDA и установит совместимый PyTorch
- Установит все необходимые библиотеки

3. **Запуск программы:**
```bash
run.bat
```

## 🎯 Процесс работы

### Шаг 1: Выбор модели
- Выберите модель из предустановленного списка или введите URL вручную
- Программа автоматически определит тип модели и покажет информацию о ней

### Шаг 2: Настройка параметров
- **Устройство:** Auto (рекомендуется), GPU или CPU
- **Длина контекста:** Выберите подходящее значение для ваших задач  
- **Квантование:** NF4 (bnb4) - стандарт для 4-битного квантования
- **Папка сохранения:** Укажите путь для сохранения модели

### Шаг 3: Запуск конвертации
- Нажмите "🚀 Запустить конвертацию"
- Следите за прогрессом в реальном времени
- Получите готовую квантованную модель в указанной папке

## 💡 Преимущества квантования

- **Экономия памяти:** Уменьшение размера модели в ~4 раза
- **Доступность:** Запуск больших моделей на потребительских GPU
- **Скорость:** Ускорение инференса при минимальной потере качества
- **Совместимость:** Полная совместимость с библиотекой transformers

## 🔧 Дополнительные возможности

- **Автосохранение настроек** - ваши предпочтения сохраняются между сессиями
- **Логирование** - полная информация о процессе конвертации
- **Обработка ошибок** - понятные сообщения об ошибках и способах их решения
- **Гибкая настройка** - возможность добавления собственных моделей

***

⭐ **Поставь звезду, если было полезно!**


# 🇬🇧 English Documentation

# 🔄 vLLM/LLM Model BNB4 Converter

**User-friendly AI model converter to 4-bit format (bnb4) with graphical interface**

The application allows easy quantization of modern AI models for efficient use on consumer graphics cards. Supports Vision-Language models and regular LLMs with automatic type detection and optimal settings.

## ✨ Key Features

- **🖼️ Support for various model types:**
  - Vision-Language models (NuMarkdown, Qwen2.5-VL, MiniCPM-V, etc.)
  - Machine translation models (Hunyuan-MT, NLLB, M2M100)
  - Large Language Models (Qwen, Mistral, Llama, etc.)

- **⚙️ Flexible conversion settings:**
  - Processing device selection (GPU/CPU/Auto)
  - Context length configuration (from 512 to 262k tokens)
  - Quantization type (NF4/FP4)
  - SafeTensors format

- **🎮 Simple graphical interface:**
  - Intuitive user interface
  - Grouped list of popular models
  - Detailed operation log with timestamps
  - Real-time progress indicator
  - Auto-save user settings

- **🚀 Automatic installation:**
  - Automatic CUDA version detection
  - Compatible PyTorch installation
  - Dependency conflict resolution
  - Support for both GPU and CPU modes

<img width="1102" height="832" alt="image" src="https://github.com/user-attachments/assets/e170c302-8620-4b86-a2ef-211ad5f992cb" />


## 📋 System Requirements

### Minimum Requirements:
- **OS:** Windows 10/11
- **Python:** 3.8 or newer
- **RAM:** 8 GB (16+ GB recommended)
- **Free space:** 50+ GB for models

### For GPU operation:
- **Graphics card:** NVIDIA with CUDA support
- **VRAM:** 8+ GB
- **Drivers:** Latest NVIDIA drivers

## 🛠️ Installation and Launch

1. **Clone repository:**
```bash
git clone https://github.com/username/BNB4-Model-converter.git
cd BNB4-Model-converter
```

2. **Automatic dependency installation:**
```bash
install.bat
```
The script automatically:
- Checks Python version
- Creates virtual environment
- Detects CUDA version and installs compatible PyTorch
- Installs all necessary libraries

3. **Launch application:**
```bash
run.bat
```

## 🎯 Workflow

### Step 1: Model Selection
- Choose a model from the preset list or enter URL manually
- The program automatically detects model type and displays information

### Step 2: Parameter Configuration
- **Device:** Auto (recommended), GPU, or CPU
- **Context length:** Choose appropriate value for your tasks
- **Quantization:** NF4 (bnb4) - standard for 4-bit quantization
- **Save folder:** Specify path for model storage

### Step 3: Start Conversion
- Click "🚀 Start Conversion"
- Monitor progress in real-time
- Get ready quantized model in specified folder

## 💡 Quantization Benefits

- **Memory savings:** Model size reduction by ~4x
- **Accessibility:** Running large models on consumer GPUs
- **Speed:** Inference acceleration with minimal quality loss
- **Compatibility:** Full compatibility with transformers library

## 🔧 Additional Features

- **Auto-save settings** - your preferences are saved between sessions
- **Logging** - complete information about the conversion process
- **Error handling** - clear error messages and solution guidance
- **Flexible configuration** - ability to add custom models

If you encounter any issues or have questions, create an Issue in this repository.

***

⭐ **Star this repository if it was helpful!**
