# -*- coding: utf-8 -*-
"""
Графический интерфейс для конвертации AI-моделей в 4-битный формат (bnb4).
Поддерживает Vision-Language модели, модели перевода и обычные LLM.
"""

import sys
import json
import traceback
import warnings
from pathlib import Path
from multiprocessing import Process, Event, Pipe, set_start_method
import torch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

warnings.filterwarnings('ignore', category=FutureWarning)

set_start_method('spawn', force=True)

def download_worker(cfg, conn, stop_evt):
    """Фоновый процесс загрузки и квантования модели"""
    import logging, gc, warnings
    from transformers import (
        AutoModelForImageTextToText, AutoModelForCausalLM,
        AutoTokenizer, AutoImageProcessor, BitsAndBytesConfig
    )
    
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', message=".*Xet Storage.*")
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def send_log(msg):
        if stop_evt.is_set(): 
            raise KeyboardInterrupt()
        conn.send(('log', msg))

    def send_progress(pct, stage=""):
        if stop_evt.is_set(): 
            raise KeyboardInterrupt()
        conn.send(('progress', pct, stage))

    def send_status(status):
        if stop_evt.is_set(): 
            raise KeyboardInterrupt()
        conn.send(('status', status))

    try:
        url = cfg['TARGET_MODEL_URL']
        repo = url.rstrip('/').split('/')[-1]
        out_dir = Path(cfg['OUTPUT_PATH'])
        out_dir.mkdir(parents=True, exist_ok=True)

        send_log(f"▶️ Начало конвертации: {url}")
        send_status("Инициализация")
        send_progress(0)

        # Определение устройства для загрузки
        device_map = "auto"
        if cfg.get('FORCE_CPU', False):
            device_map = "cpu"
            send_log("🖥️ Принудительное использование CPU")
        elif cfg.get('DEVICE') == 'cpu':
            device_map = "cpu"
            send_log("🖥️ Использование CPU по настройкам")
        elif cfg.get('DEVICE') == 'cuda':
            device_map = "cuda"
            send_log("🎮 Принудительное использование GPU")
        else:
            send_log("🔄 Автоматический выбор устройства")

        # Конфигурация 4-битного квантования
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() and device_map != "cpu"
                else torch.float16
            )
        )

        send_status("Загрузка модели")
        send_progress(15)
        
        # Загрузка модели с учётом типа и устройства
        try:
            if cfg['MODEL_TYPE'] == 'vision':
                model = AutoModelForImageTextToText.from_pretrained(
                    url, 
                    quantization_config=bnb_config if device_map != "cpu" else None,
                    device_map=device_map, 
                    trust_remote_code=True,
                    dtype=torch.float16 if device_map == "cpu" else None  # Исправлено: dtype вместо torch_dtype
                )
            else:
                raise ValueError
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                url, 
                quantization_config=bnb_config if device_map != "cpu" else None,
                device_map=device_map, 
                trust_remote_code=True,
                dtype=torch.float16 if device_map == "cpu" else None  # Исправлено: dtype вместо torch_dtype
            )

        send_status("Загрузка токенизатора")
        send_progress(40)
        tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)

        # Загрузка процессора изображений для VL-моделей
        image_processor = None
        if cfg['MODEL_TYPE'] == 'vision':
            try:
                send_status("Загрузка процессора изображений")
                send_progress(55)
                image_processor = AutoImageProcessor.from_pretrained(
                    url, 
                    trust_remote_code=True,
                    use_fast=True
                )
            except:
                send_log("ℹ️ Процессор изображений не найден (возможно, текстовая модель)")

        # Освобождение памяти
        send_status("Очистка памяти")
        send_progress(70)
        if torch.cuda.is_available() and device_map != "cpu":
            torch.cuda.empty_cache()
        gc.collect()

        # Сохранение квантованной модели
        final_path = out_dir / f"{repo}-bnb4"
        final_path.mkdir(exist_ok=True)

        send_status("Сохранение модели")
        send_progress(85)
        model.save_pretrained(final_path, safe_serialization=cfg['SAFE_SERIALIZATION'])
        tokenizer.save_pretrained(final_path)
        if image_processor:
            image_processor.save_pretrained(final_path)

        send_log(f"✅ Модель сохранена: {final_path}")
        send_status("Завершено успешно")
        send_progress(100)
        
    except KeyboardInterrupt:
        send_log("⏹️ Операция прервана пользователем")
    except Exception as e:
        send_log(f"❌ Ошибка: {e}\n{traceback.format_exc()}")
    finally:
        conn.send(('done', None))
        conn.close()


class ConverterGUI:
    def __init__(self, root):
        self.root = root
        self._setup_window()
        self._init_variables()
        self._define_models()
        
        self.build_ui()
        self.load_settings()
        self.redirect_output()
        self.check_pipe()

    def _setup_window(self):
        """Настройка главного окна и стилей"""
        self.root.title("🔄 Конвертер AI-моделей в bnb4 формат")
        self.root.geometry("1100x800")  # Увеличено для лучшего отображения лога
        self.root.minsize(1000, 750)
        
        try:
            style = ttk.Style()
            if 'vista' in style.theme_names():
                style.theme_use('vista')
            elif 'clam' in style.theme_names():
                style.theme_use('clam')
                
            style.configure('Title.TLabel', font=('TkDefaultFont', 14, 'bold'))
            style.configure('Heading.TLabel', font=('TkDefaultFont', 10, 'bold'))
            style.configure('Info.TLabel', font=('TkDefaultFont', 8), foreground='#666666')
            style.configure('Success.TLabel', foreground='#28a745')
            style.configure('Error.TLabel', foreground='#dc3545')
            style.configure('Warning.TLabel', foreground='#fd7e14')
        except:
            pass

    def _init_variables(self):
        """Инициализация переменных для multiprocessing"""
        self.proc = None
        self.stop_evt = Event()
        self.parent_conn, self.child_conn = Pipe()

    def _define_models(self):
        """Определение каталога доступных моделей"""
        self.model_groups = {
            "🖼️ Vision-Language модели": {
                "NuMarkdown-8B-Thinking": {
                    "url": "numind/NuMarkdown-8B-Thinking",
                    "type": "vision", "params": "8B",
                    "desc": "OCR-модель с reasoning токенами для конвертации документов"
                },
                "Qwen2.5-VL-7B-Instruct": {
                    "url": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "type": "vision", "params": "7B",
                    "desc": "Многомодальная модель от Alibaba"
                },
                "MiniCPM-V-4_5": {
                    "url": "openbmb/MiniCPM-V-4_5",
                    "type": "vision", "params": "8B",
                    "desc": "Высокопроизводительная VL-модель"
                },
                "GLM-4V-9B": {
                    "url": "THUDM/glm-4v-9b", 
                    "type": "vision", "params": "9B", 
                    "desc": "Визуально-языковая модель THUDM"
                },
                "InternVL2-8B": {
                    "url": "OpenGVLab/InternVL2-8B", 
                    "type": "vision", "params": "8B",
                    "desc": "Продвинутая VL-модель OpenGVLab"
                }
            },
            "🌐 Модели перевода": {
                "Hunyuan-MT-7B": {
                    "url": "tencent/Hunyuan-MT-7B",
                    "type": "text", "params": "7B",
                    "desc": "Переводчик Tencent на 33 языка"
                },
                "Hunyuan-MT-Chimera-7B": {
                    "url": "tencent/Hunyuan-MT-Chimera-7B",
                    "type": "text", "params": "7B",
                    "desc": "Ансамблевая модель перевода"
                },
                "NLLB-200-Distilled-600M": {
                    "url": "facebook/nllb-200-distilled-600M",
                    "type": "text", "params": "600M",
                    "desc": "Многоязычный переводчик Meta"
                },
                "M2M100-12B": {
                    "url": "facebook/m2m100_12B",
                    "type": "text", "params": "12B",
                    "desc": "Переводчик на 100+ языков"
                }
            },
            "🤖 LLM модели": {
                "Qwen2.5-7B-Instruct": {
                    "url": "Qwen/Qwen2.5-7B-Instruct",
                    "type": "text", "params": "7B",
                    "desc": "Инструкционная модель Qwen"
                },
                "Qwen2.5-14B-Instruct": {
                    "url": "Qwen/Qwen2.5-14B-Instruct",
                    "type": "text", "params": "14B",
                    "desc": "Qwen 14B для сложных задач"
                },
                "Qwen2.5-32B-Instruct": {
                    "url": "Qwen/Qwen2.5-32B-Instruct",
                    "type": "text", "params": "32B",
                    "desc": "Топовая Qwen для потребительских GPU"
                },
                "Mistral-7B-Instruct-v0.3": {
                    "url": "mistralai/Mistral-7B-Instruct-v0.3",
                    "type": "text", "params": "7B",
                    "desc": "Популярная модель Mistral AI"
                },
                "Llama-3.1-8B-Instruct": {
                    "url": "meta-llama/Llama-3.1-8B-Instruct",
                    "type": "text", "params": "8B",
                    "desc": "Инструкционная модель Meta"
                }
            },
            "⚙️ Пользовательская": {
                "Своя модель": {
                    "url": "", 
                    "type": "vision", "params": "?",
                    "desc": "Введите URL модели вручную"
                }
            }
        }

    def build_ui(self):
        """Создание основного интерфейса"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=15, pady=15)

        self.main_frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(self.main_frame, text="📋 Конвертация")
        
        self.settings_frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(self.settings_frame, text="⚙️ Настройки")

        self.build_main_tab()
        self.build_settings_tab()

    def build_main_tab(self):
        """Построение основной вкладки конвертации"""
        frm = self.main_frame
        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(9, weight=1)

        # Заголовок приложения
        title_frame = ttk.Frame(frm)
        title_frame.grid(row=0, column=0, columnspan=2, pady=(0,20))
        ttk.Label(title_frame, text="🔄 Конвертер AI-моделей в bnb4 формат", 
                 style='Title.TLabel').pack()
        ttk.Label(title_frame, text="Квантование моделей для эффективного использования на потребительских GPU",
                 style='Info.TLabel').pack(pady=(5,0))

        # Секция выбора модели
        model_frame = ttk.LabelFrame(frm, text="📋 Выбор модели", padding=15)
        model_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0,15))
        model_frame.columnconfigure(1, weight=1)

        ttk.Label(model_frame, text="Категория и модель:", style='Heading.TLabel').grid(row=0, column=0, sticky="w")
        
        self.model_var = tk.StringVar()
        self.model_cb = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                    state="readonly", width=60, font=('TkDefaultFont', 9))
        self.model_cb.grid(row=0, column=1, sticky="ew", padx=(10,0))

        ttk.Label(model_frame, text="URL модели:", style='Heading.TLabel').grid(row=1, column=0, sticky="w", pady=(15,0))
        self.url_var = tk.StringVar()
        self.url_entry = ttk.Entry(model_frame, textvariable=self.url_var, font=('TkDefaultFont', 9))
        self.url_entry.grid(row=1, column=1, sticky="ew", pady=(15,0), padx=(10,0))

        # Информационная строка о выбранной модели
        self.model_info = ttk.Label(model_frame, text="", style='Info.TLabel')
        self.model_info.grid(row=2, column=0, columnspan=2, sticky="w", pady=(10,0))

        # Секция параметров конвертации
        params_frame = ttk.LabelFrame(frm, text="⚙️ Параметры конвертации", padding=15)
        params_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0,15))
        params_frame.columnconfigure(1, weight=1)
        params_frame.columnconfigure(3, weight=1)

        ttk.Label(params_frame, text="Длина контекста:", style='Heading.TLabel').grid(row=0, column=0, sticky="w")
        self.ctx_var = tk.IntVar(value=4096)
        ctx_values = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
        ctx_combo = ttk.Combobox(params_frame, textvariable=self.ctx_var, values=ctx_values, 
                                state="readonly", width=12)
        ctx_combo.grid(row=0, column=1, sticky="w")

        ttk.Label(params_frame, text="Квантование:", style='Heading.TLabel').grid(row=0, column=2, sticky="w", padx=(30,10))
        self.quant_var = tk.StringVar(value="nf4 (bnb4)")  # Исправлено: единообразное название
        quant_combo = ttk.Combobox(params_frame, textvariable=self.quant_var, 
                                  values=["nf4 (bnb4)", "fp4"], state="readonly", width=15)
        quant_combo.grid(row=0, column=3, sticky="w")

        ttk.Label(params_frame, text="Устройство:", style='Heading.TLabel').grid(row=1, column=0, sticky="w", pady=(15,0))
        self.device_var = tk.StringVar(value="gpu")  # Исправлено: по умолчанию GPU
        device_combo = ttk.Combobox(params_frame, textvariable=self.device_var, 
                                   values=["auto", "gpu", "cpu"], state="readonly", width=12)
        device_combo.grid(row=1, column=1, sticky="w", pady=(15,0))

        # Секция выбора папки сохранения
        output_frame = ttk.LabelFrame(frm, text="📁 Папка сохранения", padding=15)
        output_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0,15))
        output_frame.columnconfigure(1, weight=1)

        ttk.Label(output_frame, text="Папка:", style='Heading.TLabel').grid(row=0, column=0, sticky="w")
        self.out_var = tk.StringVar(value="./output")
        ttk.Entry(output_frame, textvariable=self.out_var).grid(row=0, column=1, sticky="ew", padx=(10,10))
        ttk.Button(output_frame, text="📁 Обзор", command=self.browse_folder).grid(row=0, column=2)

        # Секция прогресса выполнения
        progress_frame = ttk.LabelFrame(frm, text="📊 Прогресс выполнения", padding=15)
        progress_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0,15))
        progress_frame.columnconfigure(0, weight=1)

        self.progress = ttk.Progressbar(progress_frame, maximum=100, length=400)
        self.progress.grid(row=0, column=0, sticky="ew", pady=(0,10))

        self.status = ttk.Label(progress_frame, text="Готово к работе", foreground="#28a745")
        self.status.grid(row=1, column=0, sticky="w")

        # Кнопки управления процессом
        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=20)

        self.start_btn = ttk.Button(btn_frame, text="🚀 Запустить конвертацию", 
                                   command=self.start, width=25)
        self.start_btn.pack(side="left", padx=(0,10))

        self.stop_btn = ttk.Button(btn_frame, text="⏹️ Остановить", 
                                  command=self.stop, state="disabled", width=15)
        self.stop_btn.pack(side="left", padx=(0,10))

        ttk.Button(btn_frame, text="🗑️ Очистить лог", 
                   command=self.clear_log, width=20).pack(side="left")  # Исправлено: расширена кнопка

        # Журнал операций
        log_frame = ttk.LabelFrame(frm, text="📝 Журнал операций")
        log_frame.grid(row=6, column=0, columnspan=2, sticky="nsew", pady=(20,0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log = tk.Text(log_frame, state="disabled", wrap="word", height=15,  # Увеличена высота лога
                          background="#f8f9fa", font=('Consolas', 9))
        self.log.grid(row=0, column=0, sticky="nsew", padx=(5,0), pady=5)

        vs = ttk.Scrollbar(log_frame, command=self.log.yview)
        vs.grid(row=0, column=1, sticky="ns", pady=5)
        self.log.config(yscrollcommand=vs.set)

        # Инициализация списка моделей и установка значений по умолчанию
        self.populate_model_cb()
        self.model_cb.bind('<<ComboboxSelected>>', self.on_model_change)
        self._set_default_model()

    def build_settings_tab(self):
        """Построение вкладки настроек"""
        frm = self.settings_frame

        # Настройки сохранения файлов
        save_frame = ttk.LabelFrame(frm, text="💾 Настройки сохранения", padding=15)
        save_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0,20))
        save_frame.columnconfigure(1, weight=1)

        self.safe_ser_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(save_frame, text="Безопасная сериализация", 
                       variable=self.safe_ser_var).grid(row=0, column=0, sticky="w")
        ttk.Label(save_frame, text="Использовать SafeTensors формат (рекомендуется для безопасности)",
                 style='Info.TLabel').grid(row=0, column=1, sticky="w", padx=(10,0))

        # Информация о системе
        sys_frame = ttk.LabelFrame(frm, text="ℹ️ Информация о системе", padding=15)
        sys_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
                sys_info = f"🎮 GPU: {gpu_name}\n💾 Видеопамять: {gpu_memory} GB\n✅ CUDA доступна"
            else:
                sys_info = "❌ CUDA недоступна, будет использоваться только CPU"
        except:
            sys_info = "ℹ️ Информация о системе недоступна"
            
        ttk.Label(sys_frame, text=sys_info, font=('TkDefaultFont', 9)).pack(anchor="w")

        # Пояснения по устройствам обработки
        note_frame = ttk.LabelFrame(frm, text="📋 Примечание", padding=15)
        note_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(20,0))
        
        note_text = ("Выбор устройства обработки:\n"
                    "• auto - автоматический выбор (GPU при наличии)\n"
                    "• gpu - принудительное использование GPU\n" 
                    "• cpu - использование только процессора (медленно)")
        ttk.Label(note_frame, text=note_text, style='Info.TLabel').pack(anchor="w")

    def populate_model_cb(self):
        """Заполнение выпадающего списка доступными моделями"""
        values = []
        for group_name, models in self.model_groups.items():
            values.append(f"--- {group_name} ---")
            for model_name in models.keys():
                values.append(f"  {model_name}")
        
        self.model_cb['values'] = values

    def _set_default_model(self):
        """Установка модели по умолчанию при запуске"""
        values = self.model_cb['values']
        if len(values) > 1:
            self.model_cb.set(values[1])
            self.on_model_change()

    def get_model_info(self):
        """Получение информации о выбранной модели из каталога"""
        selected = self.model_var.get().strip()
        if not selected or selected.startswith("---"):
            return None
            
        model_name = selected.strip()
        for group_models in self.model_groups.values():
            if model_name in group_models:
                return group_models[model_name]
        return None

    def on_model_change(self, _=None):
        """Обработчик изменения выбранной модели"""
        info = self.get_model_info()
        if not info:
            return
            
        self.url_var.set(info['url'])
        
        if info['url']:
            self.url_entry.config(state="readonly")
        else:
            self.url_entry.config(state="normal")
            
        info_text = f"📏 {info['params']} параметров | 🏷️ {info['type']} | 📝 {info['desc']}"
        self.model_info.config(text=info_text)

    def browse_folder(self):
        """Диалог выбора папки для сохранения модели"""
        folder = filedialog.askdirectory(initialdir=self.out_var.get())
        if folder:
            self.out_var.set(folder)

    def redirect_output(self):
        """Перенаправление stdout/stderr в окно лога"""
        class LogStream:
            def __init__(self, log_func):
                self.log_func = log_func
            def write(self, message):
                if message.strip():
                    self.log_func(message)
            def flush(self):
                pass

        sys.stdout = LogStream(self.log_msg)
        sys.stderr = LogStream(self.log_msg)

    def log_msg(self, message):
        """Добавление сообщения в журнал операций"""
        self.log.config(state="normal")
        self.log.insert("end", message if message.endswith("\n") else message + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def clear_log(self):
        """Очистка журнала операций"""
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")

    def start(self):
        """Запуск процесса конвертации модели"""
        url = self.url_var.get().strip()
        if not url:
            messagebox.showerror("Ошибка", "Пожалуйста, укажите URL модели")
            return

        info = self.get_model_info() or {}
        
        device_setting = self.device_var.get()
        force_cpu = device_setting == "cpu"
        device_map = device_setting if device_setting in ["cpu", "cuda"] else "auto"
        
        config = {
            'TARGET_MODEL_URL': url,
            'OUTPUT_PATH': self.out_var.get(),
            'MODEL_TYPE': info.get('type', 'vision'),
            'SAFE_SERIALIZATION': self.safe_ser_var.get(),
            'MAX_SEQ_LENGTH': self.ctx_var.get(),
            'FORCE_CPU': force_cpu,
            'DEVICE': device_map
        }

        # Обновление интерфейса для режима выполнения
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress['value'] = 0
        self.status.config(text="Запуск конвертации...", foreground="#fd7e14")

        # Запуск фонового процесса конвертации
        self.stop_evt.clear()
        self.proc = Process(target=download_worker, args=(config, self.child_conn, self.stop_evt), daemon=True)
        self.proc.start()

    def stop(self):
        """Остановка процесса конвертации по запросу пользователя"""
        if messagebox.askyesno("Подтверждение", "Вы действительно хотите остановить конвертацию?"):
            self.stop_evt.set()
            if self.proc and self.proc.is_alive():
                self.proc.terminate()
            self._reset_interface()
            self.status.config(text="Операция прервана пользователем", foreground="#dc3545")

    def _reset_interface(self):
        """Сброс интерфейса в исходное состояние"""
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def check_pipe(self):
        """Проверка сообщений от фонового процесса"""
        try:
            while self.parent_conn.poll():
                message = self.parent_conn.recv()
                if isinstance(message, tuple) and len(message) >= 2:
                    msg_type = message[0]
                    
                    if msg_type == 'progress':
                        self.progress['value'] = message[1]
                        if len(message) > 2 and message[2]:
                            self.status.config(text=f"Выполняется: {message[2]}", foreground="#fd7e14")
                    elif msg_type == 'log':
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        self.log_msg(f"[{timestamp}] {message[1]}")
                    elif msg_type == 'status':
                        self.status.config(text=message[1], foreground="#fd7e14")
                    elif msg_type == 'done':
                        self._reset_interface()
                        self.status.config(text="✅ Конвертация завершена успешно!", foreground="#28a745")
                        messagebox.showinfo("Успех", "Модель успешно сконвертирована в bnb4 формат!")
        except:
            pass
        finally:
            self.root.after(100, self.check_pipe)

    def load_settings(self):
        """Загрузка пользовательских настроек из файла"""
        settings_file = "gui_settings.json"
        if Path(settings_file).exists():
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    
                if settings.get('model') in self.model_cb['values']:
                    self.model_cb.set(settings.get('model'))
                    self.on_model_change()
                    
                self.out_var.set(settings.get('output_path', self.out_var.get()))
                self.ctx_var.set(settings.get('context_length', self.ctx_var.get()))
                self.device_var.set(settings.get('device', 'gpu'))  # По умолчанию GPU
                self.safe_ser_var.set(settings.get('safe_serialization', True))
            except Exception as e:
                print(f"Ошибка загрузки настроек: {e}")

    def save_settings(self):
        """Сохранение пользовательских настроек в файл"""
        settings = {
            'model': self.model_cb.get(),
            'output_path': self.out_var.get(),
            'context_length': self.ctx_var.get(),
            'device': self.device_var.get(),
            'safe_serialization': self.safe_ser_var.get()
        }
        
        try:
            with open("gui_settings.json", 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения настроек: {e}")

    def on_close(self):
        """Обработчик закрытия приложения"""
        if self.proc and self.proc.is_alive():
            if not messagebox.askokcancel("Выход", "Процесс конвертации выполняется. Завершить работу?"):
                return
            self.stop_evt.set()
            self.proc.terminate()

        self.save_settings()
        self.root.destroy()


def main():
    """Точка входа приложения"""
    root = tk.Tk()
    app = ConverterGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
