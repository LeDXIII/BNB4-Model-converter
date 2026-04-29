# -*- coding: utf-8 -*-
"""BNB4 Model Converter v2.0 — GUI для квантования HuggingFace моделей в 4-бит."""

import sys
import json
import traceback
import warnings
import threading
from pathlib import Path
from multiprocessing import Process, Event, Pipe, set_start_method
import torch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message=r'.*Xet Storage.*')

try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass


def download_worker(cfg, conn, stop_evt):
    """Worker-процесс: загрузка модели, квантование, сохранение."""
    from transformers import (
        AutoModelForImageTextToText, AutoModelForCausalLM,
        AutoModel, AutoTokenizer, AutoImageProcessor,
        BitsAndBytesConfig, AutoConfig
    )

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

        send_log(f"▶ Начало конвертации: {url}")
        send_status("Инициализация")
        send_progress(0)

        device_map = cfg.get('DEVICE', 'auto')
        if device_map == 'cpu':
            send_log("🖥 CPU")
        elif device_map == 'cuda':
            send_log("🎮 GPU")
        else:
            device_map = "auto"
            send_log("🔄 Auto device")

        # Автоопределение типа модели
        model_type = cfg.get('MODEL_TYPE', 'auto')
        if model_type == 'auto':
            try:
                hf_config = AutoConfig.from_pretrained(url, trust_remote_code=True)
                arch = hf_config.architectures or []
                if any('Vision' in str(a) or 'ImageText' in str(a) for a in arch):
                    model_type = 'vision'
                elif any('Embedding' in str(a) for a in arch):
                    model_type = 'embedding'
                else:
                    model_type = 'text'
                send_log(f"✅ Тип: {model_type}")
            except Exception as e:
                send_log(f"⚠ Не удалось определить тип ({e}), используем text")
                model_type = 'text'

        # Конфигурация квантования
        compute_dtype = torch.float16
        if device_map != "cpu" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type=cfg.get('QUANT_TYPE', 'nf4'),
            bnb_4bit_compute_dtype=compute_dtype
        )

        send_status("Загрузка модели")
        send_progress(15)

        # 3-этапный fallback загрузки
        model = None
        errors = []

        # Попытка 1 — по типу модели
        if model_type == 'vision':
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    url, quantization_config=bnb_config if device_map != "cpu" else None,
                    device_map=device_map, trust_remote_code=True,
                    torch_dtype=compute_dtype if device_map == "cpu" else None)
                send_log("📦 AutoModelForImageTextToText ✅")
            except Exception as e:
                errors.append(str(e))
        elif model_type == 'embedding':
            try:
                model = AutoModel.from_pretrained(
                    url, quantization_config=bnb_config if device_map != "cpu" else None,
                    device_map=device_map, trust_remote_code=True,
                    torch_dtype=compute_dtype if device_map == "cpu" else None)
                send_log("📦 AutoModel (embedding) ✅")
            except Exception as e:
                errors.append(str(e))

        # Попытка 2 — CausalLM
        if model is None:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    url, quantization_config=bnb_config if device_map != "cpu" else None,
                    device_map=device_map, trust_remote_code=True,
                    torch_dtype=compute_dtype if device_map == "cpu" else None)
                send_log("📦 AutoModelForCausalLM ✅")
            except Exception as e:
                errors.append(str(e))

        # Попытка 3 — универсальный AutoModel
        if model is None:
            try:
                model = AutoModel.from_pretrained(
                    url, quantization_config=bnb_config if device_map != "cpu" else None,
                    device_map=device_map, trust_remote_code=True,
                    torch_dtype=compute_dtype if device_map == "cpu" else None)
                send_log("📦 AutoModel ✅")
            except Exception as e:
                errors.append(str(e))

        if model is None:
            raise RuntimeError("Не удалось загрузить модель.\n" + "\n".join(f"- {e}" for e in errors))

        # Токенизатор
        send_status("Токенизатор")
        send_progress(40)
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
        except Exception as e:
            send_log(f"⚠ Токенизатор не найден: {e}")

        # Процессор изображений для VL
        image_processor = None
        if model_type == 'vision':
            try:
                send_status("Процессор изображений")
                send_progress(55)
                image_processor = AutoImageProcessor.from_pretrained(url, trust_remote_code=True)
            except Exception:
                pass

        # Очистка памяти
        send_status("Очистка памяти")
        send_progress(70)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Сохранение
        final_path = out_dir / f"{repo}-bnb4"
        final_path.mkdir(exist_ok=True)
        send_status("Сохранение")
        send_progress(85)

        model.save_pretrained(final_path, safe_serialization=cfg.get('SAFE_SERIALIZATION', True))
        send_log(f"💾 {final_path}")

        if tokenizer:
            tokenizer.save_pretrained(final_path)
        if image_processor:
            image_processor.save_pretrained(final_path)

        send_log("✅ Модель успешно сохранена")
        send_status("Готово")
        send_progress(100)

    except KeyboardInterrupt:
        send_log("⏹ Прервано пользователем")
    except Exception as e:
        send_log(f"❌ {e}\n{traceback.format_exc()}")
    finally:
        conn.send(('done', None))
        conn.close()


class ConverterGUI:
    def __init__(self, root):
        self.root = root
        self._setup_window()
        self._define_models()
        self.proc = None
        self.stop_evt = Event()
        self.parent_conn, self.child_conn = Pipe()
        self.vram_running = False
        self.build_ui()
        self.load_settings()
        self.redirect_output()
        self.check_pipe()
        self.start_vram_monitor()

    def _setup_window(self):
        self.root.title("BNB4 Converter v2.0")
        self.root.geometry("900x800")
        self.root.minsize(860, 700)
        self.colors = {'success': '#28a745', 'warning': '#ff9800', 'info': '#2196F3', 'error': '#dc3545'}
        style = ttk.Style()
        try:
            if 'vista' in style.theme_names():
                style.theme_use('vista')
            style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'))
            style.configure('Heading.TLabel', font=('Segoe UI', 10, 'bold'))
            style.configure('Info.TLabel', font=('Segoe UI', 9), foreground='#666')
        except Exception:
            pass

    def _define_models(self):
        self.model_groups = {
            "🔥 Qwen 3.5": {
                "Qwen3.5-0.8B": {"url": "Qwen/Qwen3.5-0.8B", "type": "auto", "params": "0.8B", "desc": "Ультра-лёгкая Qwen 3.5"},
                "Qwen3.5-2B": {"url": "Qwen/Qwen3.5-2B", "type": "auto", "params": "2B", "desc": "Лёгкая Qwen 3.5"},
                "Qwen3.5-4B": {"url": "Qwen/Qwen3.5-4B", "type": "auto", "params": "4B", "desc": "Средняя Qwen 3.5"},
                "Qwen3.5-9B": {"url": "Qwen/Qwen3.5-9B", "type": "auto", "params": "9B", "desc": "Полноразмерная Qwen 3.5"},
            },
            "🤖 Qwen 3": {
                "Qwen3-0.6B": {"url": "Qwen/Qwen3-0.6B", "type": "auto", "params": "0.6B", "desc": "Ультра-лёгкая Qwen 3"},
                "Qwen3-1.7B": {"url": "Qwen/Qwen3-1.7B", "type": "auto", "params": "1.7B", "desc": "Лёгкая Qwen 3"},
                "Qwen3-4B": {"url": "Qwen/Qwen3-4B", "type": "auto", "params": "4B", "desc": "Средняя Qwen 3"},
                "Qwen3-8B": {"url": "Qwen/Qwen3-8B", "type": "auto", "params": "8B", "desc": "Полноразмерная Qwen 3"},
                "Qwen3-Emb-0.6B": {"url": "Qwen/Qwen3-Embedding-0.6B", "type": "embedding", "params": "0.6B", "desc": "Embedding Qwen 3"},
                "Qwen3-Emb-4B": {"url": "Qwen/Qwen3-Embedding-4B", "type": "embedding", "params": "4B", "desc": "Embedding Qwen 3"},
            },
            "🖼️ Vision-Language": {
                "MinerU2.5-Pro-1.2B": {"url": "opendatalab/MinerU2.5-Pro-2604-1.2B", "type": "vision", "params": "1.2B", "desc": "OCR от opendatalab"},
                "Qwen2.5-VL-7B": {"url": "Qwen/Qwen2.5-VL-7B-Instruct", "type": "vision", "params": "7B", "desc": "VL Qwen"},
                "Qwen2.5-VL-3B": {"url": "Qwen/Qwen2.5-VL-3B-Instruct", "type": "vision", "params": "3B", "desc": "Лёгкая VL Qwen"},
                "MiniCPM-V-4.5": {"url": "openbmb/MiniCPM-V-4_5", "type": "vision", "params": "8B", "desc": "VL MiniCPM"},
                "GLM-4V-9B": {"url": "THUDM/glm-4v-9b", "type": "vision", "params": "9B", "desc": "VL THUDM"},
                "InternVL2-8B": {"url": "OpenGVLab/InternVL2-8B", "type": "vision", "params": "8B", "desc": "VL OpenGVLab"},
                "NuMarkdown-8B": {"url": "numind/NuMarkdown-8B-Thinking", "type": "vision", "params": "8B", "desc": "OCR для документов"},
                "DeepSeek-OCR-2": {"url": "deepseek-ai/DeepSeek-OCR-2", "type": "vision", "params": "3B", "desc": "OCR DeepSeek"},
            },
            "🌐 Перевод": {
                "Hunyuan-MT-7B": {"url": "tencent/Hunyuan-MT-7B", "type": "text", "params": "7B", "desc": "Переводчик Tencent"},
                "Hunyuan-MT-Chimera": {"url": "tencent/Hunyuan-MT-Chimera-7B", "type": "text", "params": "7B", "desc": "Ансамбль перевод"},
                "NLLB-600M": {"url": "facebook/nllb-200-distilled-600M", "type": "text", "params": "600M", "desc": "Meta NLLB лёгкий"},
                "M2M100-12B": {"url": "facebook/m2m100_12B", "type": "text", "params": "12B", "desc": "Meta M2M100"},
            },
            "🧠 LLM": {
                "Qwen2.5-0.5B": {"url": "Qwen/Qwen2.5-0.5B-Instruct", "type": "auto", "params": "0.5B", "desc": "Ультра-лёгкая Qwen"},
                "Qwen2.5-1.5B": {"url": "Qwen/Qwen2.5-1.5B-Instruct", "type": "auto", "params": "1.5B", "desc": "Лёгкая Qwen"},
                "Qwen2.5-3B": {"url": "Qwen/Qwen2.5-3B-Instruct", "type": "auto", "params": "3B", "desc": "Средняя Qwen"},
                "Qwen2.5-7B": {"url": "Qwen/Qwen2.5-7B-Instruct", "type": "auto", "params": "7B", "desc": "Qwen Instruct"},
                "Qwen2.5-14B": {"url": "Qwen/Qwen2.5-14B-Instruct", "type": "auto", "params": "14B", "desc": "Qwen 14B"},
                "Mistral-7B-v0.3": {"url": "mistralai/Mistral-7B-Instruct-v0.3", "type": "auto", "params": "7B", "desc": "Mistral AI"},
                "Llama-3.2-1B": {"url": "meta-llama/Llama-3.2-1B-Instruct", "type": "auto", "params": "1B", "desc": "Llama лёгкая"},
                "Llama-3.2-3B": {"url": "meta-llama/Llama-3.2-3B-Instruct", "type": "auto", "params": "3B", "desc": "Llama средняя"},
                "Llama-3.1-8B": {"url": "meta-llama/Llama-3.1-8B-Instruct", "type": "auto", "params": "8B", "desc": "Llama 8B"},
                "Gemma-2-2B": {"url": "google/gemma-2-2b", "type": "auto", "params": "2B", "desc": "Gemma 2 лёгкая"},
                "Gemma-2-9B": {"url": "google/gemma-2-9b", "type": "auto", "params": "9B", "desc": "Gemma 2"},
                "Phi-3.5-mini": {"url": "microsoft/Phi-3.5-mini-instruct", "type": "auto", "params": "3.8B", "desc": "Phi-3.5 Microsoft"},
            },
            "⚙️ Своя модель": {
                "Своя модель": {"url": "", "type": "auto", "params": "?", "desc": "Введите URL вручную"},
            },
        }

    def build_ui(self):
        ttk.Label(self.root, text="BNB4 Конвертер моделей v2.0", style='Title.TLabel').pack(anchor="w", padx=15, pady=(15, 4))
        ttk.Label(self.root, text="Квантование моделей в 4-бит для потребительских GPU", style='Info.TLabel').pack(anchor="w", padx=15, pady=(0, 10))

        model_frame = ttk.LabelFrame(self.root, text=" Выбор модели", padding=10)
        model_frame.pack(fill="x", padx=15, pady=(0, 6))
        model_frame.columnconfigure(1, weight=1)
        ttk.Label(model_frame, text="Модель:", style='Heading.TLabel').grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar()
        self.model_cb = ttk.Combobox(model_frame, textvariable=self.model_var, state="readonly")
        self.model_cb.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Label(model_frame, text="URL:", style='Heading.TLabel').grid(row=1, column=0, sticky="w", pady=(6, 0))
        url_row = ttk.Frame(model_frame)
        url_row.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        url_row.columnconfigure(0, weight=1)
        self.url_var = tk.StringVar()
        self.url_entry = tk.Entry(url_row, textvariable=self.url_var, font=('Consolas', 9), bg="#ffffff", bd=1, relief="solid")
        self.url_entry.grid(row=0, column=0, sticky="ew", ipady=3)
        ttk.Button(url_row, text="Вставить", command=lambda: self._paste_entry(self.url_entry), width=8).grid(row=0, column=1, padx=(6, 0))
        self.model_info = ttk.Label(model_frame, text="", style='Info.TLabel')
        self.model_info.grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))

        params_frame = ttk.LabelFrame(self.root, text=" Параметры конвертации", padding=10)
        params_frame.pack(fill="x", padx=15, pady=(0, 6))
        params_frame.columnconfigure(1, weight=1)
        params_frame.columnconfigure(3, weight=1)
        params_frame.columnconfigure(5, weight=1)
        ttk.Label(params_frame, text="Контекст:").grid(row=0, column=0, sticky="w")
        self.ctx_var = tk.IntVar(value=4096)
        ttk.Combobox(params_frame, textvariable=self.ctx_var, values=[512,1024,2048,4096,8192,16384,32768,65536,131072,262144], state="readonly", width=12).grid(row=0, column=1, sticky="w")
        ttk.Label(params_frame, text="Квантование:").grid(row=0, column=2, sticky="w", padx=(16, 6))
        self.quant_var = tk.StringVar(value="nf4")
        ttk.Combobox(params_frame, textvariable=self.quant_var, values=["nf4","fp4"], state="readonly", width=10).grid(row=0, column=3, sticky="w")
        ttk.Label(params_frame, text="Устройство:").grid(row=0, column=4, sticky="w", padx=(16, 6))
        self.device_var = tk.StringVar(value="auto")
        ttk.Combobox(params_frame, textvariable=self.device_var, values=["auto","gpu","cpu"], state="readonly", width=10).grid(row=0, column=5, sticky="w")

        output_frame = ttk.LabelFrame(self.root, text=" Папка сохранения", padding=10)
        output_frame.pack(fill="x", padx=15, pady=(0, 6))
        out_row = ttk.Frame(output_frame)
        out_row.pack(fill="x")
        out_row.columnconfigure(0, weight=1)
        self.out_var = tk.StringVar(value="./output")
        self.out_entry = tk.Entry(out_row, textvariable=self.out_var, font=('Consolas', 9), bg="#ffffff", bd=1, relief="solid")
        self.out_entry.grid(row=0, column=0, sticky="ew", ipady=3)
        ttk.Button(out_row, text="Обзор", command=self.browse_folder).grid(row=0, column=1, padx=(6, 0))

        progress_frame = ttk.LabelFrame(self.root, text=" Прогресс", padding=10)
        progress_frame.pack(fill="x", padx=15, pady=(0, 6))
        self.progress = ttk.Progressbar(progress_frame, maximum=100)
        self.progress.pack(fill="x", pady=(0, 4))
        status_row = ttk.Frame(progress_frame)
        status_row.pack(fill="x")
        self.status = ttk.Label(status_row, text="Готово к работе", foreground=self.colors['success'])
        self.status.pack(side="left")
        self.vram_label = ttk.Label(status_row, text="", foreground=self.colors['info'])
        self.vram_label.pack(side="right")

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=8)
        self.start_btn = ttk.Button(btn_frame, text=" Запустить конвертацию", command=self.start)
        self.start_btn.pack(side="left", padx=(0, 6))
        self.stop_btn = ttk.Button(btn_frame, text=" Остановить", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 6))
        ttk.Button(btn_frame, text=" Очистить лог", command=self.clear_log).pack(side="left")

        log_frame = ttk.LabelFrame(self.root, text=" Журнал операций", padding=6)
        log_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        ttk.Button(log_frame, text=" Копировать лог", command=self._copy_log).pack(anchor="w", pady=(0, 4))
        self.log = tk.Text(log_frame, wrap="word", background="#1e1e1e", foreground="#d4d4d4", font=('Consolas', 9), selectbackground="#264f78")
        self.log.pack(fill="both", expand=True)
        self.log.bind('<Key>', lambda e: 'break')

        self.populate_model_cb()
        self.model_cb.bind('<<ComboboxSelected>>', self.on_model_change)
        self._set_default_model()

    def populate_model_cb(self):
        values = []
        for group_name, models in self.model_groups.items():
            values.append(f"── {group_name} ──")
            values.extend(f"  {m}" for m in models.keys())
        self.model_cb['values'] = values

    def _set_default_model(self):
        values = self.model_cb['values']
        if len(values) > 1:
            self.model_cb.set(values[1])
            self.on_model_change()

    def get_model_info(self):
        selected = self.model_var.get().strip()
        if not selected or selected.startswith("──"):
            return None
        for group_models in self.model_groups.values():
            if selected in group_models:
                return group_models[selected]
        return None

    def on_model_change(self, _=None):
        info = self.get_model_info()
        if not info:
            return
        self.url_var.set(info['url'])
        type_labels = {'vision': '🖼 Vision', 'text': '🧠 Text', 'embedding': '📊 Embedding', 'auto': '🔄 Auto'}
        self.model_info.config(text=f"{info['params']} | {type_labels.get(info['type'], 'Auto')} | {info['desc']}")

    def browse_folder(self):
        folder = filedialog.askdirectory(initialdir=self.out_var.get())
        if folder:
            self.out_var.set(folder)

    def check_model_availability(self):
        url = self.url_var.get().strip()
        if not url:
            messagebox.showwarning("Предупреждение", "Введите URL")
            return

        def _check():
            try:
                from huggingface_hub import HfApi
                info = HfApi().model_info(url)
                self.root.after(0, lambda: messagebox.showinfo("Успех", f"Модель доступна\nЗагрузок: {getattr(info,'downloads',0):,}"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showwarning("Предупреждение", str(e)))
            finally:
                self.root.after(0, lambda: self.check_btn.config(state="normal", text="Проверить"))

        self.check_btn.config(state="disabled", text="⏳")
        threading.Thread(target=_check, daemon=True).start()

    def redirect_output(self):
        class LogStream:
            def __init__(self, f): self.f = f
            def write(self, m):
                if m.strip(): self.f(m)
            def flush(self): pass
        sys.stdout = LogStream(self.log_msg)
        sys.stderr = LogStream(self.log_msg)

    def log_msg(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg if msg.endswith("\n") else msg + "\n")
        self.log.see("end")

    def clear_log(self):
        self.log.delete("1.0", "end")

    def _copy_log(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.log.get("1.0", "end-1c"))
        self.root.update()

    def _paste_entry(self, entry):
        try:
            entry.delete(0, "end")
            entry.insert(0, self.root.clipboard_get().strip())
        except Exception:
            pass

    def start_vram_monitor(self):
        self.vram_running = True
        self._update_vram()

    def _update_vram(self):
        try:
            import subprocess
            r = subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=2)
            if r.returncode == 0:
                used, total = map(float, r.stdout.strip().split(','))
                self.vram_label.config(text=f"VRAM: {used/1024:.1f}/{total/1024:.1f} GB ({used/total*100:.0f}%)")
            else:
                self.vram_label.config(text="VRAM: N/A")
        except Exception:
            self.vram_label.config(text="VRAM: N/A")
        if self.vram_running:
            self.root.after(2000, self._update_vram)

    def start(self):
        url = self.url_var.get().strip()
        if not url:
            messagebox.showerror("Ошибка", "Укажите URL модели")
            return
        info = self.get_model_info() or {}
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress['value'] = 0
        self.status.config(text="Запуск...", foreground=self.colors['warning'])
        self.stop_evt.clear()
        cfg = {
            'TARGET_MODEL_URL': url,
            'OUTPUT_PATH': self.out_var.get(),
            'MODEL_TYPE': info.get('type', 'auto'),
            'DEVICE': 'auto' if self.device_var.get() == 'auto' else self.device_var.get(),
            'QUANT_TYPE': self.quant_var.get(),
            'SAFE_SERIALIZATION': True,
            'MAX_SEQ_LENGTH': self.ctx_var.get(),
        }
        self.proc = Process(target=download_worker, args=(cfg, self.child_conn, self.stop_evt), daemon=True)
        self.proc.start()
        self.log_msg(f" Запуск: {url}\n")

    def stop(self):
        if messagebox.askyesno("Подтверждение", "Остановить конвертацию?"):
            self.stop_evt.set()
            if self.proc and self.proc.is_alive():
                self.proc.terminate()
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.status.config(text="Прервано", foreground=self.colors['error'])
            self.log_msg("⏹ Прервано\n")

    def check_pipe(self):
        try:
            while self.parent_conn.poll():
                msg = self.parent_conn.recv()
                if isinstance(msg, tuple):
                    t = msg[0]
                    if t == 'progress':
                        self.progress['value'] = msg[1]
                        if len(msg) > 2 and msg[2]:
                            self.status.config(text=msg[2], foreground=self.colors['warning'])
                    elif t == 'log':
                        self.log_msg(msg[1])
                    elif t == 'status':
                        self.status.config(text=msg[1], foreground=self.colors['warning'])
                    elif t == 'done':
                        self.start_btn.config(state="normal")
                        self.stop_btn.config(state="disabled")
                        self.status.config(text="✅ Готово!", foreground=self.colors['success'])
                        messagebox.showinfo("Успех", "Модель сконвертирована!")
        except Exception:
            pass
        self.root.after(100, self.check_pipe)

    def load_settings(self):
        f = "gui_settings.json"
        if Path(f).exists():
            try:
                s = json.loads(Path(f).read_text(encoding='utf-8'))
                if s.get('model') in self.model_cb['values']:
                    self.model_cb.set(s['model'])
                    self.on_model_change()
                self.out_var.set(s.get('output_path', self.out_var.get()))
                self.ctx_var.set(s.get('context_length', self.ctx_var.get()))
                self.device_var.set(s.get('device', 'auto'))
                self.quant_var.set(s.get('quant_type', 'nf4'))
            except Exception as e:
                print(f"Load settings error: {e}")

    def save_settings(self):
        s = {'model': self.model_cb.get(), 'output_path': self.out_var.get(),
             'context_length': self.ctx_var.get(), 'device': self.device_var.get(),
             'quant_type': self.quant_var.get()}
        try:
            Path("gui_settings.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            print(f"Save settings error: {e}")

    def on_close(self):
        self.vram_running = False
        if self.proc and self.proc.is_alive():
            if not messagebox.askokcancel("Выход", "Конвертация выполняется. Выйти?"):
                return
            self.stop_evt.set()
            self.proc.terminate()
        self.save_settings()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ConverterGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
