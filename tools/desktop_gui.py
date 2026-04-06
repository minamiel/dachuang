import argparse
import json
import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class App(tk.Tk):
    def __init__(self, queue_json: str, history_json: str):
        super().__init__()
        self.title("Text Enhancement Control Panel")
        self.geometry("980x700")

        self.queue_json_var = tk.StringVar(value=queue_json)
        self.history_json_var = tk.StringVar(value=history_json)

        self.command_var = tk.StringVar(value="compare")
        self.input_var = tk.StringVar(value="eval_inputs")
        self.output_var = tk.StringVar(value="eval_outputs/cmp_local")
        self.preset_var = tk.StringVar(value="text-balanced")
        self.model_profile_var = tk.StringVar(value="active")
        self.extra_args_var = tk.StringVar(value="")

        self._build_ui()

    def _build_ui(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        run_frame = ttk.Frame(notebook)
        queue_frame = ttk.Frame(notebook)
        model_frame = ttk.Frame(notebook)
        notebook.add(run_frame, text="Run")
        notebook.add(queue_frame, text="Queue")
        notebook.add(model_frame, text="Model")

        self._build_run_tab(run_frame)
        self._build_queue_tab(queue_frame)
        self._build_model_tab(model_frame)

        log_frame = ttk.LabelFrame(self, text="Logs")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.log_text = tk.Text(log_frame, height=18)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _build_run_tab(self, frame: ttk.Frame):
        pad = {"padx": 6, "pady": 4}
        ttk.Label(frame, text="Command").grid(row=0, column=0, sticky="w", **pad)
        ttk.Combobox(frame, textvariable=self.command_var, values=["enhance", "compare", "batch", "full-eval"], state="readonly").grid(row=0, column=1, sticky="ew", **pad)

        ttk.Label(frame, text="Input").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.input_var).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(frame, text="Browse", command=lambda: self._pick_dir(self.input_var)).grid(row=1, column=2, **pad)

        ttk.Label(frame, text="Output").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.output_var).grid(row=2, column=1, sticky="ew", **pad)
        ttk.Button(frame, text="Browse", command=lambda: self._pick_dir(self.output_var)).grid(row=2, column=2, **pad)

        ttk.Label(frame, text="Preset").grid(row=3, column=0, sticky="w", **pad)
        ttk.Combobox(frame, textvariable=self.preset_var, values=["fast", "balanced", "best", "text-fast", "text-balanced", "text-best"], state="normal").grid(row=3, column=1, sticky="ew", **pad)

        ttk.Label(frame, text="Model Profile").grid(row=4, column=0, sticky="w", **pad)
        ttk.Combobox(frame, textvariable=self.model_profile_var, values=["active", "text-priority", "natural-priority"], state="normal").grid(row=4, column=1, sticky="ew", **pad)

        ttk.Label(frame, text="Extra Args").grid(row=5, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.extra_args_var).grid(row=5, column=1, columnspan=2, sticky="ew", **pad)

        ttk.Button(frame, text="Run Now", command=self.run_now).grid(row=6, column=1, sticky="e", **pad)
        ttk.Button(frame, text="Add To Queue", command=self.add_to_queue).grid(row=6, column=2, sticky="e", **pad)

        frame.columnconfigure(1, weight=1)

    def _build_queue_tab(self, frame: ttk.Frame):
        pad = {"padx": 6, "pady": 4}
        ttk.Label(frame, text="Queue JSON").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.queue_json_var).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(frame, text="Browse", command=lambda: self._pick_file(self.queue_json_var, save=True)).grid(row=0, column=2, **pad)

        ttk.Label(frame, text="History JSON").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.history_json_var).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(frame, text="Browse", command=lambda: self._pick_file(self.history_json_var, save=True)).grid(row=1, column=2, **pad)

        ttk.Button(frame, text="Run Queue", command=self.run_queue).grid(row=2, column=1, sticky="e", **pad)
        ttk.Button(frame, text="Open History", command=self.open_history).grid(row=2, column=2, sticky="e", **pad)

        frame.columnconfigure(1, weight=1)

    def _build_model_tab(self, frame: ttk.Frame):
        pad = {"padx": 6, "pady": 4}
        ttk.Button(frame, text="List Profiles", command=lambda: self.run_command(["model-registry", "--action", "list"])).grid(row=0, column=0, **pad)
        ttk.Button(frame, text="Status", command=lambda: self.run_command(["model-registry", "--action", "status"])).grid(row=0, column=1, **pad)
        ttk.Button(frame, text="Activate Text", command=lambda: self.run_command(["model-registry", "--action", "activate", "--model", "text-priority"])).grid(row=1, column=0, **pad)
        ttk.Button(frame, text="Activate Natural", command=lambda: self.run_command(["model-registry", "--action", "activate", "--model", "natural-priority"])).grid(row=1, column=1, **pad)

    def _pick_dir(self, var: tk.StringVar):
        d = filedialog.askdirectory(initialdir=ROOT_DIR)
        if d:
            var.set(os.path.relpath(d, ROOT_DIR))

    def _pick_file(self, var: tk.StringVar, save: bool = False):
        if save:
            p = filedialog.asksaveasfilename(initialdir=ROOT_DIR, defaultextension=".json", filetypes=[("JSON", "*.json")])
        else:
            p = filedialog.askopenfilename(initialdir=ROOT_DIR, filetypes=[("JSON", "*.json")])
        if p:
            var.set(os.path.relpath(p, ROOT_DIR))

    def _log(self, text: str):
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

    def run_command(self, argv):
        cmd = [sys.executable, os.path.join(ROOT_DIR, "run_all.py")] + argv
        self._log("$ " + " ".join(cmd))

        def worker():
            proc = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True)
            if proc.stdout:
                self._log(proc.stdout.rstrip())
            if proc.stderr:
                self._log(proc.stderr.rstrip())
            self._log(f"[exit] {proc.returncode}")

        threading.Thread(target=worker, daemon=True).start()

    def build_current_argv(self):
        argv = [
            self.command_var.get(),
            "--input_dir",
            self.input_var.get(),
            "--output_dir",
            self.output_var.get(),
            "--preset",
            self.preset_var.get(),
            "--model_profile",
            self.model_profile_var.get(),
        ]
        if self.command_var.get() == "enhance":
            argv = [
                "enhance",
                "-i",
                self.input_var.get(),
                "-o",
                self.output_var.get(),
                "--preset",
                self.preset_var.get(),
                "--model_profile",
                self.model_profile_var.get(),
            ]

        extra = self.extra_args_var.get().strip()
        if extra:
            argv.extend(extra.split())
        return argv

    def run_now(self):
        self.run_command(self.build_current_argv())

    def add_to_queue(self):
        queue_path = os.path.join(ROOT_DIR, self.queue_json_var.get())
        os.makedirs(os.path.dirname(queue_path), exist_ok=True)

        payload = {"tasks": []}
        if os.path.exists(queue_path):
            try:
                with open(queue_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict) and isinstance(loaded.get("tasks"), list):
                    payload = loaded
            except Exception:
                pass

        payload["tasks"].append({
            "name": f"gui-{self.command_var.get()}",
            "argv": self.build_current_argv(),
        })
        with open(queue_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        self._log(f"Queued task saved to: {queue_path}")

    def run_queue(self):
        self.run_command([
            "queue",
            "--queue_json",
            self.queue_json_var.get(),
            "--history_json",
            self.history_json_var.get(),
        ])

    def open_history(self):
        p = os.path.join(ROOT_DIR, self.history_json_var.get())
        if not os.path.exists(p):
            messagebox.showwarning("History", f"History file not found: {p}")
            return
        with open(p, "r", encoding="utf-8") as f:
            self._log(f.read())


def main():
    parser = argparse.ArgumentParser(description="Desktop GUI for local enhancement workflows")
    parser.add_argument("--queue_json", type=str, default="tasks.json")
    parser.add_argument("--history_json", type=str, default="queue_history.json")
    args = parser.parse_args()

    app = App(queue_json=args.queue_json, history_json=args.history_json)
    app.mainloop()


if __name__ == "__main__":
    main()
