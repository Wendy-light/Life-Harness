"""Portable SGLang launcher; use a separate SGLang Python environment."""
import argparse
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BOOTSTRAP = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location('local_xlam', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from sglang.launch_server import run_server
from sglang.srt.server_args import prepare_server_args
run_server(prepare_server_args(sys.argv[2:]))
"""


def main(default_parser=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--served-model-name")
    p.add_argument("--tool-call-parser", default=default_parser, required=default_parser is None)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--dp-size", type=int, default=1)
    p.add_argument("--mem-fraction-static", type=float, default=0.9)
    p.add_argument("--port", type=int, default=30001)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--context-length", type=int)
    p.add_argument("--attention-backend", default="triton")
    p.add_argument("--llama-history-template", action="store_true",
                   help="Use the Llama 3.3 parallel-history compatibility template; not for xLAM/Qwen")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.llama_history_template and a.tool_call_parser != "llama3":
        p.error("--llama-history-template requires --tool-call-parser llama3")
    if a.tool_call_parser == "xlam":
        cmd = [sys.executable, "-c", BOOTSTRAP, str(HERE / "xlam_tool_call_parser.py")]
    else:
        cmd = [sys.executable, "-m", "sglang.launch_server"]
    cmd += ["--model-path", a.model_path, "--host", a.host, "--port", str(a.port),
            "--tp", str(a.tp), "--dp-size", str(a.dp_size), "--mem-fraction-static", str(a.mem_fraction_static),
            "--tool-call-parser", a.tool_call_parser, "--attention-backend", a.attention_backend,
            "--random-seed", "100", "--trust-remote-code"]
    if a.served_model_name:
        cmd += ["--served-model-name", a.served_model_name]
    if a.context_length:
        cmd += ["--context-length", str(a.context_length)]
    if a.llama_history_template:
        cmd += ["--chat-template", str(HERE / "llama_appworld_history.jinja")]
    if a.dry_run:
        import shlex
        print(shlex.join(cmd))
        return
    env = dict(os.environ)
    for key in ["NO_PROXY", "no_proxy"]:
        env[key] = ",".join(filter(None, [env.get(key), "localhost,127.0.0.1,::1"]))
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    # Replace this launcher so signals and service lifetime remain unambiguous.
    os.execve(sys.executable, cmd, env)


if __name__ == "__main__":
    main()
