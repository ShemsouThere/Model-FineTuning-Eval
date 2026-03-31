from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


DEFAULT_FINE_TUNED_PATH = Path(
    "/workspace/Model-FineTuning/artifacts/spider1_qwen25_7b/gguf/model-q4_k_m.gguf"
)
DEFAULT_BASE_REPO = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare models.json for base-vs-fine-tuned GGUF evaluation on the server."
    )
    parser.add_argument(
        "--fine-tuned-gguf-path",
        type=Path,
        default=DEFAULT_FINE_TUNED_PATH,
        help="Path to the fine-tuned GGUF model on the server.",
    )
    parser.add_argument(
        "--fine-tuned-name",
        default="qwen25-coder-7b-finetuned-q4_k_m",
    )
    parser.add_argument(
        "--base-repo",
        default=DEFAULT_BASE_REPO,
        help="Hugging Face GGUF repo for the base model.",
    )
    parser.add_argument(
        "--base-file",
        default=None,
        help="Specific GGUF filename to download. If omitted, auto-detect a Q4_K_M file.",
    )
    parser.add_argument(
        "--base-download-dir",
        type=Path,
        default=Path("models/base"),
    )
    parser.add_argument(
        "--base-name",
        default="qwen25-coder-7b-base-q4_k_m",
    )
    parser.add_argument(
        "--models-config-out",
        type=Path,
        default=Path("models.server.json"),
    )
    return parser


def resolve_base_file(repo_id: str, explicit_filename: str | None) -> str:
    if explicit_filename:
        return explicit_filename

    files = list_repo_files(repo_id)
    candidates = [
        file_name
        for file_name in files
        if file_name.lower().endswith(".gguf") and "q4_k_m" in file_name.lower()
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a Q4_K_M GGUF file in {repo_id}. "
            "Pass --base-file explicitly."
        )
    raise RuntimeError(
        f"Found multiple Q4_K_M GGUF candidates in {repo_id}: {candidates}. "
        "Pass --base-file explicitly."
    )


def main() -> int:
    args = build_parser().parse_args()

    fine_tuned_path = args.fine_tuned_gguf_path.expanduser().resolve()
    if not fine_tuned_path.exists():
        raise FileNotFoundError(
            f"Fine-tuned GGUF path does not exist: {fine_tuned_path}"
        )

    base_download_dir = args.base_download_dir.expanduser().resolve()
    base_download_dir.mkdir(parents=True, exist_ok=True)

    base_file = resolve_base_file(args.base_repo, args.base_file)
    downloaded_base = Path(
        hf_hub_download(
            repo_id=args.base_repo,
            filename=base_file,
            local_dir=str(base_download_dir),
            local_dir_use_symlinks=False,
        )
    ).resolve()

    config_out = args.models_config_out.expanduser().resolve()
    config_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "models": [
            {
                "name": args.base_name,
                "path": str(downloaded_base),
            },
            {
                "name": args.fine_tuned_name,
                "path": str(fine_tuned_path),
            },
        ]
    }
    config_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Base model: {downloaded_base}")
    print(f"Fine-tuned model: {fine_tuned_path}")
    print(f"Wrote models config: {config_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
