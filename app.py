from __future__ import annotations

import argparse

import gradio as gr

from gradio_ui.layout import CUSTOM_CSS, build_demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    demo = build_demo()
    demo.queue(max_size=16)
    demo.launch(server_name=args.host, server_port=args.port,
                theme=gr.themes.Soft(), css=CUSTOM_CSS, show_error=True)


if __name__ == "__main__":
    main()
