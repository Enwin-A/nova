"""Gradio chat UI for the Novakid Review Chat Agent."""

import logging
import sys
import traceback


def _patch_gradio_client_schema_parser() -> None:
    """Fix gradio-client 1.3.x crash on bool additionalProperties in JSON schema."""
    try:
        import gradio_client.utils as client_utils
    except ImportError:
        return

    if getattr(client_utils, "_nova_schema_patch", False):
        return

    original = client_utils._json_schema_to_python_type

    def patched(schema, defs):
        if isinstance(schema, bool):
            return "Any" if schema else "Never"
        return original(schema, defs)

    client_utils._json_schema_to_python_type = patched
    client_utils._nova_schema_patch = True


_patch_gradio_client_schema_parser()

import gradio as gr

import config
from agent.agent import ChatResponse, ReviewAgent

logger = logging.getLogger(__name__)

EXAMPLE_QUERIES = [
    "What are users complaining about most in 1-star reviews?",
    "Show me a review where someone praised the teachers",
    "How do Turkish parents and Spanish parents differ on teacher feedback?",
]

MONO_CSS = """
/* ── GLOBAL RESET ── */
*, *::before, *::after {
    border-radius: 0 !important;
    font-family: monospace !important;
    box-shadow: none !important;
}

/* ── BASE ── */
html, body,
.gradio-container,
.gradio-container > .main,
.contain,
.gap,
.form,
.panel,
.block,
.prose,
.wrap,
.scroll-hide,
.overflow-y-auto,
.svelte-po1wcb,
div[data-testid="chatbot"],
div[data-testid="chatbot"] > div {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* ── CHATBOT OUTER SHELL ── */
div[data-testid="chatbot"],
div[data-testid="chatbot"] > div,
div[data-testid="chatbot"] > div > div {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border: 3px solid #000000 !important;
}

/* ── CHATBOT SCROLL AREA ── */
div[data-testid="chatbot"] .overflow-y-auto,
div[data-testid="chatbot"] .scroll-hide,
div[data-testid="chatbot"] .min-h-\[40vh\],
div[data-testid="chatbot"] .min-h-full,
div[data-testid="chatbot"] [class*="bubble"],
div[data-testid="chatbot"] [class*="message"],
div[data-testid="chatbot"] [class*="wrap"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
}

/* ── MESSAGE BUBBLES — every known variant ── */
.message,
.message-bubble-border,
.user,
.bot,
.human,
.assistant,
[class*="message"],
[class*="bubble"],
div[data-testid="user"],
div[data-testid="bot"],
div[data-testid="human"],
div[data-testid="assistant"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 2px solid #000000 !important;
}

/* Kill any ::before / ::after pseudo that Gradio uses for speech-bubble triangles */
.message::before, .message::after,
[class*="bubble"]::before, [class*="bubble"]::after,
div[data-testid="user"]::before, div[data-testid="user"]::after,
div[data-testid="bot"]::before, div[data-testid="bot"]::after {
    display: none !important;
}

/* ── ALL TEXT INSIDE CHAT ── */
div[data-testid="chatbot"] *,
.message *,
[class*="bubble"] *,
[class*="message"] * {
    color: #000000 !important;
    background-color: transparent !important;
}

/* ── PROCESSING / TIMER BAR ── */
/* Gradio renders the "processing" + elapsed time in these elements */
.eta-bar,
.eta-bar *,
.generating,
.generating *,
.load-status,
.load-status *,
[class*="progress"],
[class*="progress"] *,
[class*="pending"],
[class*="pending"] *,
[class*="eta"],
[class*="eta"] *,
[class*="timer"],
[class*="timer"] *,
[class*="status-tracker"],
[class*="status-tracker"] * {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #000000 !important;
    border-color: #000000 !important;
}

/* ── TEXTBOX ── */
textarea, input[type="text"], input[type="search"] {
    background: #ffffff !important;
    color: #000000 !important;
    border: 2px solid #000000 !important;
    caret-color: #000000 !important;
}

/* ── BUTTONS ── */
button[variant="primary"], .primary, button.primary,
.gr-button-primary {
    background: #000000 !important;
    color: #ffffff !important;
    border: 2px solid #000000 !important;
}
button[variant="primary"]:hover {
    background: #333333 !important;
}

button, .gr-button, .secondary, button.secondary {
    background: #ffffff !important;
    color: #000000 !important;
    border: 2px solid #000000 !important;
}
button:hover {
    background: #f0f0f0 !important;
}

/* ── HEADINGS / LABELS ── */
h1, h2, h3, h4, h5, h6, p, label, span, small, .markdown, .prose, .label {
    color: #000000 !important;
}

/* ── SIDEBAR / COLUMN BACKGROUNDS ── */
.column, .col, [class*="column"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar-track { background: #ffffff !important; }
::-webkit-scrollbar-thumb { background: #000000 !important; }

/* ── FOOTER ── */
footer, .footer { display: none !important; }
"""

FORCE_WHITE_JS = """
<script>
(function() {
    function forceWhite() {
        /* chatbot area + message bubbles */
        var bgSelectors = [
            '[data-testid="chatbot"]',
            '[data-testid="chatbot"] > *',
            '[data-testid="chatbot"] > * > *',
            '[data-testid="bot"]',
            '[data-testid="user"]',
            '.message', '.bubble', '.user', '.bot',
            '.overflow-y-auto', '.scroll-hide'
        ];
        bgSelectors.forEach(function(sel) {
            document.querySelectorAll(sel).forEach(function(el) {
                el.style.setProperty('background', '#ffffff', 'important');
                el.style.setProperty('background-color', '#ffffff', 'important');
                el.style.setProperty('color', '#000000', 'important');
            });
        });

        /* borders on actual bubbles only */
        ['[data-testid="bot"]', '[data-testid="user"]', '.message'].forEach(function(sel) {
            document.querySelectorAll(sel).forEach(function(el) {
                el.style.setProperty('border', '2px solid #000000', 'important');
            });
        });

        /* processing / timer bar — white bg, black text */
        var statusSelectors = [
            '.eta-bar', '.generating', '.load-status',
            '[class*="progress"]', '[class*="pending"]',
            '[class*="eta"]', '[class*="timer"]',
            '[class*="status-tracker"]', '[class*="status"]'
        ];
        statusSelectors.forEach(function(sel) {
            document.querySelectorAll(sel).forEach(function(el) {
                el.style.setProperty('background', '#ffffff', 'important');
                el.style.setProperty('background-color', '#ffffff', 'important');
                el.style.setProperty('color', '#000000', 'important');
            });
        });
    }

    document.addEventListener('DOMContentLoaded', forceWhite);
    setInterval(forceWhite, 800);
    var obs = new MutationObserver(forceWhite);
    obs.observe(document.body, { childList: true, subtree: true });
})();
</script>
"""

KEYBINDINGS_JS = """
<script>
(function() {
    function attachKeybindings() {
        var textareas = document.querySelectorAll('textarea');
        textareas.forEach(function(ta) {
            if (ta._novaKeybind) return;
            ta._novaKeybind = true;
            ta.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    e.stopPropagation();
                    var container = ta.closest('.gradio-container') || document;
                    var btn = container.querySelector('button.primary')
                        || container.querySelector('button[variant="primary"]')
                        || Array.from(container.querySelectorAll('button')).find(function(b) {
                            return b.textContent.trim().toLowerCase() === 'send';
                        });
                    if (btn) btn.click();
                }
            });
        });
    }
    document.addEventListener('DOMContentLoaded', attachKeybindings);
    setInterval(attachKeybindings, 1000);
    var obs = new MutationObserver(attachKeybindings);
    document.addEventListener('DOMContentLoaded', function() {
        obs.observe(document.body, { childList: true, subtree: true });
    });
})();
</script>
"""

_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = ReviewAgent()
    return _agent


def _reset_agent():
    global _agent
    _agent = None


def format_bot_message(response: ChatResponse) -> str:
    parts = [response.answer]

    meta = []
    if response.mode == "comparison":
        meta.append("cross-country comparison")
    if response.tools_used:
        meta.append(" + ".join(response.tools_used))
    if meta:
        parts.append(f"\n\n**{' · '.join(meta)}**")

    if response.citations:
        parts.append("\n\n**Citations:**")
        for citation in response.citations:
            parts.append(
                f"- [{citation.review_id}] (★{citation.rating}) {citation.language} | "
                f"'{citation.snippet}'"
            )

    return "\n".join(parts)


def respond(message, history):
    if not message or not str(message).strip():
        return history or []

    agent = _get_agent()
    chat_history = []
    for turn in history or []:
        user_text = turn[0] or ""
        bot_text = turn[1] or ""
        if user_text and bot_text:
            clean_bot = bot_text.split("\n\n**")[0]
            chat_history.append((user_text, clean_bot))

    try:
        response = agent.chat(str(message).strip(), chat_history)
        bot_text = format_bot_message(response)
        return (history or []) + [[str(message).strip(), bot_text]]
    except Exception:
        logger.exception("chat_error")
        traceback.print_exc()
        error_message = "Something went wrong. Details logged to traces/agent_trace.jsonl."
        return (history or []) + [[str(message).strip(), error_message]]


def on_clear():
    _reset_agent()
    return [], ""


def build_app():
    with gr.Blocks(
        title="Novakid Review Chat Agent",
        theme=gr.themes.Default(
            primary_hue=gr.themes.colors.neutral,
            secondary_hue=gr.themes.colors.neutral,
            neutral_hue=gr.themes.colors.neutral,
        ),
        css=MONO_CSS,
        head=FORCE_WHITE_JS + KEYBINDINGS_JS,
    ) as demo:
        gr.Markdown("# Novakid Review Chat Agent")

        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### Example queries")
                example_buttons = []
                for example in EXAMPLE_QUERIES:
                    btn = gr.Button(example, size="sm")
                    example_buttons.append((btn, example))

                # gr.HTML guarantees inline style wins — no theme can override it
                gr.HTML(
                    '<p style="font-family:monospace;font-size:12px;'
                    'color:#000000 !important;margin-top:12px;">'
                    "<strong>Enter</strong> → send &nbsp;|&nbsp; "
                    "<strong>Shift+Enter</strong> → new line"
                    "</p>"
                )

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=500,
                    label="Chat",
                    type="tuples",
                    bubble_full_width=False,
                    avatar_images=None,
                    render_markdown=True,
                )
                msg = gr.Textbox(
                    label="Your question",
                    placeholder="Ask about reviews, complaints, ratings, languages...  (Enter to send, Shift+Enter for new line)",
                    lines=2,
                )
                with gr.Row():
                    submit = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear")

        for btn, example in example_buttons:
            btn.click(lambda ex=example: ex, outputs=msg)

        submit.click(respond, [msg, chatbot], [chatbot]).then(lambda: "", outputs=msg)
        msg.submit(respond, [msg, chatbot], [chatbot]).then(lambda: "", outputs=msg)
        clear.click(on_clear, outputs=[chatbot, msg])

    return demo


def _free_port(port: int) -> None:
    if sys.platform != "win32":
        return
    try:
        import subprocess
        output = subprocess.check_output(
            ["netstat", "-ano"], text=True, encoding="utf-8", errors="replace"
        )
        pids: set[int] = set()
        for line in output.splitlines():
            if f":{port} " not in line and not line.rstrip().endswith(f":{port}"):
                continue
            if "LISTENING" not in line:
                continue
            parts = line.split()
            if parts:
                pids.add(int(parts[-1]))
        for pid in pids:
            if pid > 0:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F"],
                    check=False,
                    capture_output=True,
                )
    except Exception:
        pass


def _resolve_server_port(preferred: int) -> int:
    import socket
    import time

    for _ in range(3):
        _free_port(preferred)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((config.UI_HOST, preferred))
                return preferred
            except OSError:
                pass
        time.sleep(1)
    raise OSError(f"Port {preferred} is still in use.")


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    logging.basicConfig(level=logging.INFO)
    demo = build_app()
    port = _resolve_server_port(config.UI_PORT)
    demo.launch(server_name=config.UI_HOST, server_port=port, share=False)


if __name__ == "__main__":
    main()