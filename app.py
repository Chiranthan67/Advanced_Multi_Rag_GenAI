from __future__ import annotations

import json
from pathlib import Path
import streamlit as st

from main import (
    AdvanceRAG,
    SUPPORTED_AUDIO_SUFFIXES,
    SUPPORTED_IMAGE_SUFFIXES,
    is_audio_transcription_available,
)


st.set_page_config(
    page_title="Advance RAG Studio",
    page_icon="AI",
    layout="wide",
)


def apply_custom_css(theme_mode: str) -> None:
    is_dark = theme_mode.lower() == "dark"

    if is_dark:
        bg_a = "#1d1a17"
        bg_b = "#26221d"
        ink = "#f8efe8"
        ink_soft = "#dcbca3"
        line = "rgba(255, 188, 131, 0.2)"
        panel = "rgba(36, 31, 27, 0.84)"
        side_bg = "rgba(29, 25, 22, 0.88)"
        shadow = "0 8px 26px rgba(0, 0, 0, 0.32)"
    else:
        bg_a = "#fffaf5"
        bg_b = "#ffffff"
        ink = "#322217"
        ink_soft = "#875d44"
        line = "rgba(222, 136, 71, 0.16)"
        panel = "rgba(255, 255, 255, 0.95)"
        side_bg = "rgba(255, 255, 255, 0.94)"
        shadow = "0 8px 22px rgba(173, 109, 65, 0.08)"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {{
            --bg-a: {bg_a};
            --bg-b: {bg_b};
            --ink: {ink};
            --ink-soft: {ink_soft};
            --line: {line};
            --accent: #db6e22;
            --accent-soft: #f8e7d9;
            --panel: {panel};
            --side-bg: {side_bg};
            --shadow: {shadow};
        }}

        body {{
            background-color: #ffffff !important;
            color: #111111 !important;
        }}

        .stApp {{
            font-family: 'Space Grotesk', sans-serif;
            color: var(--ink);
            background:
                radial-gradient(circle at 0% 0%, rgba(219, 110, 34, 0.08), transparent 36%),
                linear-gradient(135deg, var(--bg-a), var(--bg-b));
        }}

        section[data-testid="stSidebar"] {{
            border-right: 1px solid var(--line);
            background: var(--side-bg);
        }}

        button[data-baseweb="tab"] {{
            border-radius: 8px;
            border: 1px solid var(--line);
            background: var(--panel);
            margin-right: 4px;
            font-weight: 500;
            color: var(--ink);
        }}

        button[data-baseweb="tab"][aria-selected="true"] {{
            background: var(--accent);
            color: #fff;
            border-color: var(--accent);
        }}

        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button {{
            border-radius: 8px;
            border: 1px solid var(--accent);
            background: var(--accent);
            color: white;
            font-weight: 600;
        }}

        div[data-testid="stButton"] > button:hover,
        div[data-testid="stDownloadButton"] > button:hover {{
            filter: brightness(0.97);
        }}

        div[data-testid="stChatMessage"] {{
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 6px 8px;
            background: var(--panel);
            box-shadow: var(--shadow);
        }}

        div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] {{
            color: #111111 !important;
        }}

        div[data-testid="stChatInput"] {{
            border-top: 1px solid var(--line);
            padding-top: 0.6rem;
        }}

        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea {{
            border-radius: 8px;
            border: 1px solid var(--line);
        }}

        div[data-testid="stMetric"] {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 10px 12px;
        }}

        @media (max-width: 1024px) {{
            div[data-testid="stMetric"] {{
                padding: 8px 10px;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_engine(
    groq_api_key: str,
    jina_api_key: str,
    model: str,
    vision_model: str,
    chunk_size: int,
    top_k: int,
    rerank_model: str,
) -> AdvanceRAG:
    signature = (groq_api_key, jina_api_key, model, vision_model, chunk_size, top_k, rerank_model)
    if st.session_state.get("engine_signature") != signature:
        st.session_state.engine = AdvanceRAG(
            groq_api_key=groq_api_key,
            jina_api_key=jina_api_key,
            model=model,
            vision_model=vision_model,
            chunk_size=chunk_size,
            top_k=top_k,
            rerank_model=rerank_model,
        )
        st.session_state.engine_signature = signature
        st.session_state.chat_history = []
    return st.session_state.engine


def render_header(engine: AdvanceRAG | None) -> None:
    chunk_count = len(engine.chunks) if engine else 0
    text_count = sum(1 for chunk in engine.chunks if chunk.modality == "text") if engine else 0
    image_count = sum(1 for chunk in engine.chunks if chunk.modality == "image") if engine else 0
    source_count = len({chunk.source for chunk in engine.chunks}) if engine else 0
    index_state = "Ready" if engine and engine.is_ready else "Not Ready"
    modalities = f"T:{text_count} / I:{image_count}" if engine else "T:0 / I:0"
    chat_turns = len(st.session_state.get("chat_history", []))

    st.title("Advance RAG Studio")
    st.caption("Upload files, build the index, and chat with grounded answers from your own content.")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Chunks", str(chunk_count))
    col2.metric("Sources", str(source_count))
    col3.metric("Index", index_state)
    col4.metric("Modalities", modalities)
    col5.metric("Messages", str(chat_turns))
    st.caption(f"Text chunks: {text_count} | Image chunks: {image_count}")


def get_sources(engine: AdvanceRAG) -> list[str]:
    return sorted({chunk.source for chunk in engine.chunks})


def format_context(ctx: dict, idx: int) -> None:
    st.caption(f"Chunk {idx} - {ctx.get('modality', 'text')} - {ctx.get('source', 'source')}")
    st.write(ctx.get("text", ""))
    if ctx.get("modality") == "image" and ctx.get("meta", {}).get("image_b64"):
        st.image(ctx["meta"]["image_b64"], caption=ctx.get("meta", {}).get("image", ""))


def read_secret(key: str) -> str:
    try:
        return str(st.secrets.get(key, ""))
    except Exception:
        return ""


def main() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "theme_dark" not in st.session_state:
        st.session_state.theme_dark = False

    audio_ok, audio_status = is_audio_transcription_available()

    with st.sidebar:
        st.subheader("Control Panel")
        st.session_state.theme_dark = st.toggle(
            "Dark Theme",
            value=st.session_state.theme_dark,
        )
        theme_mode = "Dark" if st.session_state.theme_dark else "Light"

        with st.expander("Credentials", expanded=True):
            default_groq_key = read_secret("GROQ_API_KEY")
            groq_api_key = st.text_input(
                "GROQ API Key",
                type="password",
                value=default_groq_key,
                help="Used in this session. Prefer Streamlit secrets in deployment.",
            )
            default_jina_key = read_secret("JINA_API_KEY")
            jina_api_key = st.text_input(
                "Jina API Key",
                type="password",
                value=default_jina_key,
                help="Used for embeddings. Prefer Streamlit secrets in deployment.",
            )

        with st.expander("Model Settings", expanded=False):
            model = st.selectbox("Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"], index=0)
            vision_model = "meta-llama/llama-4-scout-17b-16e-instruct"
            st.caption(f"Vision model: {vision_model}")
            rerank_model = st.selectbox(
                "Reranker",
                ["cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L-12-v2"],
                index=0,
            )
            chunk_size = st.slider("Chunk Size (words)", min_value=120, max_value=900, value=400, step=20)
            top_k = st.slider("Top K Retrieval", min_value=1, max_value=8, value=3, step=1)
            candidate_k = st.slider("Candidate K (before rerank)", min_value=3, max_value=20, value=8, step=1)
            use_rerank = st.toggle("Use Cross-Encoder Rerank", value=True)
            modality_filter = st.selectbox("Retrieve", ["both", "text", "image"], index=0)
            use_vision = st.toggle("Use Vision for Image Q&A", value=True)
            temperature = st.slider("Answer Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
            memory_turns = st.slider("Memory Turns", min_value=0, max_value=8, value=4, step=1)

        with st.expander("Knowledge Files", expanded=True):
            uploads = st.file_uploader(
                "Upload one or more files",
                type=[
                    "txt",
                    "md",
                    "pdf",
                    "wav",
                    "mp3",
                    "mp4",
                    "m4a",
                    "flac",
                    "ogg",
                    "png",
                    "jpg",
                    "jpeg",
                    "webp",
                ],
                accept_multiple_files=True,
            )

            process_files = st.button("Process + Build Index", use_container_width=True, type="primary")
            rebuild_index = st.button("Rebuild Index", use_container_width=True)
            clear_kb = st.button("Clear Knowledge Base", use_container_width=True)

    apply_custom_css(theme_mode)

    engine = None
    if groq_api_key and jina_api_key:
        try:
            engine = get_engine(
                groq_api_key,
                jina_api_key,
                model,
                vision_model,
                chunk_size,
                top_k,
                rerank_model,
            )
        except Exception as exc:
            st.error(f"Unable to initialize engine: {exc}")
            return

    if clear_kb and engine:
        engine.reset()
        st.session_state.chat_history = []
        st.success("Knowledge base cleared.")
    elif rebuild_index and engine:
        if not engine.chunks:
            st.warning("No chunks found. Upload and process files first.")
        else:
            with st.spinner("Rebuilding index..."):
                engine.build_index()
            st.success("Index rebuilt successfully.")

    render_header(engine)

    if not audio_ok:
        st.caption(f"Audio note: {audio_status} Audio files will be skipped.")

    if process_files:
        if not groq_api_key or not jina_api_key:
            st.warning("Add both GROQ and Jina API keys before processing files.")
        elif not uploads:
            st.warning("Upload at least one file first.")
        else:
            added = 0
            with st.spinner("Processing files and building index..."):
                for file in uploads:
                    file_name = file.name
                    suffix = Path(file_name).suffix.lower()
                    data = file.getvalue()

                    try:
                        if suffix in {".txt", ".md"}:
                            added += engine.ingest_txt_bytes(data, source=file_name)
                        elif suffix == ".pdf":
                            added += engine.ingest_pdf_bytes(data, source=file_name)
                        elif suffix in SUPPORTED_IMAGE_SUFFIXES:
                            added += engine.ingest_image_bytes(data, source=file_name)
                        elif suffix in SUPPORTED_AUDIO_SUFFIXES:
                            if not audio_ok:
                                st.warning(f"Skipped {file_name}: {audio_status}")
                                continue
                            transcript = engine.transcribe_audio_bytes(data, file_name, whisper_model="base")
                            if transcript:
                                added += engine.ingest_text(transcript, source=f"{file_name} (transcript)")
                        else:
                            st.info(f"Skipped unsupported file: {file_name}")
                    except Exception as exc:
                        st.error(f"Failed to process {file_name}: {exc}")

                if engine.chunks:
                    engine.build_index()
                    st.success(f"Indexed successfully. Added {added} chunks from {len(uploads)} file(s).")
                else:
                    st.warning("No usable text found in the uploaded files.")

    if not groq_api_key or not jina_api_key:
        st.info("Enter your GROQ + Jina API keys in the sidebar to start.")

    tab_chat, tab_library, tab_retrieval = st.tabs(["Assistant", "Library", "Inspector"])

    with tab_chat:
        st.subheader("Assistant")
        st.caption("Ask grounded questions against the indexed knowledge base.")
        quick_cols = st.columns(3)
        prompts = [
            "Summarize the key topics from my documents.",
            "List important facts with source references.",
            "What are the main action items?",
        ]
        for i, prompt in enumerate(prompts):
            if quick_cols[i].button(prompt, use_container_width=True):
                st.session_state["pending_query"] = prompt

        export_payload = json.dumps(st.session_state.chat_history, indent=2)
        st.download_button(
            "Download Chat History (JSON)",
            data=export_payload,
            file_name="rag_chat_history.json",
            mime="application/json",
            use_container_width=True,
        )

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("context"):
                    with st.expander("Retrieved Context"):
                        for idx, ctx in enumerate(message["context"], start=1):
                            format_context(ctx, idx)

        input_default = st.session_state.pop("pending_query", "")
        user_query = st.chat_input(
            "Ask a question about your uploaded knowledge base...",
            key="chat_input",
        )
        if not user_query and input_default:
            user_query = input_default

        if user_query:
            if not engine or not engine.is_ready:
                st.warning("Process files first so the FAISS index is ready.")
            else:
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)

                with st.chat_message("assistant"):
                    with st.spinner("Generating grounded answer..."):
                        try:
                            candidate_k = max(candidate_k, top_k)
                            memory = []
                            if memory_turns > 0:
                                memory = st.session_state.chat_history[:-1]
                                memory = memory[-(memory_turns * 2) :]
                            result = engine.answer(
                                user_query,
                                top_k=top_k,
                                candidate_k=candidate_k,
                                modality_filter=modality_filter,
                                use_rerank=use_rerank,
                                use_vision=use_vision,
                                memory=memory,
                                temperature=temperature,
                            )
                            answer = result["answer"]
                            context_docs = result["context_docs"]
                            latency = result.get("latency", {})
                        except Exception as exc:
                            answer = f"Error while generating answer: {exc}"
                            context_docs = []
                            latency = {}

                    st.markdown(answer)
                    if context_docs:
                        with st.expander("Retrieved Context"):
                            for idx, ctx in enumerate(context_docs, start=1):
                                format_context(ctx, idx)

                    if latency:
                        with st.expander("Latency Breakdown"):
                            for key, value in latency.items():
                                st.write(f"{key}: {value:.2f}s")

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer, "context": context_docs}
                )

    with tab_library:
        st.subheader("Source Library")
        st.caption("Browse indexed files, chunk counts, and remove sources when needed.")
        if not engine or not engine.chunks:
            st.info("Upload and process files to explore sources.")
        else:
            source_stats = engine.source_stats()
            stats_rows = [
                {
                    "source": src,
                    "chunks": val["chunks"],
                    "text": val["text"],
                    "image": val["image"],
                }
                for src, val in source_stats.items()
            ]
            st.dataframe(stats_rows, use_container_width=True, hide_index=True)

            remove_targets = st.multiselect("Remove source(s)", sorted(source_stats.keys()))
            if st.button("Remove Selected Sources", type="primary", use_container_width=True):
                if not remove_targets:
                    st.warning("Select at least one source.")
                else:
                    with st.spinner("Removing sources and rebuilding index..."):
                        removed_count = engine.remove_sources(remove_targets)
                    st.session_state.chat_history = []
                    if removed_count:
                        st.success(f"Removed {removed_count} chunk(s) from {len(remove_targets)} source(s).")
                    else:
                        st.info("No matching sources were removed.")

            sources = get_sources(engine)
            selected_source = st.selectbox("Source", ["All"] + sources, index=0)
            selected_modality = st.selectbox("Modality", ["both", "text", "image"], index=0)

            filtered = []
            for chunk in engine.chunks:
                if selected_source != "All" and chunk.source != selected_source:
                    continue
                if selected_modality != "both" and chunk.modality != selected_modality:
                    continue
                filtered.append(chunk)

            st.caption(f"Showing {len(filtered)} chunk(s)")
            for idx, chunk in enumerate(filtered[:50], start=1):
                with st.expander(f"Chunk {idx} - {chunk.source} - {chunk.modality}"):
                    st.write(chunk.text)
                    if chunk.modality == "image" and chunk.meta.get("image_b64"):
                        st.image(chunk.meta["image_b64"], caption=chunk.meta.get("image", chunk.source))
            if len(filtered) > 50:
                st.info("Showing first 50 chunks for performance.")

    with tab_retrieval:
        st.subheader("Retrieval Inspector")
        st.caption("Run direct retrieval tests to inspect what the model sees before generation.")
        if not engine or not engine.is_ready:
            st.info("Process files to enable retrieval inspection.")
        else:
            inspect_query = st.text_input("Test a retrieval query")
            inspect_mode = st.selectbox("Inspect modality", ["both", "text", "image"], index=0)
            if st.button("Run Retrieval Test", type="primary", use_container_width=True):
                if not inspect_query.strip():
                    st.warning("Enter a query first.")
                else:
                    try:
                        candidates = engine.retrieve(
                            inspect_query.strip(),
                            top_k=max(candidate_k, top_k),
                            modality_filter=inspect_mode,
                        )
                        final_docs = candidates
                        if use_rerank:
                            final_docs = engine.rerank(inspect_query.strip(), candidates, top_k=top_k)
                    except Exception as exc:
                        st.error(f"Retrieval test failed: {exc}")
                        candidates = []
                        final_docs = []

                    st.write(f"Candidates: {len(candidates)}")
                    st.write(f"Final after rerank: {len(final_docs)}")
                    for idx, doc in enumerate(final_docs, start=1):
                        with st.expander(f"Result {idx} - {doc.source} - {doc.modality}"):
                            st.write(doc.text)
                            if doc.modality == "image" and doc.meta.get("image_b64"):
                                st.image(doc.meta["image_b64"], caption=doc.meta.get("image", doc.source))


if __name__ == "__main__":
    main()
