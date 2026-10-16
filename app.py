from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from src.chasm_rag.services.document_service import process_document
from src.chasm_rag.services.qa_service import answer_question
from src.chasm_rag.services.summary_service import create_summary
from src.chasm_rag.evidence.evidence_models import EvidenceType
from src.chasm_rag.evidence.measurement_validator import classify_measurement, is_elevated
from src.chasm_rag.utils.exceptions import ChasmError

st.set_page_config(page_title="CHASM-RAG", layout="wide")
st.title("CHASM-RAG")
st.caption("Research/education only. This tool extracts document evidence; it does not diagnose or prescribe treatment.")

PROCESSING_VERSION = "visual-layout-extraction-v4"


@st.cache_resource(show_spinner=False)
def _process(file_bytes: bytes, filename: str, processing_version: str):
    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
        temporary.write(file_bytes)
        return process_document(temporary.name)


def _display_measurement_name(name: str) -> str:
    return "Total cholesterol" if name.casefold() == "cholesterol total" else name


def _render_rag_answer(question: str, answer) -> None:
    """Render a concise RAG answer with citations and expandable evidence."""
    st.subheader("Answer")

    if answer.uncertainty:
        st.warning(answer.uncertainty)

    # Deduplicate measurements while preserving source order.
    measurements = []
    seen = set()

    for result in answer.evidence:
        item = result.evidence
        measurement = item.measurement

        if measurement is None:
            continue

        key = (
            item.document_id,
            item.evidence_id,
            measurement.test_name,
            measurement.value,
            measurement.unit,
        )

        if key not in seen:
            seen.add(key)
            measurements.append((item, measurement))

    normalized_question = question.casefold()
    asks_abnormal = any(
        term in normalized_question
        for term in (
            "outside",
            "abnormal",
            "elevated",
            "high",
            "low",
            "out of range",
        )
    ) and any(
        term in normalized_question
        for term in ("range", "value", "result", "measurement", "test")
    )

    if measurements and asks_abnormal:
        abnormal = [
            (source, item)
            for source, item in measurements
            if classify_measurement(item).startswith(("HIGH", "LOW"))
        ]

        within = [
            (source, item)
            for source, item in measurements
            if classify_measurement(item) == "WITHIN SUPPLIED RANGE"
        ]

        unassessed = [
            (source, item)
            for source, item in measurements
            if classify_measurement(item) == "NOT ASSESSED"
        ]

        if abnormal:
            st.markdown("**Results outside the supplied reference ranges**")
            display_items = abnormal
        elif within and not unassessed:
            st.info(
                "No retrieved measurement was classified as outside "
                "its supplied reference range."
            )
            display_items = within
        else:
            st.info(
                "The retrieved evidence is insufficient to establish "
                "which results are outside their supplied reference ranges."
            )
            display_items = []

        if display_items:
            rows = []

            for source, item in display_items:
                reference = (
                    item.reference_range.raw
                    if item.reference_range and item.reference_range.raw
                    else "Not supplied"
                )

                rows.append({
                    "Test": _display_measurement_name(item.test_name),
                    "Reported value": (
                        f"{item.value:g} {item.unit or ''}".strip()
                        if item.value is not None
                        else "Not reported"
                    ),
                    "Reference range": reference,
                    "Classification": classify_measurement(item),
                    "Reported flag": item.reported_flag or "Not stated",
                    "Page": source.page,
                })

            st.dataframe(
                rows,
                use_container_width=True,
                hide_index=True,
            )

        if unassessed:
            st.warning(
                "Reference-range status could not be established for: "
                + ", ".join(
                    _display_measurement_name(item.test_name)
                    for _, item in unassessed
                )
            )

        # Surface source consistency warnings without changing reported values.
        warnings = []
        for _, item in measurements:
            for warning in item.consistency_warnings:
                message = f"{item.test_name}: {warning}"
                if message not in warnings:
                    warnings.append(message)

        for warning in warnings:
            st.warning(f"Source consistency warning — {warning}")

    else:
        # For general questions, retain the answer produced by the QA service.
        st.markdown(answer.answer or "No answer was generated.")

    st.markdown("**Sources**")

    cited = set()
    source_count = 0

    for result in answer.evidence:
        item = result.evidence

        if not item.document_id or not item.evidence_id or item.page <= 0:
            continue

        key = (item.document_id, item.page, item.evidence_id)
        if key in cited:
            continue

        cited.add(key)
        source_count += 1

        st.caption(
            f"Page {item.page} · "
            f"Document {item.document_id} · "
            f"Chunk {item.evidence_id}"
        )

    if source_count == 0:
        st.caption("No valid source citations are available.")

    with st.expander("View retrieved evidence"):
        if not answer.evidence:
            st.info("No evidence was retrieved.")

        for index, result in enumerate(answer.evidence, start=1):
            item = result.evidence

            st.markdown(
                f"**Evidence {index} · Page {item.page} · "
                f"{item.evidence_type.value}**"
            )

            if item.measurement:
                measurement = item.measurement
                st.caption(
                    f"{_display_measurement_name(measurement.test_name)} · "
                    f"{classify_measurement(measurement)}"
                )

            st.code(item.original_text)

            st.caption(
                f"Retrieval score: {result.score:.3f} · "
                f"Channels: {', '.join(result.channels)}"
            )



uploaded = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])
if uploaded:
    try:
        with st.spinner("Extracting source evidence and building the document-local index…"):
            document = _process(uploaded.getvalue(), uploaded.name, PROCESSING_VERSION)
        st.success(f"Processed {uploaded.name}: {len(document.extracted.pages)} page(s), {len(document.store.evidence)} evidence unit(s).")
        warnings = document.extracted.warnings + [warning for page in document.extracted.pages for warning in page.warnings]
        if warnings:
            st.warning("\n".join(sorted(set(warnings))))
        with st.expander("Extracted document text"):
            for page in document.extracted.pages:
                st.caption(f"Page {page.page} · extraction: {page.method}")
                st.text(page.text or "No text extracted from this page.")
        measurements = document.store.measurements
        if measurements:
            st.subheader("Structured measurements")
            measurement_rows = [{"Test": item.test_name, "Value": item.value, "Unit": item.unit, "Reference": item.reference_range.raw if item.reference_range else None, "Status": classify_measurement(item), "Reported flag": item.reported_flag or "—", "Page": item.page, "Warnings": "; ".join(item.consistency_warnings) or "—"} for item in measurements]
            st.dataframe(measurement_rows, use_container_width=True, hide_index=True)
            elevated = [item for item in measurements if is_elevated(item)]
            st.subheader("Results marked elevated")
            if elevated:
                st.warning("These results are marked high by the report or are above its supplied reference threshold. This is not a diagnosis.")
                st.dataframe([{"Test": item.test_name, "Result": f"{item.value:g} {item.unit or ''}".strip(), "Supplied reference": item.reference_range.raw if item.reference_range else "Not supplied", "Why shown": classify_measurement(item), "Page": item.page} for item in elevated], use_container_width=True, hide_index=True)
            else:
                st.success("No extracted result is marked high or above its supplied reference threshold.")
            st.subheader("Follow-up and recommendations")
            st.info("For elevated or uncertain results, review the original report with a qualified clinician. CHASM-RAG does not provide treatment, medication, or diagnostic recommendations.")
            source_guidance = [item for item in document.store.evidence if item.evidence_type == EvidenceType.INSTRUCTION and len(item.original_text.strip()) > 20]
            if source_guidance:
                st.caption("Source-provided follow-up text (shown verbatim; not generated by the system):")
                for item in source_guidance:
                    st.write(f"p. {item.page}: {item.original_text}")
            else:
                st.caption("The report does not contain an extracted follow-up instruction. No recommendation has been inferred.")
        else:
            st.info("No report-shaped measurements were extracted. Inspect the source evidence below.")
        if st.button("Generate validated summary", type="primary"):
            with st.spinner("Creating and verifying a hierarchical summary…"):
                result = create_summary(document)
            st.subheader(
                "Validated summary"
                if result.output.is_validated
                else "Conservative source-based summary"
            )
            if result.output.used_fallback:
                st.info(
                    "Automatic checks could not verify the generated draft. "
                    "This summary is based on information extracted from the report."
                )
            st.markdown(result.output.text)
            if result.output.reason:
                with st.expander("Automatic check details"):
                    st.caption(result.output.reason)
            with st.expander("Claim verification"):
                st.dataframe([{"Claim": item.claim.text, "Status": item.status.value, "Evidence IDs": ", ".join(item.evidence_ids), "Warnings": "; ".join(item.warnings)} for item in result.claims], use_container_width=True)
        st.subheader("Ask about this document")
        question = st.text_input("Question", placeholder="Which values are outside their supplied reference ranges?")
        
        if st.button("Answer from evidence") and question.strip():
            with st.spinner("Retrieving and verifying evidence…"):
                answer = answer_question(document, question.strip())
            _render_rag_answer(question.strip(), answer)

            if answer.uncertainty:
                st.warning(answer.uncertainty)
            st.markdown("**Retrieved evidence**")
            for result in answer.evidence:
                item = result.evidence
                if item.measurement:
                    measurement = item.measurement
                    status = f"; {measurement.reported_flag.title()}" if measurement.reported_flag else ""
                    st.markdown(f"- **p. {item.page} — {_display_measurement_name(measurement.test_name)}:** {measurement.value:g} {measurement.unit or ''}; reference: {measurement.reference_range.raw if measurement.reference_range else 'not supplied'}{status}.")
                else:
                    st.caption(f"Page {item.page} · {item.evidence_type.value} · score {result.score:.3f} · {', '.join(result.channels)}")
                    st.code(item.original_text)
        with st.expander("Inspect extracted source evidence"):
            for item in document.store.evidence:
                st.markdown(f"**p. {item.page} · {item.evidence_type.value} · {item.evidence_id}**")
                st.code(item.original_text)
    except (ChasmError, ValueError) as exc:
        st.error(str(exc))
