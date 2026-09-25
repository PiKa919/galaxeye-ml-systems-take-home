"""Small local Streamlit viewer for models, tile predictions, and reports."""

import httpx
import pandas as pd
import streamlit as st

from galaxeye.contracts import CLASSES
from galaxeye.settings import API_URL

st.set_page_config(page_title="GalaxEye Tile Lab", page_icon="🛰️", layout="wide")
st.title("GalaxEye Tile Lab")
st.caption("Local land-use tile classification · YOLO26n v1, MobileNetV3 v2, YOLO26s v3")


def api_get(path: str, **params):
    response = httpx.get(f"{API_URL}{path}", params=params, timeout=10)
    response.raise_for_status()
    return response.json()


try:
    models = api_get("/models")
    health = api_get("/health")
except httpx.HTTPError as exc:
    st.error(f"Local API unavailable: {exc}")
    st.stop()

if not health["inference"].get("ready"):
    st.warning(
        "The CPU inference process is not ready. Start it before classifying a tile."
    )

version_labels = {
    version: f"{version} · {manifest['architecture']}"
    for version, manifest in models.items()
}
version_options = {"All versions": tuple(models)}
version_options.update(
    {version_labels[version]: (version,) for version in models}
)

tab_upload, tab_history, tab_reports = st.tabs(
    ["Classify a tile", "Saved predictions", "Model reports"]
)

with tab_upload:
    left, right = st.columns([1, 2])
    with left:
        uploaded = st.file_uploader(
            "Choose a 64×64 RGB PNG tile", type=["png"], max_upload_size=2
        )
        selection = st.radio(
            "Run", list(version_options), horizontal=True
        )
        if uploaded:
            st.image(uploaded, caption=uploaded.name, width=256)
        run = st.button("Classify and save", type="primary", disabled=uploaded is None)
    with right:
        if run and uploaded is not None:
            raw = uploaded.getvalue()
            versions = version_options[selection]
            results = []
            for version in versions:
                try:
                    response = httpx.post(
                        f"{API_URL}/classify",
                        params={"version": version},
                        files={"file": (uploaded.name, raw, "image/png")},
                        timeout=130,
                    )
                    response.raise_for_status()
                    results.append(response.json())
                except httpx.HTTPError as exc:
                    detail = (
                        exc.response.text
                        if isinstance(exc, httpx.HTTPStatusError)
                        else str(exc)
                    )
                    st.error(f"{version} failed: {detail}")
            st.session_state["latest_results"] = results
        for result in st.session_state.get("latest_results", []):
            st.subheader(
                f"{result['version']} · {models[result['version']]['architecture']}"
            )
            st.metric("Predicted class", result["predicted_class"])
            st.write(
                "Review required" if result["review_required"] else "No review flag"
            )
            st.caption(
                f"Stored result #{result['id']} · CPU inference {result['inference_ms']:.1f} ms"
            )
            st.bar_chart(pd.Series(result["scores"], name="model score"))
        if not st.session_state.get("latest_results"):
            st.info("Upload a tile to compare the saved model versions.")

with tab_history:
    a, b, c = st.columns(3)
    version_filter = a.selectbox("Version", ["All", *models])
    class_filter = b.selectbox("Predicted class", ["All", *CLASSES])
    review_filter = c.selectbox("Review", ["All", "Required", "Not required"])
    params = {"limit": 100}
    if version_filter != "All":
        params["version"] = version_filter
    if class_filter != "All":
        params["predicted_class"] = class_filter
    if review_filter != "All":
        params["review_required"] = review_filter == "Required"
    try:
        history = api_get("/predictions", **params)
    except httpx.HTTPError as exc:
        st.error(f"Could not load saved predictions: {exc}")
        history = []
    if history:
        rows = [
            {
                "id": result["id"],
                "version": result["version"],
                "class": result["predicted_class"],
                "top score": max(result["scores"].values()),
                "review": result["review_required"],
                "saved at": result["created_at_utc"],
            }
            for result in history
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        chosen = st.selectbox(
            "Inspect saved result", [result["id"] for result in history]
        )
        result = next(result for result in history if result["id"] == chosen)
        preview = httpx.get(f"{API_URL}/images/{result['image_sha256']}", timeout=10)
        if preview.is_success:
            st.image(preview.content, width=256)
        else:
            st.error("The saved image file could not be read.")
        st.json(result)
    else:
        st.info("No saved predictions match these filters.")

with tab_reports:
    try:
        reports = api_get("/reports")
    except httpx.HTTPError as exc:
        st.error(f"Could not load model reports: {exc}")
        reports = {}
    if not reports:
        st.info("Run the evaluation command to create saved model reports.")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "version": version,
                        "model": report["architecture"],
                        "validation": f"{report['validation']['accuracy']:.1%}",
                        "internal test": f"{report['internal_test']['accuracy']:.1%}",
                        "supplied eval": f"{report['supplied_eval']['accuracy']:.1%}",
                        "CPU p50": f"{report['internal_test']['latency_ms_p50']:.1f} ms",
                    }
                    for version, report in reports.items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    for version, report in reports.items():
        with st.expander(f"Details · {version} · {report['architecture']}"):
            columns = st.columns(4)
            columns[0].metric(
                "Validation accuracy", f"{report['validation']['accuracy']:.1%}"
            )
            columns[1].metric(
                "Internal test accuracy", f"{report['internal_test']['accuracy']:.1%}"
            )
            columns[2].metric(
                "Supplied eval accuracy", f"{report['supplied_eval']['accuracy']:.1%}"
            )
            columns[3].metric(
                "CPU p50", f"{report['internal_test']['latency_ms_p50']:.1f} ms"
            )
            st.caption(
                "Scores and accuracy are measured on small fixed sets. The review threshold was selected on validation only."
            )
            st.write("Per-class internal test results")
            st.dataframe(
                pd.DataFrame(report["internal_test"]["by_class"]).T.loc[
                    list(CLASSES), ["precision", "recall", "f1-score", "support"]
                ],
                width="stretch",
            )
            st.write(
                "Internal test confusion matrix: rows are true class, columns are predicted class"
            )
            st.dataframe(
                pd.DataFrame(
                    report["internal_test"]["confusion_matrix"],
                    index=CLASSES,
                    columns=CLASSES,
                ),
                width="stretch",
            )
            st.json(report["review_policy"])
