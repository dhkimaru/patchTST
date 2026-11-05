"""ALD Lab dashboard entry point.

Run from the ``ald-lab`` folder::

    streamlit run src/ald_lab/dashboard/app.py -- --work-root work
"""

from __future__ import annotations

import streamlit as st

from ald_lab.dashboard.state import setup_page
from ald_lab.dashboard.views import data_overview, home, preprocessing, timeline, training

st.set_page_config(page_title="ALD Lab", layout="wide")
setup_page()

pages = [
    st.Page(home.render, title="현황", default=True),
    st.Page(data_overview.render, title="1. 원본 데이터", url_path="data"),
    st.Page(preprocessing.render, title="2. 전처리", url_path="preprocess"),
    st.Page(training.render, title="3. 하이퍼파라미터·학습", url_path="training"),
    st.Page(timeline.render, title="4. 타임라인·결과", url_path="timeline"),
]
st.navigation(pages).run()
