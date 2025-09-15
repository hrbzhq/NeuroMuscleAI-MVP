import os
import streamlit as st
import json
from datetime import datetime

# Simple Streamlit app to submit results (predictions/metrics) to submissions/ directory

st.title('Submit results - NeuroMuscleAI')

st.markdown('Use this page to submit prediction files and metrics. Files will be saved locally to `submissions/` and can be reviewed by maintainers.')

username = st.text_input('GitHub username')
email = st.text_input('Email (optional)')
project = st.text_input('Model/Project name', 'resnet18')
notes = st.text_area('Notes: preprocessing, training params, dataset split')
contrib_type = st.selectbox('Contribution type', ['software', 'medical', 'dataset', 'model', 'other'])

pred_file = st.file_uploader('Predictions CSV (optional)', type=['csv'])
metrics_file = st.file_uploader('Metrics JSON (recommended)', type=['json'])

if st.button('Submit'):
    if not username:
        st.error('Please provide a GitHub username')
    else:
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        subdir = os.path.join('submissions', f"{ts}_{username}")
        os.makedirs(subdir, exist_ok=True)
        meta = {
            'username': username,
            'email': email,
            'project': project,
            'contribution_type': contrib_type,
            'notes': notes,
            'timestamp': ts
        }
        with open(os.path.join(subdir, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        if pred_file is not None:
            with open(os.path.join(subdir, 'predictions.csv'), 'wb') as f:
                f.write(pred_file.getbuffer())
        if metrics_file is not None:
            with open(os.path.join(subdir, 'metrics.json'), 'wb') as f:
                f.write(metrics_file.getbuffer())
        st.success(f'Submission saved to {subdir}')
        st.markdown('Maintainers can review `submissions/` folder and optionally add entries to leaderboard or create releases.')
