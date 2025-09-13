while true; do
    nohup streamlit run discourse_parser_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless false > streamlit.log 2>&1
    echo "Streamlit app crashed or was stopped. Restarting in 5 seconds..." >&2
    sleep 5
done