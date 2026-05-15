@echo off
setlocal

REM Optional: install Hugging Face CLI if not installed
python -m pip install -U "huggingface_hub[cli]"

REM Create target folder
if not exist indexes mkdir indexes

huggingface-cli download Tevatron/browsecomp-plus-indexes --repo-type=dataset --include="bm25/*" --local-dir indexes
huggingface-cli download Tevatron/browsecomp-plus-indexes --repo-type=dataset --include="qwen3-embedding-0.6b/*" --local-dir indexes
huggingface-cli download Tevatron/browsecomp-plus-indexes --repo-type=dataset --include="qwen3-embedding-4b/*" --local-dir indexes
huggingface-cli download Tevatron/browsecomp-plus-indexes --repo-type=dataset --include="qwen3-embedding-8b/*" --local-dir indexes

echo.
echo Done. Files downloaded into .\indexes
pause