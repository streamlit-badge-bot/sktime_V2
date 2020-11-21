mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"mt.ulas.yilmaz@gmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
