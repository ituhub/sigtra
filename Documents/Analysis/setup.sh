mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"malik.ali642@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
port = \$PORT\n\
enableCORS = false\n\
" > ~/.streamlit/config.toml