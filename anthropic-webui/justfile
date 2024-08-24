run:
    docker compose up --build

chat:
    curl -X POST "http://0.0.0.0:8000/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d @testdata/request.json \
    | grep --line-buffered 'data: ' \
    | sed -u 's/data: //' \
    | jq -r '.choices[0].delta.content // empty' \
    | tr -d '\n'
