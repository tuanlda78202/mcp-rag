curl -X POST "http://localhost:8000/api/v1/search" \
    -H "Search-Key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" \
    -H "Content-Type: application/json" \
    -d '{"query": "phòng chống rửa tiền", "match_threshold": 0.5, "match_count": 5}'
