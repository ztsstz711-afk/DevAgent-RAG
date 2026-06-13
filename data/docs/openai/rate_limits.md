# OpenAI Rate Limits

OpenAI API rate limits protect service reliability. Handle HTTP 429 responses with exponential backoff and jitter. Keep requests within request-per-minute and token-per-minute limits, and inspect response headers for current limits.

## Python retry example

```python
import random
import time

for attempt in range(5):
    try:
        response = client.responses.create(model="gpt-4.1-mini", input="Hello")
        break
    except RateLimitError:
        time.sleep((2 ** attempt) + random.random())
```
