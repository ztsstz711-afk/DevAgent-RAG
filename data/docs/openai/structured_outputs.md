# OpenAI Structured Outputs

Structured Outputs constrain model responses to a supplied JSON schema. Define required fields and parse the validated response instead of extracting values from free-form text.

```python
response = client.responses.parse(model="gpt-4.1-mini", input=prompt, text_format=ResultSchema)
```
