# mantis_router

`mantis_router.gateway.dispatch(request)` executes one routed step and returns
a response envelope.

## Supported operations

- `echo`: return `value` unchanged
- `upper`: return `str(value).upper()`
- `count`: return `len(str(value))`

Requests must be dicts with an `op` key. Unsupported ops raise `ValueError`.
Requests without an `op` key are rejected with a `{"status": "error", ...}`
envelope.

## Example

```python
>>> from mantis_router import dispatch
>>> dispatch({"op": "upper", "value": "hi"})
{'status': 'ok', 'result': 'HI'}
```
