"""JSON export with explicit null for undefined numeric quantities."""
import json
import math
import numbers
def clean(obj):
    if isinstance(obj, dict): return {str(k): clean(v) for k,v in obj.items()}
    if isinstance(obj, (list,tuple)): return [clean(v) for v in obj]
    if isinstance(obj, numbers.Integral): return int(obj) if not isinstance(obj, bool) else obj
    if isinstance(obj, numbers.Real) and not isinstance(obj, (bool,int)):
        return float(obj) if math.isfinite(float(obj)) else None
    if hasattr(obj, 'tolist'): return clean(obj.tolist())
    return obj
def dumps(obj):
    return json.dumps(clean(obj), indent=2, allow_nan=False, default=str)
